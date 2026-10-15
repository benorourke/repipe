from __future__ import annotations

import abc
import enum
import inspect
import itertools
import pickle
import shutil
import uuid
from abc import abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast, overload

import polars as pl
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from pydantic import BaseModel, model_validator

from repipe.asset.partition import RESERVED_VALUE, PartitionKey, PartitionSettings

if TYPE_CHECKING:
    from repipe.asset.asset import Asset

type LGBMReturnType = LGBMClassifier | LGBMRegressor | LGBMRanker


class StorageType(enum.StrEnum):
    CSV = "CSV"
    PARQUET = "PARQUET"
    ARROW = "ARROW"
    LGBM = "LGBM"


class Mode(enum.StrEnum):

    OVERWRITE = "overwrite"
    """ Once the run succeeds, deletes everything it didn't write - the whole asset, or for a
        partitioned asset, each partition covered by `PartitionSettings.overwrite_by`. A failed
        run changes nothing. """

    APPEND = "append"
    """ Just writes a new pending part - nothing is deleted or compacted. Parts accumulate,
        pending alongside any existing canonical set, until something calls `consolidate()`. """

    APPEND_THEN_MERGE = "append_then_merge"
    """ Like APPEND, but `materialize()` calls `consolidate()` immediately afterwards: every
        pending part just written is folded into the canonical set, deduped via
        `primary_key`/`is_duplicate`, capped at `max_rows_per_file` (unset = one file). """


class BaseStorageConfig(BaseModel):
    """ Everything inheriting StorageConfig must have optionals as the other fields """
    model_config = {"arbitrary_types_allowed": True}

    mode: Mode = Mode.OVERWRITE
    is_duplicate: pl.Expr | None = None
    max_rows_per_file: int | None = None
    """ Caps each file `consolidate()` writes to, via polars' native partitioned sink.
        `None` (default) merges everything into a single file instead. """

    def check_compatible(self, inner_func: Callable[..., Any], partition: PartitionSettings | None) -> None:
        """ Hook for a storage backend to reject an asset function shape it can't represent. """


class ParquetStorageConfig(BaseStorageConfig):
    pass


class ArrowStorageConfig(BaseStorageConfig):
    pass


class CsvStorageConfig(BaseStorageConfig):
    pass


class LGBMStorageConfig(BaseStorageConfig):
    """ A trained model has no rows, so none of `BaseStorageConfig`'s part-compaction fields apply. """

    @model_validator(mode="after")
    def _check_fields_are_compatible_with_a_model(self) -> "LGBMStorageConfig":
        if self.mode != Mode.OVERWRITE:
            raise ValueError(
                f"LGBMStorageConfig only supports Mode.OVERWRITE, got {self.mode!r} - "
                f"a model has no parts to append or merge"
            )
        if self.is_duplicate is not None:
            raise ValueError("LGBMStorageConfig doesn't support is_duplicate - a model has no rows to dedupe")
        if self.max_rows_per_file is not None:
            raise ValueError("LGBMStorageConfig doesn't support max_rows_per_file - a model isn't rows")
        return self

    def check_compatible(self, inner_func: Callable[..., Any], partition: PartitionSettings | None) -> None:
        # A partitioned model asset yields one keyed `Output` per partition, so only an
        # unpartitioned one has to return a single model.
        if partition is None and (inspect.isgeneratorfunction(inner_func) or inspect.isasyncgenfunction(inner_func)):
            raise ValueError(
                f"asset {inner_func.__name__!r} is a generator but uses LGBMStorageConfig - "
                f"a trained model has no per-chunk parts to write; return a single model instead"
            )


type StorageConfig = ParquetStorageConfig | ArrowStorageConfig | CsvStorageConfig | LGBMStorageConfig


def make_default_storage_config(storage_type: StorageType) -> StorageConfig:
    match storage_type:
        case StorageType.PARQUET:
            return ParquetStorageConfig()
        case StorageType.ARROW:
            return ArrowStorageConfig()
        case StorageType.CSV:
            return CsvStorageConfig()
        case StorageType.LGBM:
            return LGBMStorageConfig()
        case _:
            raise NotImplementedError()


def staging_root(base_storage_path: Path) -> Path:
    """ Where runs stage their writes before committing them. Outside every asset directory,
    since stray files inside one break scans, and under `base_storage_path` so the commit's
    renames stay on one filesystem. """
    return base_storage_path / ".repipe" / "staging"


def staging_dir(base_storage_path: Path, run_id: str) -> Path:
    return staging_root(base_storage_path) / run_id


class StorageConcrete[LoadType, SaveType, ConfigType](abc.ABC):
    """ Writes happen in two steps: `stage` writes a chunk into a staging directory, then
    `promote` moves everything staged into the asset directory as `run-{run_id}-part-{n}`.
    A partitioned asset has the same layout, one level per partition column. """

    # A plain class attribute, not an abstract `@property` - each subclass sets this directly
    # (`extension = "parquet"`), which pyright can only check against a plain attribute.
    extension: str

    @property
    @abstractmethod
    def config_type(self) -> type[ConfigType]:
        raise NotImplementedError()

    @abstractmethod
    def load(self, config: ConfigType, asset: Asset) -> LoadType:
        raise NotImplementedError()

    @abstractmethod
    def load_previous(self, config: ConfigType, asset: Asset) -> LoadType | None:
        """ A snapshot of what is on disk for `asset` right now, or `None` if nothing has been
        written yet. Fixed to the parts that exist at call time - unlike `load`, which may
        resolve lazily and so would pick up (or lose) parts written during this run. """
        raise NotImplementedError()

    @abstractmethod
    def consolidate(self, config: ConfigType, asset: Asset, partition: PartitionKey | None = None) -> None:
        """ Compacts (and, per `config`, dedupes) the part files of each partition matching
        `partition`, or of the whole asset if it isn't partitioned. """
        raise NotImplementedError()

    @abstractmethod
    def stage(
            self, config: ConfigType, asset: Asset, obj: SaveType, directory: Path, chunk_number: int,
            partition: Mapping[str, Any] | None = None,
    ) -> set[PartitionKey]:
        """ Writes `obj` under `directory` and returns the partitions it wrote to. `partition`
        is an explicit key, for data that can't be split by its columns (e.g. a model). """
        raise NotImplementedError()

    def save(
            self, config: ConfigType, asset: Asset, obj: SaveType, run_id: str, part_number: int,
            partition: Mapping[str, Any] | None = None,
    ) -> None:
        """ Stages and promotes a single chunk straight away, outside of any write session. """
        directory = staging_dir(asset.base_storage_path, uuid.uuid4().hex)
        try:
            self.stage(config, asset, obj, directory, 1, partition)
            self.promote(asset, directory, run_id, first_number=part_number)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def promote(self, asset: Asset, directory: Path, run_id: str, first_number: int = 1) -> None:
        """ Moves every staged file into the asset directory, numbered in staging order so that
        later chunks of a run sort after earlier ones. """
        staged = sorted(directory.rglob(f"*.{self.extension}")) if directory.exists() else []
        numbers: dict[Path, itertools.count[int]] = {}
        for path in staged:
            target = self.asset_dir(asset) / path.parent.relative_to(directory)
            if target not in numbers:
                target.mkdir(parents=True, exist_ok=True)
                numbers[target] = itertools.count(first_number)
            path.rename(target / self._pending_name(run_id, next(numbers[target])))

    def replace(self, asset: Asset, run_id: str, partition: PartitionKey | None = None) -> None:
        """ Deletes everything under `partition` (the whole asset if `None`) not written by `run_id`. """
        directory = self.partition_dir(asset, partition)
        self._delete_under(directory, keep=lambda path: path.name.startswith(f"run-{run_id}-"))
        if directory != self.asset_dir(asset) and directory.exists() and not any(directory.iterdir()):
            directory.rmdir()
            self._remove_empty_parents(directory, stop=self.asset_dir(asset))

    def clear(self, asset: Asset, partition: PartitionKey | None = None) -> None:
        """ Drops everything written for `asset` (or one partition of it), immediately. """
        directory = self.partition_dir(asset, partition)
        shutil.rmtree(directory, ignore_errors=True)
        self._remove_empty_parents(directory, stop=self.asset_dir(asset))

    def existing_parts(self, asset: Asset, partition: PartitionKey | None = None) -> set[Path]:
        directory = self.partition_dir(asset, partition)
        return set(directory.rglob(f"*.{self.extension}")) if directory.exists() else set()

    def existing_partitions(self, asset: Asset) -> list[PartitionKey]:
        """ Read off the directory names - polars can't list partitions without opening every file. """
        settings = asset.settings.partition
        if settings is None:
            return []
        keys = {
            PartitionKey.from_relative_path(settings, part.parent.relative_to(self.asset_dir(asset)))
            for part in self.existing_parts(asset)
        }
        return sorted((key for key in keys if key is not None), key=lambda key: str(key.relative_path()))

    def layout_matches(self, asset: Asset) -> bool:
        """ False if existing files sit where the asset's current partition settings wouldn't put
        them, e.g. it was partitioned since they were written. Polars can't read a mix. """
        settings = asset.settings.partition
        directory = self.asset_dir(asset)
        for part in self.existing_parts(asset):
            relative = part.parent.relative_to(directory)
            if settings is None:
                if relative != Path():
                    return False
            elif PartitionKey.from_relative_path(settings, relative) is None:
                return False
        return True

    def asset_dir(self, asset: Asset) -> Path:
        return asset.base_storage_path_with_schema / asset.name

    def partition_dir(self, asset: Asset, partition: PartitionKey | None) -> Path:
        directory = self.asset_dir(asset)
        return directory / partition.relative_path() if partition else directory

    def _leaf_partitions(self, asset: Asset, partition: PartitionKey | None) -> list[PartitionKey]:
        """ Every existing leaf partition under `partition`, which may be a prefix. """
        return [
            key for key in self.existing_partitions(asset)
            if partition is None or key.prefix(len(partition)) == partition
        ]

    # Pending: promoted from staging, not yet swept into the canonical set by a `consolidate`.
    # Zero-padded so that parts from the same run sort in the order they were written, which
    # `keep="last"` relies on.
    def _pending_name(self, run_id: str, number: int) -> str:
        return f"run-{run_id}-part-{number:06d}.{self.extension}"

    def _chunk_name(self, chunk_number: int, index: int = 0) -> str:
        return f"chunk-{chunk_number:06d}-{index:08d}.{self.extension}"

    @staticmethod
    def _delete_under(directory: Path, keep: Callable[[Path], bool]) -> None:
        if not directory.exists():
            return
        for path in sorted(directory.rglob("*"), reverse=True):  # children before their parents
            if path.is_dir():
                if not any(path.iterdir()):
                    path.rmdir()
            elif not keep(path):
                path.unlink()

    @staticmethod
    def _remove_empty_parents(directory: Path, stop: Path) -> None:
        while directory != stop and directory.is_relative_to(stop):
            directory = directory.parent
            if directory.exists() and directory != stop and not any(directory.iterdir()):
                directory.rmdir()


class PolarsPartitionedStorage[ConfigType](
    StorageConcrete[pl.LazyFrame, pl.DataFrame | pl.LazyFrame, ConfigType]
):
    """ Parquet/arrow/csv: formats differ only in `extension` and their read/write/scan functions.
    Partition key columns are kept inside the files as well as in the directory names. """

    @abstractmethod
    def scan(self, source: Path | list[Path], hive_schema: dict[str, pl.DataType] | None) -> pl.LazyFrame:
        """ `hive_schema` is `None` for an unpartitioned asset. """
        raise NotImplementedError()

    @abstractmethod
    def write(self, df: pl.DataFrame, path: Path) -> None:
        raise NotImplementedError()

    @abstractmethod
    def sink(self, lf: pl.LazyFrame, path: Path | pl.PartitionBy) -> None:
        raise NotImplementedError()

    def scan_asset(self, asset: Asset, source: Path | list[Path]) -> pl.LazyFrame:
        settings = asset.settings.partition
        return self.scan(source, settings.hive_schema if settings else None)

    # Canonical: the current consolidated set, named plainly `part-{number}` - every
    # `consolidate` call fully regenerates this whole set from 1, so there's nothing for a
    # number to disambiguate against other than itself.
    def _canonical_name(self, number: int) -> str:
        return f"part-{number}.{self.extension}"

    def stage(
            self, config: ConfigType, asset: Asset, obj: pl.DataFrame | pl.LazyFrame, directory: Path,
            chunk_number: int, partition: Mapping[str, Any] | None = None,
    ) -> set[PartitionKey]:
        if not isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
            raise TypeError(
                f"asset {asset.name!r} returned {type(obj).__name__!r}, but its storage "
                f"({type(self).__name__!r}) only persists a polars DataFrame/LazyFrame"
            )

        settings = asset.settings.partition
        if settings is None:
            if partition is not None:
                raise ValueError(f"asset {asset.name!r} isn't partitioned, but was given partition {dict(partition)!r}")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / self._chunk_name(chunk_number)
            if isinstance(obj, pl.DataFrame):
                self.write(obj, path)
            else:
                self.sink(obj, path)
            return set()

        explicit = PartitionKey.from_mapping(settings, partition) if partition is not None else None
        written = self._sink_partitioned(_with_key_columns(asset, obj.lazy(), explicit), directory, settings, chunk_number)
        if explicit is not None and written - {explicit}:
            raise ValueError(
                f"asset {asset.name!r} was given partition {dict(explicit)!r}, but its rows also cover "
                f"{[dict(key) for key in written - {explicit}]!r}"
            )
        return written

    def _sink_partitioned(
            self, lf: pl.LazyFrame, directory: Path, settings: PartitionSettings, chunk_number: int,
    ) -> set[PartitionKey]:
        """ `pl.PartitionBy` is unstable in polars 1.44 - if it breaks, collect and use
        `DataFrame.partition_by(as_dict=True)` instead, at the cost of streaming. """
        written: set[PartitionKey] = set()

        def file_path(args: Any) -> str:
            row = args.partition_keys.row(0, named=True)
            key = PartitionKey((name, row[name]) for name in settings.columns)
            for name, value in key.items():
                if value is None or value == RESERVED_VALUE:
                    raise ValueError(f"partition column {name!r} can't be null or {RESERVED_VALUE!r}")
            written.add(key)
            return str(key.relative_path() / self._chunk_name(chunk_number, args.index_in_partition))

        self.sink(lf, pl.PartitionBy(str(directory), key=list(settings.columns), include_key=True, file_path_provider=file_path))
        return written

    def load(self, config: ConfigType, asset: Asset) -> pl.LazyFrame:
        pattern = "**/*" if asset.settings.partition else "*"
        return self.scan_asset(asset, self.asset_dir(asset) / f"{pattern}.{self.extension}")

    def load_previous(self, config: ConfigType, asset: Asset) -> pl.LazyFrame | None:
        parts = sorted(self.existing_parts(asset))
        return self.scan_asset(asset, parts) if parts else None

    def consolidate(self, config: ConfigType, asset: Asset, partition: PartitionKey | None = None) -> None:
        if asset.settings.partition is None:
            self._consolidate_dir(config, asset, self.asset_dir(asset))
            return
        for key in self._leaf_partitions(asset, partition):
            self._consolidate_dir(config, asset, self.partition_dir(asset, key))

    def _consolidate_dir(self, config: ConfigType, asset: Asset, directory: Path) -> None:
        old_parts = sorted(directory.glob(f"*.{self.extension}")) if directory.exists() else []
        if not old_parts:
            return

        frame = self.scan_asset(asset, old_parts)
        base_config = cast(BaseStorageConfig, config)
        primary_key = asset.settings.primary_key
        if primary_key is not None:
            frame = frame.unique(subset=primary_key, keep="last")
        if base_config.is_duplicate is not None:
            frame = frame.filter(~base_config.is_duplicate)

        # Discard any `.tmp` orphaned by a previously failed consolidate
        for stray in directory.glob(f"part-*.{self.extension}.tmp"):
            stray.unlink()

        generation = uuid.uuid4().hex[:8]

        def generation_staged_name(number: int) -> str:
            return f"part-{generation}-{number}.{self.extension}.tmp"

        max_rows_per_file = base_config.max_rows_per_file
        if max_rows_per_file is None:
            # PartitionBy requires a split criterion, so an uncapped merge is a plain sink.
            staged_paths = [directory / generation_staged_name(1)]
            self.sink(frame, staged_paths[0])
        else:
            part_numbers = itertools.count(1)
            self.sink(frame, pl.PartitionBy(
                str(directory),
                max_rows_per_file=max_rows_per_file,
                file_path_provider=lambda args: generation_staged_name(next(part_numbers)),
            ))
            staged_paths = sorted(directory.glob(f"part-{generation}-*.{self.extension}.tmp"))

        # Phase 1: promote every staged file to its (still generation-tagged, so still
        # collision-free) visible name. A crash here leaves some files promoted and some
        # still `.tmp` - the next consolidate reads the promoted ones as ordinary data and
        # discards the still-staged ones, losing nothing either way.
        generation_paths = [
            staged_path.rename(directory / staged_path.name.removesuffix(".tmp"))
            for staged_path in staged_paths
        ]

        # Phase 2: claim the final `part-N` identities. Each rename here may overwrite a
        # stale old file of the same number - but by now every byte of new data already has
        # a safe, visible home under its generation name, so a crash mid-phase-2 can at worst
        # leave some files under their final name and others still generation-tagged (both
        # are plain, correctly-readable data) - never a state where data is missing.
        new_paths = set()
        for number, generation_path in enumerate(generation_paths, start=1):
            new_paths.add(generation_path.rename(directory / self._canonical_name(number)))

        for old_part in old_parts:
            if old_part not in new_paths:
                old_part.unlink()


def _with_key_columns(asset: Asset, lf: pl.LazyFrame, explicit: PartitionKey | None) -> pl.LazyFrame:
    """ Checks the key columns against their declared dtypes, adding them from `explicit` if missing. """
    settings = cast(PartitionSettings, asset.settings.partition)
    schema = lf.collect_schema()
    for name, dtype in settings.columns.items():
        if name not in schema:
            if explicit is None:
                raise ValueError(f"asset {asset.name!r} is partitioned by {name!r}, but its data has no such column")
            lf = lf.with_columns(pl.lit(explicit[name], dtype=dtype).alias(name))
        elif schema[name] != dtype:
            if dtype == pl.String and isinstance(schema[name], (pl.Categorical, pl.Enum)):
                lf = lf.with_columns(pl.col(name).cast(pl.String))
            else:
                raise TypeError(
                    f"asset {asset.name!r}'s partition column {name!r} is {schema[name]!r}, but is declared "
                    f"as {dtype!r} - cast it first"
                )
    return lf


# PARQUET
class ParquetPolars(PolarsPartitionedStorage[ParquetStorageConfig]):
    extension = "parquet"

    def scan(self, source: Path | list[Path], hive_schema: dict[str, pl.DataType] | None) -> pl.LazyFrame:
        # Polars only turns hive on by itself for a bare directory, not a glob or a list of files.
        return pl.scan_parquet(source, hive_partitioning=hive_schema is not None, hive_schema=hive_schema)

    def write(self, df: pl.DataFrame, path: Path) -> None:
        df.write_parquet(path)

    def sink(self, lf: pl.LazyFrame, path: Path | pl.PartitionBy) -> None:
        lf.sink_parquet(path)

    @property
    def config_type(self) -> type[ParquetStorageConfig]:
        return ParquetStorageConfig


# ARROW
class ArrowPolars(PolarsPartitionedStorage[ArrowStorageConfig]):
    extension = "arrow"  # TODO is this the right extension?

    def scan(self, source: Path | list[Path], hive_schema: dict[str, pl.DataType] | None) -> pl.LazyFrame:
        return pl.scan_ipc(source, hive_partitioning=hive_schema is not None, hive_schema=hive_schema)

    def write(self, df: pl.DataFrame, path: Path) -> None:
        df.write_ipc(path)

    def sink(self, lf: pl.LazyFrame, path: Path | pl.PartitionBy) -> None:
        lf.sink_ipc(path)

    @property
    def config_type(self) -> type[ArrowStorageConfig]:
        return ArrowStorageConfig


# CSV
class CsvPolars(PolarsPartitionedStorage[CsvStorageConfig]):
    extension = "csv"

    def scan(self, source: Path | list[Path], hive_schema: dict[str, pl.DataType] | None) -> pl.LazyFrame:
        if hive_schema is None:
            return pl.scan_csv(source)
        # No hive support for csv, so key columns come from the files. Read them as text first,
        # otherwise csv inference would turn "007" into 7.
        lf = pl.scan_csv(source, schema_overrides={name: pl.String for name in hive_schema})
        return lf.with_columns(_parse_text(name, dtype) for name, dtype in hive_schema.items())

    def write(self, df: pl.DataFrame, path: Path) -> None:
        df.write_csv(path)

    def sink(self, lf: pl.LazyFrame, path: Path | pl.PartitionBy) -> None:
        lf.sink_csv(path)

    @property
    def config_type(self) -> type[CsvStorageConfig]:
        return CsvStorageConfig


def _parse_text(name: str, dtype: pl.DataType) -> pl.Expr:
    column = pl.col(name)
    if dtype == pl.Date:
        return column.str.to_date("%Y-%m-%d")
    if dtype == pl.Boolean:
        return column == "true"
    return column.cast(dtype)


# LGBM
class LGBMModelStorage(StorageConcrete[LGBMReturnType, LGBMReturnType, LGBMStorageConfig]):
    """ One pickled model per asset, or per partition. `load` returns the latest one, or for a
    partitioned asset a dict of the latest per partition. """

    extension = "pkl"

    @property
    def config_type(self) -> type[LGBMStorageConfig]:
        return LGBMStorageConfig

    def stage(
            self, config: LGBMStorageConfig, asset: Asset, obj: LGBMReturnType, directory: Path,
            chunk_number: int, partition: Mapping[str, Any] | None = None,
    ) -> set[PartitionKey]:
        if not isinstance(obj, (LGBMClassifier, LGBMRegressor, LGBMRanker)):
            raise TypeError(
                f"asset {asset.name!r} returned {type(obj).__name__!r}, but its storage "
                f"(LGBMModelStorage) only persists a trained LGBM model"
            )

        settings = asset.settings.partition
        if settings is None:
            if partition is not None:
                raise ValueError(f"asset {asset.name!r} isn't partitioned, but was given partition {dict(partition)!r}")
            key = None
        elif partition is None:
            raise ValueError(
                f"asset {asset.name!r} is a partitioned model, so each model needs a key: "
                f"yield Output(model, partition={{...}})"
            )
        else:
            key = PartitionKey.from_mapping(settings, partition)

        target = directory / key.relative_path() if key else directory
        target.mkdir(parents=True, exist_ok=True)
        with (target / self._chunk_name(chunk_number)).open("wb") as file:
            pickle.dump(obj, file)
        return {key} if key else set()

    def load(self, config: LGBMStorageConfig, asset: Asset) -> Any:
        loaded = self.load_previous(config, asset)
        if loaded is None:
            raise FileNotFoundError(f"no model has been written yet for asset {asset.name!r}")
        return loaded

    def load_previous(self, config: LGBMStorageConfig, asset: Asset) -> Any:
        if asset.settings.partition is None:
            parts = sorted(self.existing_parts(asset))
            return self._unpickle(parts[-1]) if parts else None
        models = {
            key: self._unpickle(sorted(self.existing_parts(asset, key))[-1])
            for key in self.existing_partitions(asset)
        }
        return models or None

    @staticmethod
    def _unpickle(path: Path) -> LGBMReturnType:
        with path.open("rb") as file:
            return pickle.load(file)

    def consolidate(self, config: LGBMStorageConfig, asset: Asset, partition: PartitionKey | None = None) -> None:
        raise NotImplementedError(
            "a model has no parts to compact - LGBMStorageConfig only supports Mode.OVERWRITE, "
            "which never calls consolidate()"
        )


STORAGE_TYPE_MAPPINGS = {
    StorageType.PARQUET: ParquetPolars,
    StorageType.ARROW: ArrowPolars,
    StorageType.CSV: CsvPolars,
    StorageType.LGBM: LGBMModelStorage,
}

CONFIG_TYPE_MAPPINGS = {
    ParquetStorageConfig: StorageType.PARQUET,
    ArrowStorageConfig: StorageType.ARROW,
    CsvStorageConfig: StorageType.CSV,
    LGBMStorageConfig: StorageType.LGBM,
}


@overload
def get_storage_concrete(config: ParquetStorageConfig) -> ParquetPolars: ...
@overload
def get_storage_concrete(config: ArrowStorageConfig) -> ArrowPolars: ...
@overload
def get_storage_concrete(config: CsvStorageConfig) -> CsvPolars: ...
@overload
def get_storage_concrete(config: LGBMStorageConfig) -> LGBMModelStorage: ...
@overload
def get_storage_concrete(config: StorageConfig) -> StorageConcrete[Any, Any, Any]: ...


def get_storage_concrete(config: StorageConfig) -> StorageConcrete[Any, Any, Any]:
    storage_type = CONFIG_TYPE_MAPPINGS[type(config)]
    return STORAGE_TYPE_MAPPINGS[storage_type]()


def consolidate(config: StorageConfig, asset: Asset, partition: PartitionKey | None = None) -> None:
    """ Compacts (and dedupes, per `primary_key`/`is_duplicate`) the part files for `asset`,
    per `config.max_rows_per_file`. """
    get_storage_concrete(config).consolidate(config, asset, partition)
