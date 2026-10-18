from __future__ import annotations

import abc
import collections.abc as cabc
import enum
import inspect
import itertools
import pickle
import types
import typing
import uuid
import warnings
from abc import abstractmethod
from collections.abc import Iterator, Mapping, Set as AbstractSet
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, AsyncIterator, Callable, Generator, cast, overload

import duckdb
import polars as pl
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from pydantic import BaseModel, model_validator

from repipe.asset.partition import PartitionKey, PartitionSettings

if TYPE_CHECKING:
    from repipe.asset.asset import Asset

type LGBMModel = LGBMClassifier | LGBMRegressor | LGBMRanker


class StorageType(enum.StrEnum):
    CSV = "CSV"
    PARQUET = "PARQUET"
    ARROW = "ARROW"
    LGBM = "LGBM"


class Mode(enum.StrEnum):

    OVERWRITE = "overwrite"
    """ Once the run succeeds, deletes everything it didn't write - the whole asset, or for a
        partitioned asset, just the partitions it wrote to. A failed run changes nothing. """

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
        """ Hook for a storage backend to reject an `@asset` function shape it can't represent. """


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
                f"asset {inner_func.__name__!r} is a generator but uses LGBMStorageConfig (possibly "
                f"inferred from its return annotation; override with `storage=`) - "
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


_LGBM_TYPES = (LGBMClassifier, LGBMRegressor, LGBMRanker)
_FRAME_TYPES = (pl.DataFrame, pl.LazyFrame, duckdb.DuckDBPyRelation)
_ITERATOR_ORIGINS = (
    typing.Iterator, AsyncIterator, Generator,
    cabc.Iterator, cabc.AsyncIterator, cabc.Generator,
)
# Origins that can't be a single frame/model but are legitimate return shapes (e.g. a
# multi-asset's `dict`), so falling back to Parquet for them isn't worth a warning.
_SILENT_FALLBACK_ORIGINS = (list, dict, tuple)


def _leaf_types(annotation: object) -> list[object]:
    """ Flattens a return annotation to the concrete types it could produce. """
    if isinstance(annotation, typing.TypeAliasType):
        return _leaf_types(annotation.__value__)
    origin = typing.get_origin(annotation)
    if origin is Annotated:
        return _leaf_types(typing.get_args(annotation)[0])
    if origin in (typing.Union, types.UnionType):
        return [
            leaf for arg in typing.get_args(annotation) if arg is not type(None)
            for leaf in _leaf_types(arg)
        ]
    if origin in _ITERATOR_ORIGINS:
        return _leaf_types(typing.get_args(annotation)[0])
    return [annotation]


def _infer_storage_type(annotation: object) -> StorageType | None:
    """ `None` if the annotation isn't (only) types we recognise. Raises if it mixes a model
    and a frame, since either could be what the user means. """
    kinds: set[StorageType] = set()
    for leaf in _leaf_types(annotation):
        if isinstance(leaf, type) and issubclass(leaf, _LGBM_TYPES):
            kinds.add(StorageType.LGBM)
        elif isinstance(leaf, type) and issubclass(leaf, _FRAME_TYPES):
            kinds.add(StorageType.PARQUET)
        else:
            return None
    if len(kinds) > 1:
        raise TypeError(
            f"return annotation {annotation!r} mixes a model and a frame - "
            f"pass `storage=` explicitly to say which to persist"
        )
    return next(iter(kinds), None)


def resolve_storage_config(
        storage: StorageType | StorageConfig | None,
        annotation: object = inspect.Signature.empty,
) -> StorageConfig:
    """ An explicit `storage` wins; otherwise infers from the return `annotation`; otherwise
    Parquet. `annotation` is `inspect.Signature.empty` when absent. """
    if isinstance(storage, BaseStorageConfig):
        return storage
    if storage is not None:
        return make_default_storage_config(storage)

    if annotation is inspect.Signature.empty:
        return make_default_storage_config(StorageType.PARQUET)

    inferred = _infer_storage_type(annotation)
    if inferred is None:
        if annotation not in (None, Any) and typing.get_origin(annotation) not in _SILENT_FALLBACK_ORIGINS:
            warnings.warn(
                f"unrecognised return annotation {annotation!r}; defaulting to Parquet storage - "
                f"pass `storage=` to choose another",
                stacklevel=4,
            )
        inferred = StorageType.PARQUET
    return make_default_storage_config(inferred)


def staging_root(base_storage_path: Path) -> Path:
    """ Where runs stage their writes before committing them. Outside every asset directory,
    since stray files inside one break scans, and under `base_storage_path` so the commit's
    renames stay on one filesystem. """
    return base_storage_path / ".repipe" / "staging"


def staging_dir(base_storage_path: Path, run_id: str) -> Path:
    return staging_root(base_storage_path) / run_id


class StorageConcrete[LoadType, SaveType, ConfigType](abc.ABC):
    """ Writes happen in two steps: `stage` writes a chunk into a staging directory, then
    `promote` moves everything staged into the asset directory as `run-{run_id}-part-{n}`,
    pending until `replace` or `consolidate` settles it. A partitioned asset has the same
    layout, one level per partition column.

    Each asset gets its own directory (`asset_dir`) - that's the only thing a multi-asset's
    outputs share, since the session commits each one independently, with its own mode and
    partition settings:

        data/
          runners/                     # multi-asset output, partitioned by date, OVERWRITE
            date=2026-10-12/
              run-20261016-101500-123-0001-part-000001.parquet
            date=2026-10-13/
              run-20261016-101500-123-0001-part-000001.parquet
          odds/                        # sibling output, unpartitioned, APPEND_THEN_MERGE
            part-1.csv

    `replace` (under `OVERWRITE`) deletes everything in the touched replace unit that isn't
    from the current run - pending or already-canonical alike. `consolidate` (under
    `APPEND_THEN_MERGE`) rewrites every existing file in the directory, pending and canonical,
    into a fresh canonical `part-{n}` set - which is why a consolidated asset's files are
    always canonical: nothing pending survives past the commit that wrote it. """

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

    def promote(self, asset: Asset, directory: Path, run_id: str) -> None:
        """ Moves every staged file into the asset directory, numbered in staging order so that
        later chunks of a run sort after earlier ones. """
        staged = sorted(directory.rglob(f"*.{self.extension}")) if directory.exists() else []
        numbers: dict[Path, itertools.count[int]] = {}
        for path in staged:
            target = self.asset_dir(asset) / path.parent.relative_to(directory)
            if target not in numbers:
                target.mkdir(parents=True, exist_ok=True)
                numbers[target] = itertools.count(1)
            path.rename(target / self._pending_name(run_id, next(numbers[target])))

    def commit(
            self, config: ConfigType, asset: Asset, staged_dir: Path, run_id: str,
            partitions: AbstractSet[PartitionKey],
    ) -> None:
        """ Promotes everything staged this run into place, then applies `config.mode`:
        OVERWRITE deletes whatever each touched replace unit didn't just receive, and
        APPEND_THEN_MERGE consolidates each partition `partitions` names (the whole asset if
        it isn't partitioned). The caller only needs to know what it staged and where - not
        what each mode means. """
        self.promote(asset, staged_dir, run_id)

        base_config = cast(BaseStorageConfig, config)
        settings = asset.settings.partition
        if base_config.mode == Mode.OVERWRITE:
            if settings is None:
                self.replace(asset, run_id)
            else:
                for key in partitions:
                    self.replace(asset, run_id, key)
        elif base_config.mode == Mode.APPEND_THEN_MERGE:
            if settings is None:
                self.consolidate(config, asset)
            for key in partitions:
                self.consolidate(config, asset, key)

    def replace(self, asset: Asset, run_id: str, partition: PartitionKey | None = None) -> None:
        """ Deletes everything under `partition` (the whole asset if `None`) not written by
        `run_id`, including files left in another partition layout. """
        directory = self.partition_dir(asset, partition)
        self._delete_under(directory, keep=lambda path: path.name.startswith(f"run-{run_id}-"))
        self._remove_empty_dirs(directory, stop=self.asset_dir(asset))

    def clear(self, asset: Asset, partition: PartitionKey | None = None) -> None:
        """ Drops everything written for `asset` (or one partition of it), immediately. """
        directory = self.partition_dir(asset, partition)
        self._delete_under(directory, keep=lambda path: False)
        self._remove_empty_dirs(directory, stop=self.asset_dir(asset) if partition else directory.parent)

    def existing_parts(self, asset: Asset, partition: PartitionKey | None = None) -> set[Path]:
        """ Files that are where the asset's partition settings would put them. """
        settings = asset.settings.partition
        directory = self.asset_dir(asset)
        if settings is None:
            return set(directory.glob(f"*.{self.extension}")) if partition is None else set()
        return {
            path for path in self._own_files(self.partition_dir(asset, partition))
            if path.suffix == f".{self.extension}"
            and PartitionKey.from_relative_path(settings, path.parent.relative_to(directory)) is not None
        }

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
        parts = {path for path in self._own_files(self.asset_dir(asset)) if path.suffix == f".{self.extension}"}
        return parts == self.existing_parts(asset)

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

    def _chunk_name(self, chunk_number: int) -> str:
        return f"chunk-{chunk_number:06d}.{self.extension}"

    @classmethod
    def _own_files(cls, directory: Path) -> Iterator[Path]:
        """ Files in `directory` and its `k=v` subdirectories. Other subdirectories aren't this
        asset's: one named after it in `schemas` keeps its assets in here too. """
        if not directory.is_dir():
            return
        for path in directory.iterdir():
            if path.is_file():
                yield path
            elif "=" in path.name:
                yield from cls._own_files(path)

    @classmethod
    def _delete_under(cls, directory: Path, keep: Callable[[Path], bool]) -> None:
        """ Deletes each file `_own_files` would find that `keep` rejects, then the `k=v`
        directories that leaves empty. """
        if not directory.is_dir():
            return
        for path in list(directory.iterdir()):
            if path.is_file():
                if not keep(path):
                    path.unlink()
            elif "=" in path.name:
                cls._delete_under(path, keep)
                if not any(path.iterdir()):
                    path.rmdir()

    @staticmethod
    def _remove_empty_dirs(directory: Path, stop: Path) -> None:
        """ Removes `directory`, then each parent below `stop`, while they're empty. """
        while directory != stop and directory.is_relative_to(stop) and directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
            directory = directory.parent


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
                f"({type(self).__name__!r}) only persists a polars DataFrame/LazyFrame - pass `storage=` "
                f"(or, for a plain `@asset`, annotate the return type e.g. `-> LGBMClassifier`)"
            )

        was_eager = isinstance(obj, pl.DataFrame)
        settings = asset.settings.partition
        if settings is None:
            if partition is not None:
                raise ValueError(f"asset {asset.name!r} isn't partitioned, but was given partition {dict(partition)!r}")
            target = directory
            key = None
        else:
            if partition is None:
                raise ValueError(
                    f"asset {asset.name!r} is partitioned, so each write needs an explicit partition - "
                    f"yield/return Output(data, partition={{...}})"
                )
            key = PartitionKey.from_mapping(settings, partition)
            obj = _with_key_columns(asset, obj.lazy(), key)
            target = directory / key.relative_path()

        target.mkdir(parents=True, exist_ok=True)
        path = target / self._chunk_name(chunk_number)
        if isinstance(obj, pl.LazyFrame):
            if was_eager:
                self.write(obj.collect(), path)
            else:
                self.sink(obj, path)
        else:
            self.write(obj, path)
        return {key} if key is not None else set()

    def load(self, config: ConfigType, asset: Asset) -> pl.LazyFrame:
        # Spelled out rather than `**`, which would also pick up assets nested in here by `schemas`
        settings = asset.settings.partition
        levels = [f"{name}=*" for name in settings.columns] if settings else []
        return self.scan_asset(asset, self.asset_dir(asset).joinpath(*levels, f"*.{self.extension}"))

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


def _with_key_columns(asset: Asset, lf: pl.LazyFrame, key: PartitionKey) -> pl.LazyFrame:
    """ Adds any of `key`'s columns missing from `lf` (checked against the declared dtypes), and -
    for columns already present - checks their values agree with `key`. A frame whose own column
    disagrees with the partition it's being written to would otherwise land in a directory its
    data doesn't match. """
    settings = cast(PartitionSettings, asset.settings.partition)
    schema = lf.collect_schema()
    mismatches: list[pl.Expr] = []
    for name, dtype in settings.columns.items():
        if name not in schema:
            lf = lf.with_columns(pl.lit(key[name], dtype=dtype).alias(name))
            continue
        if schema[name] != dtype:
            if dtype == pl.String and isinstance(schema[name], (pl.Categorical, pl.Enum)):
                lf = lf.with_columns(pl.col(name).cast(pl.String))
            else:
                raise TypeError(
                    f"asset {asset.name!r}'s partition column {name!r} is {schema[name]!r}, but is declared "
                    f"as {dtype!r} - cast it first"
                )
        # `ne_missing`, not `!=` - a plain inequality is null if the column's value is null,
        # which `filter()` drops as "not a mismatch" instead of flagging it as one.
        mismatches.append(pl.col(name).ne_missing(pl.lit(key[name], dtype=dtype)))

    if mismatches and not lf.filter(pl.any_horizontal(mismatches)).limit(1).collect().is_empty():
        raise ValueError(f"asset {asset.name!r} was given partition {dict(key)!r}, but its own columns disagree with it")
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
class LGBMModelStorage(StorageConcrete[LGBMModel, LGBMModel, LGBMStorageConfig]):
    """ One pickled model per asset, or per partition. `load` returns the latest one, or for a
    partitioned asset a dict of the latest per partition. """

    extension = "pkl"

    @property
    def config_type(self) -> type[LGBMStorageConfig]:
        return LGBMStorageConfig

    def stage(
            self, config: LGBMStorageConfig, asset: Asset, obj: LGBMModel, directory: Path,
            chunk_number: int, partition: Mapping[str, Any] | None = None,
    ) -> set[PartitionKey]:
        if not isinstance(obj, (LGBMClassifier, LGBMRegressor, LGBMRanker)):
            raise TypeError(
                f"asset {asset.name!r} returned {type(obj).__name__!r}, but its storage "
                f"(LGBMModelStorage) only persists a trained LGBM model - pass `storage=` "
                f"(or, for a plain `@asset`, annotate the return type)"
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
    def _unpickle(path: Path) -> LGBMModel:
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
