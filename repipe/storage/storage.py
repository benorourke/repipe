from __future__ import annotations

import abc
import enum
import inspect
import itertools
import pickle
import shutil
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast, overload

import polars as pl
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from pydantic import BaseModel, model_validator

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
    """ After this run's new part(s) write successfully, deletes every part - pending or
        canonical - left over from before this run. Deleting only after the new write
        succeeds (never before) means a failed run can't destroy the last good output.
        Net effect: the directory holds only the latest run's part(s). """

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

    def check_compatible(self, inner_func: Callable[..., Any]) -> None:
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

    def check_compatible(self, inner_func: Callable[..., Any]) -> None:
        if inspect.isgeneratorfunction(inner_func) or inspect.isasyncgenfunction(inner_func):
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


class LoadStrategy[T, ConfigType](abc.ABC):

    @abstractmethod
    def load(self, config: ConfigType, asset: Asset) -> T:
        pass


class SaveStrategy[T, ConfigType](abc.ABC):

    @abstractmethod
    def save(self, config: ConfigType, asset: Asset, obj: T, run_id: str, part_number: int):
        pass


class StorageConcrete[LoadType, SaveType, ConfigType](
    LoadStrategy[LoadType, ConfigType], SaveStrategy[SaveType, ConfigType]
):

    @property
    @abstractmethod
    def config_type(self) -> type[ConfigType]:
        raise NotImplementedError()

    @abstractmethod
    def clear(self, asset: Asset) -> None:
        """ Drops everything previously written for `asset`, immediately. """
        raise NotImplementedError()

    @abstractmethod
    def existing_parts(self, asset: Asset) -> set[Path]:
        """ Every part file currently on disk for `asset`, before any new write - lets a
        caller (e.g. `Mode.OVERWRITE`) snapshot what to delete *after* a new write succeeds,
        instead of clearing upfront and risking the old data if the write then fails. """
        raise NotImplementedError()

    def load_previous(self, config: ConfigType, asset: Asset) -> LoadType | None:
        """ A snapshot of what is on disk for `asset` right now, or `None` if nothing has been
        written yet. Fixed to the parts that exist at call time - unlike `load`, which may
        resolve lazily and so would pick up (or lose) parts written during this run. """
        raise NotImplementedError()

    @abstractmethod
    def consolidate(self, config: ConfigType, asset: Asset) -> None:
        """ Compacts (and, per `config`, dedupes) every part file, per `config.max_rows_per_file`. """
        raise NotImplementedError()

    def asset_dir(self, asset: Asset) -> Path:
        return asset.base_storage_path_with_schema / asset.name


class PolarsPartitionedStorage[ConfigType](
    StorageConcrete[pl.LazyFrame, pl.DataFrame | pl.LazyFrame, ConfigType]
):
    """ Shared directory-of-parts layout for parquet/arrow/csv: each asset gets its own
    directory of uniquely-named part files. Formats differ only in `extension` and their
    read/write/scan functions. """

    # A plain class attribute, not an abstract `@property` - each format subclass sets this
    # directly (`extension = "parquet"`), which pyright can only check against a plain
    # attribute's declared type, not a property's.
    extension: str

    @abstractmethod
    def scan(self, glob_path: Path) -> pl.LazyFrame:
        raise NotImplementedError()

    @abstractmethod
    def write(self, df: pl.DataFrame, path: Path) -> None:
        raise NotImplementedError()

    @abstractmethod
    def sink(self, lf: pl.LazyFrame, path: Path | pl.PartitionBy) -> None:
        raise NotImplementedError()

    def part_glob(self, asset: Asset) -> Path:
        return self.asset_dir(asset) / f"*.{self.extension}"

    # Pending: written by `save`, not yet swept into the canonical set by a `consolidate`.
    # Named `run-{run_id}-part-{number}` - one `run_id` per `materialize()` call, `number`
    # zero-padded since multiple chunks under the *same* run_id must still sort correctly
    # relative to each other for `keep="last"` to mean "most recent".
    def _pending_name(self, run_id: str, number: int) -> str:
        return f"run-{run_id}-part-{number:06d}.{self.extension}"

    def _pending_staged_name(self, run_id: str, number: int) -> str:
        return f"{self._pending_name(run_id, number)}.tmp"

    # Canonical: the current consolidated set, named plainly `part-{number}` - every
    # `consolidate` call fully regenerates this whole set from 1, so there's nothing for a
    # number to disambiguate against other than itself.
    def _canonical_name(self, number: int) -> str:
        return f"part-{number}.{self.extension}"

    def clear(self, asset: Asset) -> None:
        shutil.rmtree(self.asset_dir(asset), ignore_errors=True)

    def existing_parts(self, asset: Asset) -> set[Path]:
        directory = self.asset_dir(asset)
        return set(directory.glob(f"*.{self.extension}")) if directory.exists() else set()

    def save(
            self, config: ConfigType, asset: Asset, obj: pl.DataFrame | pl.LazyFrame,
            run_id: str, part_number: int,
    ) -> None:
        if not isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
            raise TypeError(
                f"asset {asset.name!r} returned {type(obj).__name__!r}, but its storage "
                f"({type(self).__name__!r}) only persists a polars DataFrame/LazyFrame"
            )

        directory = self.asset_dir(asset)
        directory.mkdir(parents=True, exist_ok=True)
        staged_path = directory / self._pending_staged_name(run_id, part_number)

        if isinstance(obj, pl.DataFrame):
            self.write(obj, staged_path)
        else:
            self.sink(obj, staged_path)

        staged_path.rename(directory / self._pending_name(run_id, part_number))

    def load(self, config: ConfigType, asset: Asset) -> pl.LazyFrame:
        return self.scan(self.part_glob(asset))

    def load_previous(self, config: ConfigType, asset: Asset) -> pl.LazyFrame | None:
        parts = sorted(self.existing_parts(asset))
        return pl.concat([self.scan(part) for part in parts]) if parts else None

    def consolidate(self, config: ConfigType, asset: Asset) -> None:
        directory = self.asset_dir(asset)
        old_parts = sorted(directory.glob(f"*.{self.extension}")) if directory.exists() else []
        if not old_parts:
            return

        frame = pl.concat([self.scan(part) for part in old_parts])
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


# PARQUET
class ParquetPolars(PolarsPartitionedStorage[ParquetStorageConfig]):
    extension = "parquet"

    def scan(self, glob_path: Path) -> pl.LazyFrame:
        return pl.scan_parquet(glob_path)

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

    def scan(self, glob_path: Path) -> pl.LazyFrame:
        return pl.scan_ipc(glob_path)

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

    def scan(self, glob_path: Path) -> pl.LazyFrame:
        return pl.scan_csv(glob_path)

    def write(self, df: pl.DataFrame, path: Path) -> None:
        df.write_csv(path)

    def sink(self, lf: pl.LazyFrame, path: Path | pl.PartitionBy) -> None:
        lf.sink_csv(path)

    @property
    def config_type(self) -> type[CsvStorageConfig]:
        return CsvStorageConfig


# LGBM
class LGBMModelStorage(StorageConcrete[LGBMReturnType, LGBMReturnType, LGBMStorageConfig]):
    """ A single trained model per asset, not a directory of rows - but the same pending-name
    `.tmp`-then-rename atomicity. `Mode.OVERWRITE` is the only mode `LGBMStorageConfig` allows. """

    def _pending_name(self, run_id: str) -> str:
        return f"run-{run_id}-model.pkl"

    def _pending_staged_name(self, run_id: str) -> str:
        return f"{self._pending_name(run_id)}.tmp"

    @property
    def config_type(self) -> type[LGBMStorageConfig]:
        return LGBMStorageConfig

    def clear(self, asset: Asset) -> None:
        shutil.rmtree(self.asset_dir(asset), ignore_errors=True)

    def existing_parts(self, asset: Asset) -> set[Path]:
        directory = self.asset_dir(asset)
        return set(directory.glob("*.pkl")) if directory.exists() else set()

    def save(
            self, config: LGBMStorageConfig, asset: Asset, obj: LGBMReturnType,
            run_id: str, part_number: int,
    ) -> None:
        if not isinstance(obj, (LGBMClassifier, LGBMRegressor, LGBMRanker)):
            raise TypeError(
                f"asset {asset.name!r} returned {type(obj).__name__!r}, but its storage "
                f"(LGBMModelStorage) only persists a trained LGBM model"
            )

        directory = self.asset_dir(asset)
        directory.mkdir(parents=True, exist_ok=True)
        staged_path = directory / self._pending_staged_name(run_id)

        with staged_path.open("wb") as file:
            pickle.dump(obj, file)

        staged_path.rename(directory / self._pending_name(run_id))

    def load(self, config: LGBMStorageConfig, asset: Asset) -> LGBMReturnType:
        parts = sorted(self.existing_parts(asset))
        if not parts:
            raise FileNotFoundError(f"no model has been written yet for asset {asset.name!r}")
        with parts[-1].open("rb") as file:
            return pickle.load(file)

    def load_previous(self, config: LGBMStorageConfig, asset: Asset) -> LGBMReturnType | None:
        parts = sorted(self.existing_parts(asset))
        if not parts:
            return None
        with parts[-1].open("rb") as file:
            return pickle.load(file)

    def consolidate(self, config: LGBMStorageConfig, asset: Asset) -> None:
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


def consolidate(config: StorageConfig, asset: Asset) -> None:
    """ Compacts (and dedupes, per `primary_key`/`is_duplicate`) every part file for `asset`,
    per `config.max_rows_per_file`. """
    get_storage_concrete(config).consolidate(config, asset)
