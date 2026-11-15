from __future__ import annotations

import abc
import collections.abc as cabc
import enum
import glob
import inspect
import itertools
import json
import math
import os
import pickle
import types
import typing
import uuid
import warnings
from abc import abstractmethod
from collections.abc import Iterator, Mapping, Set as AbstractSet
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, AsyncIterator, Callable, ClassVar, Generator, Self, cast, overload

import duckdb
import polars as pl
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from pydantic import BaseModel, Field, StrictInt, model_validator

from repipe.asset.partition import PartitionKey, PartitionSettings

if TYPE_CHECKING:
    from repipe.asset.asset import Asset

type LGBMModel = LGBMClassifier | LGBMRegressor | LGBMRanker


class StorageType(enum.StrEnum):
    CSV = "CSV"
    PARQUET = "PARQUET"
    ARROW = "ARROW"
    LGBM = "LGBM"
    PICKLE = "PICKLE"
    JSON = "JSON"


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
    """ Text, so the types a frame's columns have are recorded beside the data (see `CsvPolars`) and read back
    from there. """


class SingleObjectStorageConfig(BaseStorageConfig):
    """ One value per asset (or per partition), written whole and replaced whole. It has no rows, so none
    of `BaseStorageConfig`'s part-compaction fields apply. """

    what: ClassVar[str] = "an object"
    """ What the asset holds, as it reads in the errors below. """
    noun: ClassVar[str] = "object"
    inferred: ClassVar[bool] = False
    """ Whether a return annotation can select this storage, so that its errors may say it might have been inferred. """

    @model_validator(mode="after")
    def _check_fields_are_compatible_with_a_single_value(self) -> Self:
        name = type(self).__name__
        if self.mode != Mode.OVERWRITE:
            raise ValueError(
                f"{name} only supports Mode.OVERWRITE, got {self.mode!r} - "
                f"{self.what} has no parts to append or merge"
            )
        if self.is_duplicate is not None:
            raise ValueError(f"{name} doesn't support is_duplicate - {self.what} has no rows to dedupe")
        if self.max_rows_per_file is not None:
            raise ValueError(f"{name} doesn't support max_rows_per_file - {self.what} isn't rows")
        return self

    def check_compatible(self, inner_func: Callable[..., Any], partition: PartitionSettings | None) -> None:
        # A partitioned asset yields one keyed `Output` per partition, so only an
        # unpartitioned one has to return a single value.
        if partition is None and (inspect.isgeneratorfunction(inner_func) or inspect.isasyncgenfunction(inner_func)):
            how = " (possibly inferred from its return annotation; override with `storage=`)" if self.inferred else ""
            raise ValueError(
                f"asset {inner_func.__name__!r} is a generator but uses {type(self).__name__}{how} - "
                f"{self.what} has no per-chunk parts to write; return a single {self.noun} instead"
            )


class LGBMStorageConfig(SingleObjectStorageConfig):
    what: ClassVar[str] = "a trained model"
    noun: ClassVar[str] = "model"
    inferred: ClassVar[bool] = True


class PickleStorageConfig(SingleObjectStorageConfig):
    """ Any picklable Python object. Never inferred from a return annotation: pass `storage=StorageType.PICKLE`. """


class JsonStorageConfig(SingleObjectStorageConfig):
    """ A value made of JSON's own types: a dict with string keys, list, str, int, float, bool or None, nested to
    any depth. Never inferred from a return annotation: pass `storage=StorageType.JSON`. """

    indent: Annotated[StrictInt, Field(ge=0)] | None = 2
    """ `None` writes it on one line. """
    sort_keys: bool = False


type StorageConfig = (
    ParquetStorageConfig | ArrowStorageConfig | CsvStorageConfig | LGBMStorageConfig | PickleStorageConfig
    | JsonStorageConfig
)


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
        case StorageType.PICKLE:
            return PickleStorageConfig()
        case StorageType.JSON:
            return JsonStorageConfig()
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
        """ Moves every staged file into the asset directory, reusing each one's own chunk number
        (not renumbering from 1) so that a chunk promoted on its own - one write session promotes
        once per chunk, as soon as it's yielded, rather than once for the whole run - never
        collides with one promoted earlier or later in the same run. """
        staged = sorted(directory.rglob(f"*.{self.extension}")) if directory.exists() else []
        for path in staged:
            target = self.asset_dir(asset) / path.parent.relative_to(directory)
            target.mkdir(parents=True, exist_ok=True)
            number = int(path.stem.removeprefix("chunk-"))
            path.rename(target / self._pending_name(run_id, number))

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

    def before_write(self, config: ConfigType, asset: Asset, frame: pl.DataFrame | pl.LazyFrame, directory: Path) -> None:
        """ Called with exactly what is about to be written (partition columns included) and the asset's
        staging `directory`, so a format can check it or stage something beside it. """

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

        self.before_write(config, asset, obj, directory)
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


def _scan_files(
    scan: Callable[[Path | list[Path]], pl.LazyFrame],
    read_schema: Callable[[Path], Mapping[str, pl.DataType]],
    source: Path | list[Path],
) -> pl.LazyFrame:
    """ `scan(source)`, or - when the files `source` names don't all have the same columns and types - one scan per file,
    stacked by column name. A plain scan takes its schema from the first file and fails on a later file with another
    one, which is what an asset's stored files look like once its output gains (or changes) a column. Stacked, a column
    a file lacks is null for that file's rows, none is dropped, and a column whose type differs between files is widened
    to one both fit. Files with one schema (the usual case) are scanned together, as they always were. """
    files = sorted(Path(path) for path in (glob.glob(str(source)) if isinstance(source, Path) else source))
    if len(files) < 2 or len({frozenset(read_schema(file).items()) for file in files}) == 1:
        return scan(source)
    return pl.concat([scan([file]) for file in files], how="diagonal_relaxed")


# PARQUET
class ParquetPolars(PolarsPartitionedStorage[ParquetStorageConfig]):
    extension = "parquet"

    def scan(self, source: Path | list[Path], hive_schema: dict[str, pl.DataType] | None) -> pl.LazyFrame:
        # Polars only turns hive on by itself for a bare directory, not a glob or a list of files.
        return _scan_files(
            lambda files: pl.scan_parquet(files, hive_partitioning=hive_schema is not None, hive_schema=hive_schema),
            lambda file: pl.read_parquet_schema(file),
            source,
        )

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
        return _scan_files(
            lambda files: pl.scan_ipc(files, hive_partitioning=hive_schema is not None, hive_schema=hive_schema),
            lambda file: pl.read_ipc_schema(file),
            source,
        )

    def write(self, df: pl.DataFrame, path: Path) -> None:
        df.write_ipc(path)

    def sink(self, lf: pl.LazyFrame, path: Path | pl.PartitionBy) -> None:
        lf.sink_ipc(path)

    @property
    def config_type(self) -> type[ArrowStorageConfig]:
        return ArrowStorageConfig


# CSV
_CSV_SCHEMA_FILE = "_schema.json"
""" The recorded schema, in the asset's directory beside the CSV files, and where a chunk's is staged. """
_CSV_SCHEMA_VERSION = 1

_CSV_SIMPLE_DTYPES: dict[str, pl.DataType] = {
    name: getattr(pl, name)
    for name in (
        "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float16", "Float32", "Float64",
        "Boolean", "String", "Date", "Time", "Categorical", "Null",
    )
    if hasattr(pl, name)
}

type _CsvSchemaRecord = list[list[Any]]
""" `[[column, {"type": ...}], ...]`: what is stored, in the columns' order. """


class CsvPolars(PolarsPartitionedStorage[CsvStorageConfig]):
    """ Plain CSV that reads back as exactly what was written. A CSV carries no types, so each write's schema is
    recorded in `_schema.json` beside the files, and goes wherever the directory goes. Reading takes every column as
    text and parses it by its recorded type, strictly: nothing is inferred, so `"007"` stays a string and a date stays
    a date. A CSV with no record (hand-placed, or from an earlier repipe) is inferred, as it always was.

    One schema per asset: chunks of a run, and an append (or a partitioned overwrite beside partitions it leaves alone),
    must match what is already there. A column that is only nulls (`Null`) fits any type. Each chunk of a partitioned
    run commits on its own, so a partitioned overwrite can't change the schema while other partitions exist, even if
    the run rewrites all of them: `repipe clear` first. Timezone-aware columns are written as UTC, which keeps the
    instant exactly (a local offset with seconds, like Kolkata's before 1906, can't be written), and read back into
    their own zone. """

    extension = "csv"

    @property
    def config_type(self) -> type[CsvStorageConfig]:
        return CsvStorageConfig

    # Without a recorded schema
    def scan(self, source: Path | list[Path], hive_schema: dict[str, pl.DataType] | None) -> pl.LazyFrame:
        if hive_schema is None:
            return pl.scan_csv(source)
        # No hive support for csv, so key columns come from the files. Read them as text first,
        # otherwise csv inference would turn "007" into 7.
        lf = pl.scan_csv(source, schema_overrides={name: pl.String for name in hive_schema})
        return lf.with_columns(_parse_text(name, dtype) for name, dtype in hive_schema.items())

    def write(self, df: pl.DataFrame, path: Path) -> None:
        df.with_columns(self._zones_to_utc(df.schema)).write_csv(path)

    def sink(self, lf: pl.LazyFrame, path: Path | pl.PartitionBy) -> None:
        lf.with_columns(self._zones_to_utc(lf.collect_schema())).sink_csv(path)

    def scan_asset(self, asset: Asset, source: Path | list[Path]) -> pl.LazyFrame:
        record = self._read_record(self.schema_path(asset))
        if record is None:
            return super().scan_asset(asset, source)
        return self._scan_recorded(asset, source, record)

    def _scan_recorded(self, asset: Asset, source: Path | list[Path], record: _CsvSchemaRecord) -> pl.LazyFrame:
        schema = {name: self._spec_to_dtype(spec) for name, spec in record}
        if not (source if isinstance(source, list) else glob.glob(str(source))):
            return pl.LazyFrame(schema=schema)  # an asset overwritten with nothing: empty, but still of its type
        raw = pl.scan_csv(source, infer_schema=False)
        found = raw.collect_schema().names()
        if set(found) != set(schema):
            raise ValueError(
                f"asset {asset.name!r}'s CSV has the columns {found}, but {list(schema)} were recorded when it was "
                f"written - run `repipe clear {asset.name}` if it was changed by hand"
            )
        return raw.with_columns(self._parse_recorded(name, dtype).alias(name) for name, dtype in schema.items())

    def before_write(self, config: CsvStorageConfig, asset: Asset, frame: pl.DataFrame | pl.LazyFrame, directory: Path) -> None:
        record = self._record_of(asset, frame.collect_schema())
        staged = directory / _CSV_SCHEMA_FILE
        previous = self._read_record(staged)
        if previous is not None:
            merged = self._merge_records(previous, record)
            if merged is None:
                raise ValueError(
                    f"asset {asset.name!r} wrote chunks with different schemas ({self._describe_schema_change(previous, record)}) "
                    f"- a CSV asset has one schema"
                )
            record = merged
        directory.mkdir(parents=True, exist_ok=True)
        self._write_record(staged, record)

    def commit(
            self, config: CsvStorageConfig, asset: Asset, staged_dir: Path, run_id: str,
            partitions: AbstractSet[PartitionKey],
    ) -> None:
        staged = self._read_record(staged_dir / _CSV_SCHEMA_FILE)
        if staged is None:  # nothing was written, so an overwrite may have just emptied the asset
            super().commit(config, asset, staged_dir, run_id, partitions)
            if not self.existing_parts(asset):
                self.schema_path(asset).unlink(missing_ok=True)
            return

        kept = self._kept_parts(config, asset, partitions)
        record = self._reconcile(asset, kept, staged)
        # What is kept, and a merge that reads the parts back, need the record to be there first. When this replaces every
        # part it must not be: a commit that fails part way would leave the old data under a record that isn't its own.
        record_first = bool(kept) or config.mode == Mode.APPEND_THEN_MERGE
        if record_first:
            self._write_record(self.schema_path(asset), record)
        super().commit(config, asset, staged_dir, run_id, partitions)
        if not record_first:
            self._write_record(self.schema_path(asset), record)

    def _kept_parts(self, config: CsvStorageConfig, asset: Asset, partitions: AbstractSet[PartitionKey]) -> list[Path]:
        """ The parts that are there now and still will be once this commit is done. """
        settings = asset.settings.partition
        if config.mode != Mode.OVERWRITE:
            return sorted(self.existing_parts(asset))
        if settings is None:
            return []  # every existing part is replaced
        directory = self.asset_dir(asset)
        return sorted(
            path for path in self.existing_parts(asset)
            if PartitionKey.from_relative_path(settings, path.parent.relative_to(directory)) not in partitions
        )

    def _reconcile(self, asset: Asset, kept: list[Path], staged: _CsvSchemaRecord) -> _CsvSchemaRecord:
        """ The record to write: `staged`, once it is known to fit the parts that stay. """
        if not kept:
            return staged
        stored = self._read_record(self.schema_path(asset))
        if stored is None:
            self._check_untyped_parts(asset, kept, staged)
            return staged
        merged = self._merge_records(stored, staged)
        if merged is None:
            raise ValueError(
                f"asset {asset.name!r} has existing data written with a different schema "
                f"({self._describe_schema_change(stored, staged)}) - run `repipe clear {asset.name}` first"
            )
        return merged

    def _check_untyped_parts(self, asset: Asset, kept: list[Path], staged: _CsvSchemaRecord) -> None:
        """ Parts written before types were recorded are about to be read by `staged`. Refuse, before anything is
        committed, if that would change a column they hold as numbers or dates into text, or if they don't parse. """
        inferred = super().scan_asset(asset, kept).collect_schema()
        retyped = [
            name for name, spec in staged
            if name in inferred and inferred[name] != pl.String and self._spec_to_dtype(spec) == pl.String
        ]
        problem = f"{retyped} would become text" if retyped else None
        if problem is None:
            try:
                self._scan_recorded(asset, kept, staged).select(pl.all().null_count()).collect(engine="streaming")
            except (pl.exceptions.PolarsError, ValueError) as error:
                problem = str(error).splitlines()[0]
        if problem is not None:
            raise ValueError(
                f"asset {asset.name!r} has CSV data written before its types were recorded, which doesn't fit this "
                f"write's schema ({problem}) - run `repipe clear {asset.name}` first"
            )

    def replace(self, asset: Asset, run_id: str, partition: PartitionKey | None = None) -> None:
        """ As `StorageConcrete.replace`, except the record: it is not a part, and an overwrite just wrote it. """
        directory = self.partition_dir(asset, partition)
        self._delete_under(directory, keep=lambda path: path.name.startswith(f"run-{run_id}-") or path.name == _CSV_SCHEMA_FILE)
        self._remove_empty_dirs(directory, stop=self.asset_dir(asset))

    def clear(self, asset: Asset, partition: PartitionKey | None = None) -> None:
        super().clear(asset, partition)
        if not self.existing_parts(asset):
            self.schema_path(asset).unlink(missing_ok=True)

    def schema_path(self, asset: Asset) -> Path:
        return self.asset_dir(asset) / _CSV_SCHEMA_FILE

    @staticmethod
    def _dtype_to_spec(dtype: pl.DataType) -> dict[str, Any] | None:
        """ `None` for a type CSV can't hold (nested, binary, duration, object ...). """
        for name, simple in _CSV_SIMPLE_DTYPES.items():
            if dtype == simple:
                return {"type": name}
        if isinstance(dtype, pl.Datetime):
            return {"type": "Datetime", "time_unit": dtype.time_unit, "time_zone": dtype.time_zone}
        if isinstance(dtype, pl.Enum):
            return {"type": "Enum", "categories": dtype.categories.to_list()}
        if isinstance(dtype, pl.Decimal):
            return {"type": "Decimal", "precision": dtype.precision, "scale": dtype.scale}
        return None

    @staticmethod
    def _spec_to_dtype(spec: Mapping[str, Any]) -> pl.DataType:
        kind = spec["type"]
        if kind in _CSV_SIMPLE_DTYPES:
            return _CSV_SIMPLE_DTYPES[kind]
        if kind == "Datetime":
            return pl.Datetime(spec["time_unit"], spec["time_zone"])
        if kind == "Enum":
            return pl.Enum(spec["categories"])
        if kind == "Decimal":
            return pl.Decimal(spec["precision"], spec["scale"])
        raise ValueError(f"unknown type {kind!r} in a recorded CSV schema")

    @staticmethod
    def _record_of(asset: Asset, schema: pl.Schema) -> _CsvSchemaRecord:
        if not schema:
            raise ValueError(f"asset {asset.name!r} has no columns, which CSV can't store")
        record: _CsvSchemaRecord = []
        for name, dtype in schema.items():
            if '"' in name or name.startswith("\ufeff"):
                raise ValueError(
                    f"asset {asset.name!r}'s column {name!r} has a quote or a byte-order mark in its name, which a CSV "
                    f"header can't hold and read back as it was - rename it"
                )
            spec = CsvPolars._dtype_to_spec(dtype)
            if spec is None:
                raise TypeError(
                    f"asset {asset.name!r}'s column {name!r} is {dtype!r}, which CSV can't store - "
                    f"use Parquet or Arrow, or cast it first"
                )
            record.append([name, spec])
        return record

    @staticmethod
    def _merge_records(old: _CsvSchemaRecord, new: _CsvSchemaRecord) -> _CsvSchemaRecord | None:
        """ The one record both fit, or `None`. A column of only nulls has the type `Null`, which says nothing about what it
        holds, so it fits any other. """
        if [name for name, _ in old] != [name for name, _ in new]:
            return None
        merged: _CsvSchemaRecord = []
        for (name, before), (_, after) in zip(old, new):
            if before == after or after["type"] == "Null":
                merged.append([name, before])
            elif before["type"] == "Null":
                merged.append([name, after])
            else:
                return None
        return merged

    @staticmethod
    def _zones_to_utc(schema: pl.Schema) -> list[pl.Expr]:
        """ CSV writes a UTC offset as `+HHMM`, which has no room for the seconds some historical zones have (Kolkata's
        +05:21:10), so the instant is written in UTC and put back in its zone on reading. """
        return [
            pl.col(name).dt.convert_time_zone("UTC")
            for name, dtype in schema.items() if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None
        ]

    @staticmethod
    def _describe_schema_change(old: _CsvSchemaRecord, new: _CsvSchemaRecord) -> str:
        before, after = {name: spec for name, spec in old}, {name: spec for name, spec in new}
        changes = [f"{name!r} added" for name in after if name not in before]
        changes += [f"{name!r} removed" for name in before if name not in after]
        changes += [
            f"{name!r} was {CsvPolars._spec_to_dtype(before[name])!r}, now {CsvPolars._spec_to_dtype(after[name])!r}"
            for name in after if name in before and before[name] != after[name]
        ]
        return "; ".join(changes) or "the column order changed"

    @staticmethod
    def _read_record(path: Path) -> _CsvSchemaRecord | None:
        if not path.is_file():
            return None
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("version") != _CSV_SCHEMA_VERSION:
            raise ValueError(f"{path} has schema version {stored.get('version')!r}, but this repipe reads {_CSV_SCHEMA_VERSION}")
        return stored["columns"]

    @staticmethod
    def _write_record(path: Path, record: _CsvSchemaRecord) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(json.dumps({"version": _CSV_SCHEMA_VERSION, "columns": record}, indent=1), encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _parse_recorded(name: str, dtype: pl.DataType) -> pl.Expr:
        """ A column read as text, as `dtype`. Strict: a value that isn't one fails the read. """
        column = pl.col(name)
        if dtype == pl.String:
            return column
        if dtype == pl.Boolean:
            return column.replace_strict({"true": True, "false": False}, return_dtype=pl.Boolean)
        if dtype == pl.Date:
            return column.str.to_date("%Y-%m-%d", strict=True)
        if dtype == pl.Time:
            return column.str.to_time(strict=True)
        if isinstance(dtype, pl.Datetime):
            if dtype.time_zone is None:
                return column.str.to_datetime(time_unit=dtype.time_unit, strict=True)
            # written with its UTC offset, so parsed as an instant and then put back in its own zone
            instant = column.str.to_datetime(time_unit=dtype.time_unit, time_zone="UTC", strict=True)
            return instant.dt.convert_time_zone(dtype.time_zone)
        if hasattr(pl, "Float16") and dtype == pl.Float16:
            return column.cast(pl.Float32, strict=True).cast(pl.Float16)
        return column.cast(dtype, strict=True)


def _parse_text(name: str, dtype: pl.DataType) -> pl.Expr:
    column = pl.col(name)
    if dtype == pl.Date:
        return column.str.to_date("%Y-%m-%d")
    if dtype == pl.Boolean:
        return column == "true"
    return column.cast(dtype)


# SINGLE OBJECTS
class SingleObjectStorage[LoadType, ConfigType: SingleObjectStorageConfig](StorageConcrete[LoadType, Any, ConfigType]):
    """ One serialised value per asset, or per partition. `load` returns the latest one, or for a
    partitioned asset a dict of the latest per partition. A format only says how to turn a value into
    the bytes of a file and back: where it goes, and how a run commits it, is the same for all of them. """

    @abstractmethod
    def serialize(self, config: ConfigType, asset: Asset, obj: Any) -> bytes:
        """ `obj` as the bytes of one file, as `config` says. Raises an error naming `asset` for anything the format can't hold. """
        raise NotImplementedError()

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        raise NotImplementedError()

    def stage(
            self, config: ConfigType, asset: Asset, obj: Any, directory: Path,
            chunk_number: int, partition: Mapping[str, Any] | None = None,
    ) -> set[PartitionKey]:
        data = self.serialize(config, asset, obj)
        if obj is None:
            # `load_previous` returns None for "nothing written yet", so a stored None couldn't be told apart.
            raise TypeError(f"asset {asset.name!r} returned None - there is nothing to persist")

        settings = asset.settings.partition
        if settings is None:
            if partition is not None:
                raise ValueError(f"asset {asset.name!r} isn't partitioned, but was given partition {dict(partition)!r}")
            key = None
        elif partition is None:
            raise ValueError(
                f"asset {asset.name!r} is partitioned, so each {config.noun} needs a key: "
                f"yield Output({config.noun}, partition={{...}})"
            )
        else:
            key = PartitionKey.from_mapping(settings, partition)

        target = directory / key.relative_path() if key else directory
        target.mkdir(parents=True, exist_ok=True)
        (target / self._chunk_name(chunk_number)).write_bytes(data)
        return {key} if key else set()

    def load(self, config: ConfigType, asset: Asset) -> Any:
        loaded = self.load_previous(config, asset)
        if loaded is None:
            raise FileNotFoundError(f"no {config.noun} has been written yet for asset {asset.name!r}")
        return loaded

    def load_previous(self, config: ConfigType, asset: Asset) -> Any:
        if asset.settings.partition is None:
            parts = sorted(self.existing_parts(asset))
            return self._read(asset, parts[-1]) if parts else None
        values = {
            key: self._read(asset, sorted(self.existing_parts(asset, key))[-1])
            for key in self.existing_partitions(asset)
        }
        return values or None

    def _read(self, asset: Asset, path: Path) -> Any:
        try:
            return self.deserialize(path.read_bytes())
        except Exception as error:
            raise RuntimeError(
                f"asset {asset.name!r}'s file {path} can't be read back ({type(error).__name__}: {error}) - "
                f"if it can't be recovered, `repipe clear {asset.name}` deletes it"
            ) from error

    def consolidate(self, config: ConfigType, asset: Asset, partition: PartitionKey | None = None) -> None:
        raise NotImplementedError(
            f"{config.what} has no parts to compact - {type(config).__name__} only supports Mode.OVERWRITE, "
            f"which never calls consolidate()"
        )


# LGBM
class LGBMModelStorage(SingleObjectStorage[LGBMModel, LGBMStorageConfig]):
    """ A pickled model. """

    extension = "pkl"

    @property
    def config_type(self) -> type[LGBMStorageConfig]:
        return LGBMStorageConfig

    def serialize(self, config: LGBMStorageConfig, asset: Asset, obj: Any) -> bytes:
        if not isinstance(obj, (LGBMClassifier, LGBMRegressor, LGBMRanker)):
            raise TypeError(
                f"asset {asset.name!r} returned {type(obj).__name__!r}, but its storage "
                f"(LGBMModelStorage) only persists a trained LGBM model - pass `storage=` "
                f"(or, for a plain `@asset`, annotate the return type)"
            )
        return pickle.dumps(obj)

    def deserialize(self, data: bytes) -> LGBMModel:
        return pickle.loads(data)


# PICKLE
class PickleStorage(SingleObjectStorage[Any, PickleStorageConfig]):
    """ Any object that pickles and unpickles again. Loading unpickles every `*.pkl` in the asset's directory, which
    runs whatever they say to, so anyone who can write to the storage path can run code: only point repipe at a
    directory you trust. A pickled class is stored by reference, so renaming or moving it makes the file unreadable. """

    extension = "pkl"

    @property
    def config_type(self) -> type[PickleStorageConfig]:
        return PickleStorageConfig

    def serialize(self, config: PickleStorageConfig, asset: Asset, obj: Any) -> bytes:
        try:
            data = pickle.dumps(obj)
        except Exception as error:  # which one depends on the Python version and the object (`__reduce__` is the user's)
            raise TypeError(
                f"asset {asset.name!r} returned a {type(obj).__name__!r} that can't be pickled: {error}"
            ) from error
        try:
            pickle.loads(data)
        except Exception as error:
            # An exception class whose __init__ takes more than it passes to super() is one: it would be written and
            # then fail every load.
            raise TypeError(
                f"asset {asset.name!r} returned a {type(obj).__name__!r} that pickles but can't be unpickled "
                f"again ({type(error).__name__}: {error})"
            ) from error
        return data

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)


# JSON
class JsonStorage(SingleObjectStorage[Any, JsonStorageConfig]):
    """ A value of JSON's own types, read back as exactly what was written. `json` would quietly store a tuple as a list,
    an int key as a string, a str subclass as a str and a `NaN` as a token that isn't JSON, so each of those, and
    anything else but the JSON types, is an error instead. """

    extension = "json"

    _LEAVE = object()
    """ Pushed after a container's children, to say it is no longer one of the values being walked through. """

    @property
    def config_type(self) -> type[JsonStorageConfig]:
        return JsonStorageConfig

    def serialize(self, config: JsonStorageConfig, asset: Asset, obj: Any) -> bytes:
        self._check_native(asset, obj)
        try:
            text = json.dumps(obj, indent=config.indent, sort_keys=config.sort_keys, ensure_ascii=False, allow_nan=False)
        except RecursionError as error:  # json recurses, so a value nested deeper than the interpreter allows can't be written
            raise ValueError(f"asset {asset.name!r}'s value is nested too deeply to write as JSON") from error
        return (text + "\n").encode("utf-8")

    def deserialize(self, data: bytes) -> Any:
        value = json.loads(data, parse_float=self._parse_finite_float, parse_constant=self._reject_constant)
        if value is None:
            raise ValueError("the file holds null, which is never written here")
        return value

    @staticmethod
    def _check_native(asset: Asset, root: Any) -> None:
        """ Raises, naming `asset` and where in the value, for anything JSON can't hold as it is: only exactly `dict`,
        `list`, `str`, `int`, `float`, `bool` and `None` pass. Walks with a stack of its own, so how deep a value may go
        is not limited by the interpreter's, and only builds a path to report one. """
        check_value, check_text, path, type_name = (
            JsonStorage._check_value, JsonStorage._check_text, JsonStorage._path, JsonStorage._type_name
        )
        active: set[int] = set()  # the containers between the root and the one being looked at, to find one inside itself
        stack: list[tuple[Any, Any]] = []
        check_value(asset, root, None, stack)
        while stack:
            container, where = stack.pop()
            if container is JsonStorage._LEAVE:
                active.discard(where)  # `where` is the container's id
                continue
            if id(container) in active:
                raise ValueError(f"asset {asset.name!r}'s value refers back to itself at {path(where)}")
            active.add(id(container))
            stack.append((JsonStorage._LEAVE, id(container)))

            is_dict = type(container) is dict
            pending: list[tuple[Any, Any]] = []
            for key, item in (container.items() if is_dict else enumerate(container)):
                if is_dict:
                    if type(key) is not str:
                        note = " (a str subclass would load back as a plain str)" if isinstance(key, str) else ""
                        raise TypeError(
                            f"asset {asset.name!r}'s value has the {type_name(key)!r} key {key!r} at {path(where)}, "
                            f"but JSON keys are strings - convert it to a str{note}"
                        )
                    if not key.isascii():
                        check_text(asset, key, (where, key))
                # What nearly every value is, settled here rather than in a call
                kind = type(item)
                if kind is str:
                    if item.isascii():
                        continue
                elif kind is float:
                    if item - item == 0:  # not NaN or infinite
                        continue
                elif kind is bool or item is None or (kind is int and item.bit_length() <= 4000):
                    continue
                check_value(asset, item, (where, key), pending)
            stack.extend(reversed(pending))  # so they are walked in order

    @staticmethod
    def _check_value(asset: Asset, value: Any, where: Any, containers: list[tuple[Any, Any]]) -> None:
        """ Checks one value, and adds it to `containers` if it is one to look inside. """
        kind = type(value)
        if value is None or kind is bool:
            return
        if kind is str:
            JsonStorage._check_text(asset, value, where)
        elif kind is int:
            if value.bit_length() > 4000:  # only a very long one can reach the interpreter's limit on digits
                try:
                    str(value)
                except ValueError as error:
                    raise ValueError(
                        f"asset {asset.name!r}'s value has an integer at {JsonStorage._path(where)} that is too long "
                        f"for this interpreter to write ({error})"
                    ) from error
        elif kind is float:
            if not math.isfinite(value):
                raise ValueError(
                    f"asset {asset.name!r}'s value has {value!r} at {JsonStorage._path(where)}, which isn't valid JSON"
                )
        elif kind is list or kind is dict:
            containers.append((value, where))
        else:
            raise JsonStorage._unsupported(asset, value, where)

    @staticmethod
    def _check_text(asset: Asset, text: str, where: Any) -> None:
        if text.isascii():
            return
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(
                f"asset {asset.name!r}'s value has a string at {JsonStorage._path(where)} with the lone surrogate "
                f"{error.object[error.start]!r}, which can't be written as UTF-8"
            ) from error

    @staticmethod
    def _unsupported(asset: Asset, value: Any, where: Any) -> TypeError:
        if isinstance(value, tuple):
            hint = "use a list"
        else:
            base = next((base for base in (str, int, float, list, dict) if isinstance(value, base)), None)
            hint = (
                f"it is a subclass of {base.__name__}, which would load back as a plain {base.__name__} - convert it first"
                if base is not None else "convert it to a str, number, bool, None, list or dict"
            )
        return TypeError(
            f"asset {asset.name!r}'s value has a {JsonStorage._type_name(value)!r} at {JsonStorage._path(where)}, "
            f"which JSON can't store - {hint}"
        )

    @staticmethod
    def _type_name(value: Any) -> str:
        kind = type(value)
        return kind.__name__ if kind.__module__ == "builtins" else f"{kind.__module__}.{kind.__qualname__}"

    @staticmethod
    def _path(where: Any) -> str:
        """ `$.rates[1]`, from the `(parent, key)` links the walk keeps instead of building a string for every value. """
        keys = []
        while where is not None:
            where, key = where
            keys.append(key)
        return "$" + "".join(
            f"[{key}]" if isinstance(key, int) else f".{key}" if key.isidentifier() else f"[{key!r}]"
            for key in reversed(keys)
        )

    @staticmethod
    def _parse_finite_float(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):  # 1e999 is valid JSON that Python reads as infinity
            raise ValueError(f"{text} doesn't fit in a float")
        return value

    @staticmethod
    def _reject_constant(name: str) -> Any:
        raise ValueError(f"{name} isn't valid JSON")


STORAGE_TYPE_MAPPINGS = {
    StorageType.PARQUET: ParquetPolars,
    StorageType.ARROW: ArrowPolars,
    StorageType.CSV: CsvPolars,
    StorageType.LGBM: LGBMModelStorage,
    StorageType.PICKLE: PickleStorage,
    StorageType.JSON: JsonStorage,
}

CONFIG_TYPE_MAPPINGS = {
    ParquetStorageConfig: StorageType.PARQUET,
    ArrowStorageConfig: StorageType.ARROW,
    CsvStorageConfig: StorageType.CSV,
    LGBMStorageConfig: StorageType.LGBM,
    PickleStorageConfig: StorageType.PICKLE,
    JsonStorageConfig: StorageType.JSON,
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
def get_storage_concrete(config: PickleStorageConfig) -> PickleStorage: ...
@overload
def get_storage_concrete(config: JsonStorageConfig) -> JsonStorage: ...
@overload
def get_storage_concrete(config: StorageConfig) -> StorageConcrete[Any, Any, Any]: ...


def get_storage_concrete(config: StorageConfig) -> StorageConcrete[Any, Any, Any]:
    storage_type = CONFIG_TYPE_MAPPINGS[type(config)]
    return STORAGE_TYPE_MAPPINGS[storage_type]()


def consolidate(config: StorageConfig, asset: Asset, partition: PartitionKey | None = None) -> None:
    """ Compacts (and dedupes, per `primary_key`/`is_duplicate`) the part files for `asset`,
    per `config.max_rows_per_file`. """
    get_storage_concrete(config).consolidate(config, asset, partition)
