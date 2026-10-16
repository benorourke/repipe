import functools
import inspect
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Coroutine, Iterable, Iterator, Mapping, Sequence, cast, overload

import duckdb
import polars as pl
from pydantic import BaseModel, field_validator, model_validator

from repipe.asset.partition import PartitionSettings
from repipe.storage import (
    BaseStorageConfig,
    LGBMReturnType,
    LGBMStorageConfig,
    Mode,
    StorageType,
    StorageConfig,
    make_default_storage_config,
)
from repipe.testing.data_quality import DataQualityTest

type AssetChunk = pl.DataFrame
type AssetData = AssetChunk | pl.LazyFrame | LGBMReturnType | duckdb.DuckDBPyRelation


@dataclass(frozen=True)
class Output:
    """ An explicit write, for what returning or yielding plain data can't say: which output of
    a multi-asset it's for, or a partition key that isn't in the data (e.g. one model per partition). """
    data: AssetData
    asset: str | None = None
    partition: Mapping[str, Any] | None = None


type AssetReturnType = (
    AssetData | Output | Mapping[str, AssetData]
    | Iterator[AssetChunk | Output] | AsyncIterator[AssetChunk | Output]
)
type PrimaryKey = str | pl.Expr | Sequence[str] | Sequence[pl.Expr]

_PRIMARY_KEY_REQUIRED_MODES = frozenset({Mode.APPEND_THEN_MERGE})


class Dependency(BaseModel):
    dependency_name: str


class AssetSettings(BaseModel):
    """ The per-output kwargs of `@asset`/`AssetOut`. """
    model_config = {"arbitrary_types_allowed": True}

    storage: StorageConfig
    primary_key: PrimaryKey | None = None
    schemas: list[str] | None = None
    labels: frozenset[str] = frozenset()
    partition: PartitionSettings | None = None
    tests: Sequence[DataQualityTest] | None = None
    is_required: bool = True

    @field_validator("primary_key", mode="before")
    @classmethod
    def _normalize_primary_key(cls, value: object) -> PrimaryKey | None:
        """ Materializes any iterable into a list, except `str`/`Expr` themselves - a bare
        string is iterable character-by-character, which isn't what a string primary key
        means. """
        if value is None or isinstance(value, (str, pl.Expr)):
            return value
        if isinstance(value, Iterable):
            return list(value)
        raise TypeError(f"primary_key must be a str, pl.Expr, or iterable of either, got {type(value)!r}")

    @model_validator(mode="after")
    def _check_primary_key_required(self) -> "AssetSettings":
        if self.storage.mode in _PRIMARY_KEY_REQUIRED_MODES and self.primary_key is None:
            raise ValueError(f"storage.mode={self.storage.mode!r} requires `primary_key` to be set on the asset")
        return self

    @model_validator(mode="after")
    def _check_primary_key_incompatible_with_lgbm(self) -> "AssetSettings":
        if isinstance(self.storage, LGBMStorageConfig) and self.primary_key is not None:
            raise ValueError("primary_key has no meaning for LGBMStorageConfig - a model has no rows to key by")
        return self

    @model_validator(mode="after")
    def _check_overwrite_by_needs_overwrite(self) -> "AssetSettings":
        if self.partition and self.partition.overwrite_by and self.storage.mode != Mode.OVERWRITE:
            raise ValueError(f"overwrite_by only applies to Mode.OVERWRITE, but storage.mode={self.storage.mode!r}")
        return self


class AssetFunction(BaseModel):
    """ A decorated function. Produces one asset, or several for `@multi_asset`. """
    model_config = {"arbitrary_types_allowed": True}

    name: str
    func: Callable
    dependencies: dict[str, Dependency]
    outputs: list[str]
    is_multi: bool

    python_func: str
    python_module: str
    python_file: str

    @property
    def dependency_names(self) -> list[str]:
        return list(self.dependencies.keys())


class Asset(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str

    base_storage_path: Path

    settings: AssetSettings

    function: AssetFunction

    @property
    def func(self) -> Callable:
        return self.function.func

    @property
    def python_func(self) -> str:
        return self.function.python_func

    @property
    def python_module(self) -> str:
        return self.function.python_module

    @property
    def python_file(self) -> str:
        return self.function.python_file

    @property
    def schemas(self) -> list[str] | None:
        return self.settings.schemas

    @property
    def base_storage_path_with_schema(self) -> Path:
        path = self.base_storage_path
        if self.schemas:
            for schema in self.schemas:
                path = path / schema
        return path


class AssetContext(BaseModel):
    manifest: Asset
    dependencies: Sequence[str]


type AssetFunc = Callable[..., AssetReturnType | Coroutine[Any, Any, AssetReturnType]]


class RegisteredAssetFunction(BaseModel):
    """ What `@asset`/`@multi_asset` set as `__asset__` on the function they decorate. """
    model_config = {"arbitrary_types_allowed": True}

    name: str
    func: AssetFunc
    dependencies: dict[str, Dependency]
    outputs: dict[str, AssetSettings]
    is_multi: bool

    @property
    def dependency_names(self) -> list[str]:
        return list(self.dependencies.keys())


@dataclass(frozen=True, kw_only=True)
class AssetOut:
    """ One output of a `@multi_asset`, taking the same per-output settings as `@asset`. """
    storage: StorageType | StorageConfig = StorageType.PARQUET
    primary_key: PrimaryKey | None = None
    schemas: list[str] | None = None
    labels: Sequence[str] = field(default_factory=list)
    partition: PartitionSettings | None = None
    tests: Sequence[DataQualityTest] | None = None
    is_required: bool = True
    """ A required output the function doesn't write fails the run. An optional one is left as it was. """


@overload
def asset(func: AssetFunc) -> AssetFunc: ...


@overload
def asset(
        func: None = None,
        *,
        storage: StorageType | StorageConfig = StorageType.PARQUET,
        primary_key: PrimaryKey | None = None,
        schemas: list[str] | None = None,
        dependencies: list[str] | list[Dependency] | None = None,
        labels: Sequence[str] | None = None,
        partition: PartitionSettings | None = None,
        tests: Sequence[DataQualityTest] | None = None,
) -> Callable[[AssetFunc], AssetFunc]: ...


def asset(
        func: AssetFunc | None = None,
        *,
        storage: StorageType | StorageConfig = StorageType.PARQUET,
        primary_key: PrimaryKey | None = None,
        schemas: list[str] | None = None,
        dependencies: list[str] | list[Dependency] | None = None,
        labels: Sequence[str] | None = None,
        partition: PartitionSettings | None = None,
        tests: Sequence[DataQualityTest] | None = None,
) -> AssetFunc | Callable[[AssetFunc], AssetFunc]:
    """ Supports both bare `@asset` and parameterized `@asset(...)` usage - the two `@overload`s
    above exist only so a type checker picks the right return type for each form. """

    def decorator(inner_func: AssetFunc) -> AssetFunc:
        output = AssetOut(
            storage=storage, primary_key=primary_key, schemas=schemas, labels=labels or [],
            partition=partition, tests=tests,
        )
        return _register(inner_func, {inner_func.__name__: output}, dependencies, labels=[], is_multi=False)

    if func is not None:
        return decorator(func)

    return decorator


def multi_asset(
        *,
        outputs: dict[str, AssetOut],
        dependencies: list[str] | list[Dependency] | None = None,
        labels: Sequence[str] | None = None,
) -> Callable[[AssetFunc], AssetFunc]:
    """ One function producing several assets. It returns a `dict` keyed by output name, or
    yields `Output(data, asset=name)`. Selecting any of its outputs runs it and writes all of
    them. `labels` apply to every output. """

    def decorator(inner_func: AssetFunc) -> AssetFunc:
        if not outputs:
            raise ValueError(f"multi_asset {inner_func.__name__!r} must declare at least one output")
        return _register(inner_func, outputs, dependencies, labels=labels or [], is_multi=True)

    return decorator


def _register(
        inner_func: AssetFunc,
        outputs: dict[str, AssetOut],
        dependencies: list[str] | list[Dependency] | None,
        labels: Sequence[str],
        is_multi: bool,
) -> AssetFunc:
    coerced_dependencies = _coerce_dependencies(dependencies)
    _check_self_references(inner_func, list(outputs), coerced_dependencies)

    settings = {}
    for name, output in outputs.items():
        storage = output.storage
        resolved_storage = storage if isinstance(storage, BaseStorageConfig) else make_default_storage_config(storage)
        if not is_multi:  # a multi-asset writes every output through `Output`, whatever its shape
            resolved_storage.check_compatible(inner_func, output.partition)
        settings[name] = AssetSettings(
            storage=resolved_storage,
            primary_key=output.primary_key,
            schemas=output.schemas,
            labels=frozenset([*output.labels, *labels]),
            partition=output.partition,
            tests=output.tests,
            is_required=output.is_required,
        )

    @functools.wraps(inner_func)
    def inner(*args, **kwargs):
        return inner_func(*args, **kwargs)

    cast(Any, inner).__asset__ = RegisteredAssetFunction(
        name=inner_func.__name__,
        func=inner_func,
        dependencies=coerced_dependencies,
        outputs=settings,
        is_multi=is_multi,
    )
    return inner


def _check_self_references(inner_func: AssetFunc, output_names: list[str], dependencies: dict[str, Dependency]) -> None:
    """ An asset can depend on itself, but if it does, the kwarg type must be optional (e.g. accept None).
    If it has not been materialized yet, None will be passed in. This is checked at run/registration time."""
    for name in output_names:
        _check_self_reference(inner_func, name, dependencies)


def _check_self_reference(inner_func: AssetFunc, name: str, dependencies: dict[str, Dependency]) -> None:
    parameter = inspect.signature(inner_func).parameters.get(name)
    if parameter is None:
        return

    if name not in dependencies:
        raise ValueError(
            f"asset {name!r} takes a parameter named after itself (its previous output) but "
            f"doesn't list itself in `dependencies` - add {name!r} to `dependencies` to opt in"
        )

    try:
        annotation = typing.get_type_hints(inner_func).get(name, parameter.annotation)
    except Exception:  # unresolvable forward reference - fall back to the raw annotation
        annotation = parameter.annotation

    if not _allows_none(annotation):
        raise TypeError(
            f"asset {name!r} takes a parameter named after itself (its previous output), which "
            f"is None on the first run - annotate it as optional, e.g. `{name}: pl.LazyFrame | None`"
        )


def _allows_none(annotation: object) -> bool:
    if annotation is None or annotation is type(None):
        return True
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return any(_allows_none(arg) for arg in typing.get_args(annotation))
    return False


def _coerce_dependencies(
        dependencies: list[str] | list[Dependency] | dict[str, Dependency] | None
) -> dict[str, Dependency]:
    """Safely converts input type to dict[str, Dependency]"""
    if dependencies is None:
        return {}
    elif isinstance(dependencies, dict):
        return dependencies
    elif isinstance(dependencies, list):
        result: dict[str, Dependency] = {}
        for dep in dependencies:
            if isinstance(dep, str):
                result[dep] = Dependency(dependency_name=dep)
            elif isinstance(dep, Dependency):
                result[dep.dependency_name] = dep
        return result
    else:
        raise ValueError()
