import functools
import inspect
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Coroutine, Iterable, Iterator, Mapping, Sequence, cast, overload

import duckdb
import polars as pl
from pydantic import BaseModel, field_validator, model_validator

from repipe.asset.partition import PartitionSettings
from repipe.storage import (
    LGBMModel,
    LGBMStorageConfig,
    Mode,
    StorageType,
    StorageConfig,
    resolve_storage_config,
)
from repipe.testing.data_quality import DataQualityTest

type AssetData = pl.DataFrame | pl.LazyFrame | LGBMModel | duckdb.DuckDBPyRelation


@dataclass(frozen=True)
class Output:
    """ An explicit write, for what returning or yielding plain data can't say: which output of
    a multi-asset it's for, and/or which partition it's for. Required whenever either applies -
    nothing about the data itself can say it (a model has no columns to read a key from, and a
    dict key can only name one of asset/partition, not both). """
    data: AssetData
    asset: str | None = None
    partition: Mapping[str, Any] | None = None


type AssetReturnType = (
    AssetData | Output | Sequence[Output] | Mapping[str, AssetData | Output]
    | Iterator[AssetData | Output] | AsyncIterator[AssetData | Output]
)
type PrimaryKey = str | pl.Expr | Sequence[str] | Sequence[pl.Expr]

_PRIMARY_KEY_REQUIRED_MODES = frozenset({Mode.APPEND_THEN_MERGE})


class Dependency(BaseModel):
    dependency_name: str


class AssetSettings(BaseModel):
    """ The per-output kwargs of `@asset`/`MultiAssetOut`. """
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


@overload
def asset(func: AssetFunc) -> AssetFunc: ...


@overload
def asset(
        func: None = None,
        *,
        storage: StorageType | StorageConfig | None = None,
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
        storage: StorageType | StorageConfig | None = None,
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
        settings = _resolve_settings(
            inner_func, is_multi=False, storage=storage, primary_key=primary_key, schemas=schemas,
            labels=frozenset(labels or []), partition=partition, tests=tests, is_required=True,
        )
        return _register(inner_func, {inner_func.__name__: settings}, dependencies, is_multi=False)

    if func is not None:
        return decorator(func)

    return decorator


def _resolve_settings(
        inner_func: AssetFunc, is_multi: bool, *,
        storage: StorageType | StorageConfig | None, primary_key: PrimaryKey | None, schemas: list[str] | None,
        labels: frozenset[str], partition: PartitionSettings | None, tests: Sequence[DataQualityTest] | None,
        is_required: bool,
) -> AssetSettings:
    """ Builds one output's `AssetSettings`, shared by `@asset` and `@multi_asset` - each just
    supplies the same per-output kwargs, `@asset`'s own or one `MultiAssetOut`'s. """
    # A multi-asset's annotation (`dict[str, ...]`, `Output`s) doesn't say what each output is
    annotation = inspect.Signature.empty if is_multi else _return_annotation(inner_func)
    resolved_storage = resolve_storage_config(storage, annotation)
    if not is_multi:  # a multi-asset writes every output through `Output`, whatever its shape
        resolved_storage.check_compatible(inner_func, partition)
    return AssetSettings(
        storage=resolved_storage, primary_key=primary_key, schemas=schemas, labels=labels,
        partition=partition, tests=tests, is_required=is_required,
    )


def _return_annotation(func: Callable) -> object:
    """ Just the `return` annotation (`inspect.Signature.empty` if absent/unresolvable). Not
    `get_type_hints(func)`: that raises if any *parameter* annotation is unresolvable, which
    would lose a perfectly good return type. """
    annotation = inspect.get_annotations(func, eval_str=False).get("return", inspect.Signature.empty)
    if isinstance(annotation, str):
        try:
            annotation = eval(annotation, getattr(func, "__globals__", {}))
        except Exception:
            return inspect.Signature.empty
    # The catch-all alias means "unknown", not "ambiguous"; an `Output` says nothing about what
    # it wraps (a partitioned model asset must set `storage=`).
    if annotation is AssetReturnType or _mentions_output(annotation):
        return inspect.Signature.empty
    return annotation


def _mentions_output(annotation: object) -> bool:
    if annotation is Output:
        return True
    if isinstance(annotation, typing.TypeAliasType):
        return _mentions_output(annotation.__value__)
    return any(_mentions_output(arg) for arg in typing.get_args(annotation))


def _register(
        inner_func: AssetFunc,
        outputs: dict[str, AssetSettings],
        dependencies: list[str] | list[Dependency] | None,
        is_multi: bool,
) -> AssetFunc:
    coerced_dependencies = _coerce_dependencies(dependencies)
    _check_self_references(inner_func, list(outputs), coerced_dependencies)

    @functools.wraps(inner_func)
    def inner(*args, **kwargs):
        return inner_func(*args, **kwargs)

    cast(Any, inner).__asset__ = RegisteredAssetFunction(
        name=inner_func.__name__,
        func=inner_func,
        dependencies=coerced_dependencies,
        outputs=outputs,
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
