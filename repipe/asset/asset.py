import functools
import inspect
import types
import typing
from functools import cached_property
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Coroutine, Iterable, Iterator, Sequence, cast, overload

import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMRanker
from pydantic import BaseModel, field_validator, model_validator

from repipe.asset.partition import PartitionSettings
from repipe.storage import BaseStorageConfig, Mode, StorageType, StorageConfig, make_default_storage_config
from repipe.testing.data_quality import DataQualityTest

type LGBMReturnType = LGBMClassifier | LGBMRegressor | LGBMRanker
type AssetChunk = pl.DataFrame
type AssetReturnType = AssetChunk | pl.LazyFrame | LGBMReturnType | Iterator[AssetChunk] | AsyncIterator[AssetChunk]
type PrimaryKey = str | pl.Expr | Sequence[str] | Sequence[pl.Expr]

_PRIMARY_KEY_REQUIRED_MODES = frozenset({Mode.APPEND_THEN_MERGE})


class Dependency(BaseModel):
    dependency_name: str


class AssetSettings(BaseModel):
    """ Bundle of the kwargs rom the asset decorator """
    model_config = {"arbitrary_types_allowed": True}

    storage: StorageConfig
    primary_key: PrimaryKey | None = None
    dependencies: dict[str, Dependency]
    schemas: list[str] | None = None
    labels: frozenset[str] = frozenset()
    partition: PartitionSettings | None = None
    tests: Sequence[DataQualityTest] | None = None

    @cached_property
    def dependency_names(self) -> list[str]:
        return list(self.dependencies.keys())

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


class Asset(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str

    base_storage_path: Path

    settings: AssetSettings

    func: Callable
    python_func: str
    python_module: str
    python_file: str

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


class RegisteredAsset(BaseModel):
    """
    Pairs a decorated function with the settings it was declared with. Set against the inner decorator function.
    """
    model_config = {"arbitrary_types_allowed": True}

    name: str
    func: AssetFunc
    settings: AssetSettings


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
        _check_self_reference(inner_func, _coerce_dependencies(dependencies))
        settings = AssetSettings(
            storage=storage if isinstance(storage, BaseStorageConfig) else make_default_storage_config(storage),
            primary_key=primary_key,
            schemas=schemas,
            dependencies=_coerce_dependencies(dependencies),
            labels=frozenset(labels or []),
            partition=partition,
            tests=tests,
        )

        @functools.wraps(inner_func)
        def inner(*args, **kwargs):
            return inner_func(*args, **kwargs)

        registered = RegisteredAsset(
            name=inner_func.__name__,
            func=inner_func,
            settings=settings,
        )
        cast(Any, inner).__asset__ = registered

        return inner

    if func is not None:
        return decorator(func)

    return decorator


def _check_self_reference(inner_func: AssetFunc, dependencies: dict[str, Dependency]) -> None:
    """ An asset can depend on itself, but if it does, the kwarg type must be optional (e.g. accept None).
    If it has not been materialized yet, None will be passed in. This is checked at run/registration time."""
    name = inner_func.__name__

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
