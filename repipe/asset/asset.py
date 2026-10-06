import functools
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMRanker
from pydantic import BaseModel

from repipe.asset.incremental import IncrementalSettings
from repipe.asset.partition import PartitionSettings
from repipe.storage import BaseStorageConfig, StorageType, StorageConfig, make_default_storage_config
from repipe.testing.data_quality import DataQualityTest

type LGBMReturnType = LGBMClassifier | LGBMRegressor | LGBMRanker
type AssetReturnType = pl.DataFrame | pl.LazyFrame | LGBMReturnType


class Dependency(BaseModel):
    dependency_name: str


class AssetSettings(BaseModel):
    """ Bundle of the kwargs rom the asset decorator """
    storage: StorageConfig
    dependencies: dict[str, Dependency]
    schemas: list[str] | None = None
    incremental: IncrementalSettings | None = None
    labels: frozenset[str] = frozenset()
    partition: PartitionSettings | None = None
    tests: Sequence[DataQualityTest] | None = None

    @cached_property
    def dependency_names(self) -> list[str]:
        return list(self.dependencies.keys())


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


type AssetFunc = Callable[[AssetContext], AssetReturnType]


class RegisteredAsset(BaseModel):
    """
    Pairs a decorated function with the settings it was declared with. Set against the inner decorator function.
    """
    model_config = {"arbitrary_types_allowed": True}

    name: str
    func: AssetFunc
    settings: AssetSettings


def asset(
        func: AssetFunc | None = None,
        *,
        storage: StorageType | StorageConfig = StorageType.PARQUET,
        schemas: list[str] | None = None,
        dependencies: list[str] | list[Dependency] | None = None,
        incremental: IncrementalSettings | None = None,
        labels: Sequence[str] | None = None,
        partition: PartitionSettings | None = None,
        tests: Sequence[DataQualityTest] | None = None,
) -> AssetFunc | Callable[[AssetFunc], AssetFunc]:
    """ Supports both bare `@asset` and parameterized `@asset(...)` usage. """

    def decorator(inner_func: AssetFunc) -> AssetFunc:
        settings = AssetSettings(
            storage=storage if isinstance(storage, BaseStorageConfig) else make_default_storage_config(storage),
            schemas=schemas,
            dependencies=_coerce_dependencies(dependencies),
            incremental=incremental,
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
