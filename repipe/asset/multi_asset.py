from dataclasses import dataclass, field
from typing import Callable, Sequence

from repipe.asset.asset import AssetFunc, Dependency, PrimaryKey, _register, _resolve_settings
from repipe.asset.partition import PartitionSettings
from repipe.storage import StorageConfig, StorageType
from repipe.testing.data_quality import DataQualityTest


@dataclass(frozen=True, kw_only=True)
class MultiAssetOut:
    """ One output of a `@multi_asset`, taking the same per-output settings as `@asset`. """
    storage: StorageType | StorageConfig = StorageType.PARQUET
    primary_key: PrimaryKey | None = None
    schemas: list[str] | None = None
    labels: Sequence[str] = field(default_factory=list)
    partition: PartitionSettings | None = None
    tests: Sequence[DataQualityTest] | None = None
    is_required: bool = True
    """ A required output the function doesn't write fails the run. An optional one is left as it was. """


def multi_asset(
        *,
        outputs: dict[str, MultiAssetOut],
        dependencies: list[str] | list[Dependency] | None = None,
        labels: Sequence[str] | None = None,
) -> Callable[[AssetFunc], AssetFunc]:
    """ One function producing several assets. It returns a `dict` keyed by output name, or
    yields `Output(data, asset=name)`. Selecting any of its outputs runs it and writes all of
    them. `labels` apply to every output. """

    def decorator(inner_func: AssetFunc) -> AssetFunc:
        if not outputs:
            raise ValueError(f"multi_asset {inner_func.__name__!r} must declare at least one output")
        settings = {
            name: _resolve_settings(
                inner_func, is_multi=True, storage=output.storage, primary_key=output.primary_key,
                schemas=output.schemas, labels=frozenset([*output.labels, *(labels or [])]),
                partition=output.partition, tests=output.tests, is_required=output.is_required,
            )
            for name, output in outputs.items()
        }
        return _register(inner_func, settings, dependencies, is_multi=True)

    return decorator
