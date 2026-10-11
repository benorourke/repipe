from typing import AsyncIterator

import polars as pl

from repipe import Repipe
from repipe.asset.asset import asset, AssetContext
from repipe.storage import Mode, ParquetStorageConfig
from repipe.testing.data_quality import data_quality_test, TestResult


@data_quality_test
def my_data_quality_test() -> TestResult:
    return TestResult(
        success=False
    )


@asset(
    dependencies=[],
    schemas=["source"],
)
def my_asset(
        context: AssetContext
) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "column_a": [1, 2, 3, 4, 5],
            "column_b": [5, 4, 3, 2, 1]
        }
    ).lazy()


@asset(
    dependencies=["my_downstream_asset", "my_asset"],
    schemas=["intermediate"],
    storage=ParquetStorageConfig(mode=Mode.APPEND_THEN_MERGE),
    primary_key="column_a",
)
async def my_downstream_asset(
        context: AssetContext,
        my_downstream_asset: pl.LazyFrame | None,
        my_asset: pl.LazyFrame,
) -> AsyncIterator[pl.DataFrame]:

    """ Streams chunks, each written as its own part file, then merged by the mode above. """
    for chunk in pl.concat([my_asset, my_asset]).collect().iter_slices(n_rows=5):
        yield chunk


@asset(
    dependencies=[
        'my_downstream_asset'
    ],
    schemas=["mart"],
    tests=[
        my_data_quality_test
    ]
)
def my_lgbm_asset(
        context: AssetContext,
        my_downstream_asset: pl.LazyFrame,
) -> pl.LazyFrame:
    # TODO: `storage.save()` doesn't yet support persisting a real LGBMReturnType (only
    # DataFrame/LazyFrame) - returning one here would crash at materialization time, so this
    # stays a LazyFrame until that's built out.
    return my_downstream_asset.first()


app = Repipe(
    assets=[
        my_asset,
        my_downstream_asset,
        my_lgbm_asset
    ]
)
