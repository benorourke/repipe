from importlib.metadata.diagnose import inspect

import polars as pl

from repipe.asset.asset import asset, AssetContext
from repipe.asset.incremental import IncrementalStrategy, IncrementalSettings
from lightgbm import LGBMClassifier

from repipe import Repipe
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
    dependencies=["my_asset"],
    schemas=["intermediate"],
    incremental=IncrementalSettings(
        date_column='test_date',
        primary_key='test_column',
        strategy=IncrementalStrategy.UPSERT
    ),
)
def my_downstream_asset(
        context: AssetContext,
        my_asset: pl.LazyFrame,
) -> pl.LazyFrame:
    return pl.concat([my_asset, my_asset])

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
) -> LGBMClassifier:
    return my_downstream_asset.first()

app = Repipe(
    assets=[
        my_asset,
        my_downstream_asset,
        my_lgbm_asset
    ]
)