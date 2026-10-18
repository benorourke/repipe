from __future__ import annotations

import asyncio
import warnings
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Iterator, Optional

import duckdb
import numpy as np
import polars as pl
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor

from repipe.asset import AssetContext, AssetReturnType, MultiAssetOut, Output, asset, multi_asset
from repipe.cli.selector import build_graph
from repipe.project import Repipe
from repipe.runtime import materialize
from repipe.storage import (
    ArrowStorageConfig,
    LGBMModel,
    LGBMStorageConfig,
    Mode,
    ParquetStorageConfig,
    StorageType,
)


class MyClassifier(LGBMClassifier):
    pass


class MyUnknownType:
    pass


def _storage(func):
    registered = func.__asset__
    return registered.outputs[registered.name].storage


class TestInferredStorage:
    def test_model_annotation_infers_lgbm(self):
        @asset
        def a() -> LGBMClassifier: ...

        assert isinstance(_storage(a), LGBMStorageConfig)

    def test_model_subclass_infers_lgbm(self):
        @asset
        def a() -> MyClassifier: ...

        assert isinstance(_storage(a), LGBMStorageConfig)

    def test_union_of_models_infers_lgbm(self):
        @asset
        def a() -> LGBMClassifier | LGBMRegressor: ...

        assert isinstance(_storage(a), LGBMStorageConfig)

    def test_type_alias_of_models_infers_lgbm(self):
        @asset
        def a() -> LGBMModel: ...

        assert isinstance(_storage(a), LGBMStorageConfig)

    @pytest.mark.parametrize("annotation", [pl.DataFrame, pl.LazyFrame, duckdb.DuckDBPyRelation])
    def test_frames_infer_parquet(self, annotation):
        def a(): ...
        a.__annotations__["return"] = annotation

        assert isinstance(_storage(asset(a)), ParquetStorageConfig)

    def test_optional_model_infers_lgbm(self):
        @asset
        def a() -> Optional[LGBMClassifier]: ...

        assert isinstance(_storage(a), LGBMStorageConfig)

    def test_annotated_model_infers_lgbm(self):
        @asset
        def a() -> Annotated[LGBMClassifier, "meta"]: ...

        assert isinstance(_storage(a), LGBMStorageConfig)

    def test_iterator_of_frames_infers_parquet(self):
        @asset
        def a() -> Iterator[pl.DataFrame]:
            yield pl.DataFrame()

        assert isinstance(_storage(a), ParquetStorageConfig)

    def test_async_iterator_of_frames_infers_parquet(self):
        @asset
        async def a() -> AsyncIterator[pl.DataFrame]:
            yield pl.DataFrame()

        assert isinstance(_storage(a), ParquetStorageConfig)

    def test_bare_and_called_decorator_both_infer(self):
        @asset()
        def a() -> LGBMClassifier: ...

        assert isinstance(_storage(a), LGBMStorageConfig)

    def test_string_annotation_is_resolved(self):
        def a(): ...
        a.__annotations__["return"] = "LGBMClassifier"
        a.__globals__["LGBMClassifier"] = LGBMClassifier

        assert isinstance(_storage(asset(a)), LGBMStorageConfig)

    def test_unresolvable_parameter_annotation_does_not_hide_return_type(self):
        @asset
        def a(x: NotDefinedAnywhere) -> LGBMClassifier: ...  # noqa: F821

        assert isinstance(_storage(a), LGBMStorageConfig)

    @pytest.mark.parametrize("annotation", [None, Any, list[int], dict[str, int], AssetReturnType])
    def test_unrecognised_annotation_falls_back_to_parquet(self, annotation):
        def a(): ...
        a.__annotations__["return"] = annotation

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert isinstance(_storage(asset(a)), ParquetStorageConfig)

    def test_missing_annotation_falls_back_to_parquet_silently(self):
        @asset
        def a(): ...

        assert isinstance(_storage(a), ParquetStorageConfig)

    def test_unrecognised_annotation_warns(self):
        def a(): ...
        a.__annotations__["return"] = MyUnknownType

        with pytest.warns(UserWarning, match="defaulting to Parquet"):
            asset(a)

    def test_mixed_model_and_frame_union_raises(self):
        def a(): ...
        a.__annotations__["return"] = LGBMClassifier | pl.DataFrame

        with pytest.raises(TypeError, match="pass `storage=`"):
            asset(a)


class TestPartitionedAndMultiAssets:
    @pytest.mark.parametrize("annotation", [Output, list[Output], Iterator[Output]])
    def test_output_annotations_fall_back_to_parquet_without_warning(self, annotation):
        def a(): ...
        a.__annotations__["return"] = annotation

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert isinstance(_storage(asset(a)), ParquetStorageConfig)

    def test_multi_asset_outputs_default_to_parquet_and_honour_explicit_storage(self):
        @multi_asset(outputs={"x": MultiAssetOut(), "m": MultiAssetOut(storage=LGBMStorageConfig())})
        def f() -> dict[str, Any]: ...

        outputs = f.__asset__.outputs
        assert isinstance(outputs["x"].storage, ParquetStorageConfig)
        assert isinstance(outputs["m"].storage, LGBMStorageConfig)


class TestExplicitStorageWins:
    def test_explicit_config_overrides_annotation(self):
        @asset(storage=ArrowStorageConfig())
        def a() -> LGBMClassifier: ...

        assert isinstance(_storage(a), ArrowStorageConfig)

    def test_explicit_storage_type_overrides_annotation(self):
        @asset(storage=StorageType.CSV)
        def a() -> LGBMClassifier: ...

        assert type(_storage(a)).__name__ == "CsvStorageConfig"

    def test_explicit_mixed_union_does_not_raise(self):
        def a(): ...
        a.__annotations__["return"] = LGBMClassifier | pl.DataFrame

        assert isinstance(_storage(asset(storage=StorageType.PARQUET)(a)), ParquetStorageConfig)


class TestInferredConfigIsValidated:
    def test_inferred_lgbm_rejects_primary_key(self):
        with pytest.raises(ValueError, match="primary_key"):
            @asset(primary_key="id")
            def a() -> LGBMClassifier: ...

    def test_inferred_lgbm_rejects_generator_and_says_inferred(self):
        with pytest.raises(ValueError, match="inferred"):
            @asset
            def a() -> Iterator[LGBMClassifier]:
                yield LGBMClassifier()


class TestEndToEnd:
    def test_unconfigured_lgbm_asset_materializes_and_loads_downstream(self, tmp_path: Path):
        @asset
        def model(context: AssetContext) -> LGBMClassifier:
            x = np.random.RandomState(0).rand(30, 2)
            return LGBMClassifier(n_estimators=2, verbose=-1).fit(x, (x[:, 0] > 0.5).astype(int))

        seen = {}

        @asset(dependencies=["model"])
        def scored(context: AssetContext, model: LGBMClassifier) -> pl.DataFrame:
            seen["model"] = model
            return pl.DataFrame({"n": [model.n_estimators]})

        app = Repipe(assets=[model, scored], base_storage_path=tmp_path)
        graph = build_graph(app.assets)

        asyncio.run(materialize(app.assets["model"], graph))
        # Nothing is shared between the two calls, so the dependency is loaded from disk with the inferred backend.
        asyncio.run(materialize(app.assets["scored"], graph))

        assert isinstance(seen["model"], LGBMClassifier)
