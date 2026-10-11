from pathlib import Path

import polars as pl

from repipe.asset.asset import Asset, AssetContext, asset
from repipe.project import Repipe


DEFERRED_ASSET_FIELDS = {
}


class TestRepipe:
    class TestConstructAssetsFromFuncs:
        def test_empty_list_returns_empty_dict(self, tmp_path: Path):
            assert Repipe.construct_assets_from_funcs([], tmp_path) == {}

        def test_single_asset_is_keyed_by_name_with_expected_fields(self, tmp_path: Path):
            @asset
            def my_asset(context: AssetContext) -> pl.LazyFrame:
                return pl.DataFrame().lazy()

            assets = Repipe.construct_assets_from_funcs([my_asset], tmp_path)

            assert list(assets.keys()) == ["my_asset"]
            built = assets["my_asset"]
            assert built.name == "my_asset"
            assert built.base_storage_path == tmp_path
            assert built.python_func == "my_asset"

        def test_no_asset_field_is_silently_forgotten(self, tmp_path: Path):
            @asset
            def my_asset(context: AssetContext) -> pl.LazyFrame:
                return pl.DataFrame().lazy()

            built = Repipe.construct_assets_from_funcs([my_asset], tmp_path)["my_asset"]

            assert built.model_fields_set == Asset.model_fields.keys() - DEFERRED_ASSET_FIELDS
