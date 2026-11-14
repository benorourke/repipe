from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from lightgbm import LGBMClassifier

from repipe.asset import AssetContext, MultiAssetOut, Output, PartitionKey, PartitionSettings, asset, multi_asset
from repipe.cli.selector import build_graph
from repipe.project import Repipe
from repipe.runtime import materialize
from repipe.storage import (
    LGBMStorageConfig,
    Mode,
    ParquetStorageConfig,
    PickleStorageConfig,
    StorageType,
    get_storage_concrete,
)


@dataclass
class Tuned:
    """ Module level, because pickle stores a class by reference. """
    depth: int
    rates: list[float]


class TwoArgumentError(Exception):
    """ Pickles, but can't be unpickled: the exception stores one argument and its __init__ wants two. """

    def __init__(self, a: int, b: int):
        super().__init__(a)


class Unreducible:
    def __reduce__(self):
        raise ValueError("not for pickling")


def _deeply_nested(depth: int = 100_000) -> list:
    value: list = []
    for _ in range(depth):
        value = [value]
    return value


def _run(app: Repipe, name: str) -> None:
    asyncio.run(materialize(app.assets[name], build_graph(app.assets)))


def _load(app: Repipe, name: str) -> Any:
    config = app.assets[name].settings.storage
    return get_storage_concrete(config).load(config, app.assets[name])


class TestConfig:
    def test_storage_type_makes_the_default_config(self):
        @asset(storage=StorageType.PICKLE)
        def a() -> dict: ...

        assert isinstance(a.__asset__.outputs["a"].storage, PickleStorageConfig)

    @pytest.mark.parametrize("mode", [Mode.APPEND, Mode.APPEND_THEN_MERGE])
    def test_append_modes_are_rejected(self, mode):
        with pytest.raises(ValueError, match="only supports Mode.OVERWRITE"):
            PickleStorageConfig(mode=mode)

    def test_is_duplicate_is_rejected(self):
        with pytest.raises(ValueError, match="is_duplicate"):
            PickleStorageConfig(is_duplicate=pl.col("a") > 1)

    def test_max_rows_per_file_is_rejected(self):
        with pytest.raises(ValueError, match="max_rows_per_file"):
            PickleStorageConfig(max_rows_per_file=10)

    def test_primary_key_is_rejected_and_names_the_storage(self):
        with pytest.raises(ValueError, match="primary_key has no meaning for PickleStorageConfig"):
            @asset(storage=StorageType.PICKLE, primary_key="id")
            def a() -> dict: ...

    def test_an_unpartitioned_generator_is_rejected(self):
        with pytest.raises(ValueError, match="return a single object"):
            @asset(storage=StorageType.PICKLE)
            def a():
                yield {"a": 1}

    def test_lgbm_config_keeps_its_own_wording(self):
        with pytest.raises(ValueError, match="a trained model has no parts to append or merge"):
            LGBMStorageConfig(mode=Mode.APPEND)

    def test_a_pickle_is_never_inferred_from_an_annotation(self):
        with pytest.warns(UserWarning, match="defaulting to Parquet"):
            @asset
            def a() -> Tuned: ...

        assert isinstance(a.__asset__.outputs["a"].storage, ParquetStorageConfig)


class TestRoundTrip:
    def test_a_dict_round_trips_and_a_downstream_asset_receives_it(self, tmp_path: Path):
        @asset(storage=StorageType.PICKLE)
        def params(context: AssetContext) -> dict:
            return {"depth": 6, "rates": (0.1, 0.2), "when": None}

        seen = {}

        @asset(dependencies=["params"])
        def used(context: AssetContext, params: dict) -> pl.DataFrame:
            seen["params"] = params
            return pl.DataFrame({"depth": [params["depth"]]})

        app = Repipe(assets=[params, used], base_storage_path=tmp_path)
        _run(app, "params")
        _run(app, "used")  # nothing is shared between the two runs, so it comes back off disk

        assert seen["params"] == {"depth": 6, "rates": (0.1, 0.2), "when": None}

    def test_a_custom_class_round_trips(self, tmp_path: Path):
        @asset(storage=StorageType.PICKLE)
        def tuned(context: AssetContext) -> Tuned:
            return Tuned(depth=4, rates=[0.5])

        app = Repipe(assets=[tuned], base_storage_path=tmp_path)
        _run(app, "tuned")

        assert _load(app, "tuned") == Tuned(depth=4, rates=[0.5])

    def test_a_list_has_to_be_wrapped_in_output(self, tmp_path: Path):
        # A bare list or tuple return means "a list of Outputs", so it can't be the value itself.
        @asset(storage=StorageType.PICKLE)
        def wrapped(context: AssetContext) -> Output:
            return Output([1, 2, 3])

        @asset(storage=StorageType.PICKLE)
        def bare(context: AssetContext) -> list:
            return [1, 2, 3]

        app = Repipe(assets=[wrapped, bare], base_storage_path=tmp_path)
        _run(app, "wrapped")
        assert _load(app, "wrapped") == [1, 2, 3]
        with pytest.raises(TypeError, match="every item must be an Output"):
            _run(app, "bare")

    def test_a_later_run_replaces_the_value(self, tmp_path: Path):
        runs = iter([{"run": 1}, {"run": 2}])

        @asset(storage=StorageType.PICKLE)
        def state(context: AssetContext) -> dict:
            return next(runs)

        app = Repipe(assets=[state], base_storage_path=tmp_path)
        _run(app, "state")
        _run(app, "state")

        assert _load(app, "state") == {"run": 2}
        assert len(list((tmp_path / "state").iterdir())) == 1

    def test_an_asset_can_read_its_own_previous_value(self, tmp_path: Path):
        @asset(storage=StorageType.PICKLE, dependencies=["counter"])
        def counter(context: AssetContext, counter: dict | None) -> dict:
            return {"n": (counter["n"] if counter else 0) + 1}

        app = Repipe(assets=[counter], base_storage_path=tmp_path)
        _run(app, "counter")
        _run(app, "counter")

        assert _load(app, "counter") == {"n": 2}

    def test_nothing_written_yet_says_so(self, tmp_path: Path):
        @asset(storage=StorageType.PICKLE)
        def a() -> dict: ...

        app = Repipe(assets=[a], base_storage_path=tmp_path)

        with pytest.raises(FileNotFoundError, match="no object has been written yet for asset 'a'"):
            _load(app, "a")

    def test_a_failed_run_keeps_the_previous_value(self, tmp_path: Path):
        fail = iter([False, True])

        @asset(storage=StorageType.PICKLE)
        def state(context: AssetContext) -> dict:
            if next(fail):
                raise RuntimeError("boom")
            return {"ok": True}

        app = Repipe(assets=[state], base_storage_path=tmp_path)
        _run(app, "state")
        with pytest.raises(RuntimeError):
            _run(app, "state")

        assert _load(app, "state") == {"ok": True}


class TestErrors:
    def test_an_unpicklable_value_names_the_asset(self, tmp_path: Path):
        @asset(storage=StorageType.PICKLE)
        def a(context: AssetContext) -> Any:
            return {"f": lambda: 1}

        app = Repipe(assets=[a], base_storage_path=tmp_path)

        with pytest.raises(TypeError, match="asset 'a' returned a 'dict' that can't be pickled"):
            _run(app, "a")
        assert not (tmp_path / "a").exists() or not any((tmp_path / "a").iterdir())

    def test_none_is_refused(self, tmp_path: Path):
        @asset(storage=StorageType.PICKLE)
        def a(context: AssetContext) -> Any:
            return None

        app = Repipe(assets=[a], base_storage_path=tmp_path)

        with pytest.raises(TypeError, match="returned None"):
            _run(app, "a")

    def test_lgbm_still_refuses_anything_that_is_not_a_model(self, tmp_path: Path):
        @asset(storage=StorageType.LGBM)
        def a(context: AssetContext) -> Any:
            return {"not": "a model"}

        app = Repipe(assets=[a], base_storage_path=tmp_path)

        with pytest.raises(TypeError, match="only persists a trained LGBM model"):
            _run(app, "a")


class TestPartitioned:
    def _app(self, tmp_path: Path, outputs):
        @asset(storage=StorageType.PICKLE, partition=PartitionSettings(columns={"year": pl.Int64}))
        def per_year(context: AssetContext) -> list[Output]:
            return outputs

        return Repipe(assets=[per_year], base_storage_path=tmp_path)

    def test_load_returns_a_dict_keyed_by_partition(self, tmp_path: Path):
        app = self._app(tmp_path, [Output({"y": 2024}, partition={"year": 2024}), Output({"y": 2025}, partition={"year": 2025})])
        _run(app, "per_year")

        settings = app.assets["per_year"].settings.partition
        loaded = _load(app, "per_year")
        assert loaded == {
            PartitionKey.from_mapping(settings, {"year": 2024}): {"y": 2024},
            PartitionKey.from_mapping(settings, {"year": 2025}): {"y": 2025},
        }

    def test_overwrite_replaces_only_the_partitions_it_wrote(self, tmp_path: Path):
        first = self._app(tmp_path, [Output("a", partition={"year": 2024}), Output("b", partition={"year": 2025})])
        _run(first, "per_year")
        second = self._app(tmp_path, [Output("c", partition={"year": 2025})])
        _run(second, "per_year")

        assert sorted(_load(second, "per_year").values()) == ["a", "c"]

    def test_a_partitioned_asset_needs_a_key(self, tmp_path: Path):
        @asset(storage=StorageType.PICKLE, partition=PartitionSettings(columns={"year": pl.Int64}))
        def per_year(context: AssetContext) -> Output:
            return Output({"y": 1})

        app = Repipe(assets=[per_year], base_storage_path=tmp_path)

        with pytest.raises(ValueError, match="each object needs a key"):
            _run(app, "per_year")

    def test_an_unpartitioned_asset_refuses_a_key(self, tmp_path: Path):
        @asset(storage=StorageType.PICKLE)
        def a(context: AssetContext) -> Output:
            return Output({"y": 1}, partition={"year": 2024})

        app = Repipe(assets=[a], base_storage_path=tmp_path)

        with pytest.raises(ValueError, match="isn't partitioned"):
            _run(app, "a")


class TestMultiAsset:
    def test_a_pickle_output_sits_beside_a_frame_output(self, tmp_path: Path):
        @multi_asset(outputs={"rows": MultiAssetOut(), "stats": MultiAssetOut(storage=StorageType.PICKLE)})
        def parse(context: AssetContext) -> dict[str, Any]:
            return {"rows": pl.DataFrame({"a": [1, 2]}), "stats": {"n": 2}}

        app = Repipe(assets=[parse], base_storage_path=tmp_path)
        _run(app, "rows")

        assert _load(app, "stats") == {"n": 2}
        assert _load(app, "rows").collect()["a"].to_list() == [1, 2]


class TestLgbmIsUnchanged:
    def test_a_model_round_trips_through_the_shared_base(self, tmp_path: Path):
        @asset
        def model(context: AssetContext) -> LGBMClassifier:
            fitted = LGBMClassifier(n_estimators=2, min_child_samples=1, verbose=-1)
            return fitted.fit([[0], [1], [0], [1]], [0, 1, 0, 1])

        app = Repipe(assets=[model], base_storage_path=tmp_path)
        _run(app, "model")

        assert _load(app, "model").predict([[1]]).tolist() == [1]


class TestEmptyWrites:
    def _twice(self, tmp_path: Path, second: Any) -> Repipe:
        runs = iter([{"v": 1}, second])

        @asset(storage=StorageType.PICKLE)
        def state(context: AssetContext) -> Any:
            return next(runs)

        app = Repipe(assets=[state], base_storage_path=tmp_path)
        _run(app, "state")
        return app

    @pytest.mark.parametrize("empty", [[], ()])
    def test_an_empty_list_or_tuple_return_is_refused_not_a_silent_delete(self, tmp_path: Path, empty):
        app = self._twice(tmp_path, empty)

        with pytest.raises(ValueError, match=r"asset 'state' wrote nothing.*return Output\(\[\]\)"):
            _run(app, "state")
        assert _load(app, "state") == {"v": 1}

    def test_an_empty_list_can_be_stored_by_wrapping_it(self, tmp_path: Path):
        app = self._twice(tmp_path, Output([]))
        _run(app, "state")

        assert _load(app, "state") == []

    def test_a_partitioned_run_that_writes_nothing_leaves_the_partitions_alone(self, tmp_path: Path):
        runs = iter([[Output("a", partition={"year": 2024})], []])

        @asset(storage=StorageType.PICKLE, partition=PartitionSettings(columns={"year": pl.Int64}))
        def per_year(context: AssetContext) -> list[Output]:
            return next(runs)

        app = Repipe(assets=[per_year], base_storage_path=tmp_path)
        _run(app, "per_year")
        _run(app, "per_year")

        assert list(_load(app, "per_year").values()) == ["a"]


class TestPickleFailures:
    def test_a_value_that_cannot_be_unpickled_is_refused_when_written(self, tmp_path: Path):
        @asset(storage=StorageType.PICKLE)
        def a(context: AssetContext) -> Any:
            return {"e": TwoArgumentError(1, 2)}

        app = Repipe(assets=[a], base_storage_path=tmp_path)

        with pytest.raises(TypeError, match="asset 'a' returned a 'dict' that pickles but can't be unpickled again"):
            _run(app, "a")
        assert not list(tmp_path.rglob("*.pkl"))

    @pytest.mark.parametrize("value", [Unreducible(), _deeply_nested()], ids=["reduce raises", "too deep"])
    def test_any_failure_to_pickle_names_the_asset(self, tmp_path: Path, value):
        @asset(storage=StorageType.PICKLE)
        def a(context: AssetContext) -> Any:
            return Output(value)

        app = Repipe(assets=[a], base_storage_path=tmp_path)

        with pytest.raises(TypeError, match="asset 'a' returned a '.*' that can't be pickled"):
            _run(app, "a")

    def test_a_file_that_cannot_be_read_back_names_the_asset_and_how_to_recover(self, tmp_path: Path):
        @asset(storage=StorageType.PICKLE)
        def a(context: AssetContext) -> dict:
            return {"a": 1}

        app = Repipe(assets=[a], base_storage_path=tmp_path)
        _run(app, "a")
        (path,) = (tmp_path / "a").glob("*.pkl")
        path.write_bytes(path.read_bytes()[:5])

        with pytest.raises(RuntimeError, match=r"asset 'a''s file .*\.pkl can't be read back.*`repipe clear a`"):
            _load(app, "a")

    def test_a_model_file_that_cannot_be_read_back_does_too(self, tmp_path: Path):
        @asset(storage=StorageType.LGBM)
        def a(context: AssetContext) -> LGBMClassifier:
            return LGBMClassifier(n_estimators=2, min_child_samples=1, verbose=-1).fit([[0], [1], [0], [1]], [0, 1, 0, 1])

        app = Repipe(assets=[a], base_storage_path=tmp_path)
        _run(app, "a")
        (path,) = (tmp_path / "a").glob("*.pkl")
        path.write_bytes(b"not a pickle")

        with pytest.raises(RuntimeError, match="asset 'a''s file .* can't be read back"):
            _load(app, "a")


class TestMessages:
    def test_a_model_asset_returning_none_keeps_the_hint(self, tmp_path: Path):
        @asset(storage=StorageType.LGBM)
        def a(context: AssetContext) -> Any:
            return None

        app = Repipe(assets=[a], base_storage_path=tmp_path)

        with pytest.raises(TypeError, match="only persists a trained LGBM model - pass `storage=`"):
            _run(app, "a")

    def test_a_pickle_generator_error_does_not_say_it_was_inferred(self):
        with pytest.raises(ValueError) as error:
            @asset(storage=StorageType.PICKLE)
            def a():
                yield {"a": 1}

        assert "inferred" not in str(error.value)
