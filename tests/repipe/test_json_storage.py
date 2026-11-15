from __future__ import annotations

import asyncio
import collections
import datetime as dt
import enum
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from repipe.asset import AssetContext, MultiAssetOut, Output, PartitionKey, PartitionSettings, asset, multi_asset
from repipe.cli.selector import build_graph
from repipe.project import Repipe
from repipe.runtime import materialize
from repipe.storage import (
    JsonStorageConfig,
    Mode,
    ParquetStorageConfig,
    StorageType,
    get_storage_concrete,
)


class Side(enum.StrEnum):
    BACK = "back"


class Rank(enum.IntEnum):
    FIRST = 1


class Tags(list):
    pass


class LoudStr(str):
    pass


def _nested(depth: int) -> list:
    value: list = []
    for _ in range(depth):
        value = [value]
    return value


def _run(app: Repipe, name: str) -> None:
    asyncio.run(materialize(app.assets[name], build_graph(app.assets)))


def _load(app: Repipe, name: str) -> Any:
    config = app.assets[name].settings.storage
    return get_storage_concrete(config).load(config, app.assets[name])


def _app_returning(tmp_path: Path, value: Any, **storage: Any) -> Repipe:
    @asset(storage=JsonStorageConfig(**storage))
    def a(context: AssetContext) -> Output:
        return Output(value)

    return Repipe(assets=[a], base_storage_path=tmp_path)


def _written(tmp_path: Path) -> Path:
    (path,) = (tmp_path / "a").glob("*.json")
    return path


class TestConfig:
    def test_storage_type_makes_the_default_config(self):
        @asset(storage=StorageType.JSON)
        def a() -> dict: ...

        config = a.__asset__.outputs["a"].storage
        assert isinstance(config, JsonStorageConfig)
        assert (config.indent, config.sort_keys) == (2, False)

    @pytest.mark.parametrize("mode", [Mode.APPEND, Mode.APPEND_THEN_MERGE])
    def test_append_modes_are_rejected(self, mode):
        with pytest.raises(ValueError, match="JsonStorageConfig only supports Mode.OVERWRITE"):
            JsonStorageConfig(mode=mode)

    def test_row_fields_are_rejected(self):
        with pytest.raises(ValueError, match="is_duplicate"):
            JsonStorageConfig(is_duplicate=pl.col("a") > 1)
        with pytest.raises(ValueError, match="max_rows_per_file"):
            JsonStorageConfig(max_rows_per_file=10)

    def test_primary_key_is_rejected_and_names_the_storage(self):
        with pytest.raises(ValueError, match="primary_key has no meaning for JsonStorageConfig"):
            @asset(storage=StorageType.JSON, primary_key="id")
            def a() -> dict: ...

    def test_an_unpartitioned_generator_is_rejected(self):
        with pytest.raises(ValueError, match="return a single object"):
            @asset(storage=StorageType.JSON)
            def a():
                yield {"a": 1}

    def test_a_bare_dict_annotation_warns_and_is_not_inferred_as_json(self):
        with pytest.warns(UserWarning, match="pass `storage=` to choose another"):
            @asset
            def a() -> dict: ...

        assert isinstance(a.__asset__.outputs["a"].storage, ParquetStorageConfig)

    @pytest.mark.parametrize("annotation", [dict[str, int], list[str]])
    def test_a_subscripted_dict_or_list_falls_back_to_parquet_without_a_warning(self, annotation):
        # These are legitimate return shapes elsewhere (a multi-asset's dict), so there is nothing to warn about.
        def a(): ...
        a.__annotations__["return"] = annotation

        assert isinstance(asset(a).__asset__.outputs["a"].storage, ParquetStorageConfig)

    def test_forgetting_storage_fails_at_write_and_says_what_to_do(self, tmp_path: Path):
        @asset
        def a(context: AssetContext) -> dict[str, int]:
            return {"a": 1}

        app = Repipe(assets=[a], base_storage_path=tmp_path)

        with pytest.raises(TypeError, match="pass `storage=`"):
            _run(app, "a")


class TestRoundTrip:
    def test_a_nested_value_round_trips_exactly(self, tmp_path: Path):
        value = {
            "depth": 6, "rate": 0.1, "name": "naïve 日本", "on": True, "off": False, "none": None,
            "rates": [0.1, 0.2, [3, {"deep": []}]], "empty": {}, "big": 2 ** 70,
        }
        app = _app_returning(tmp_path, value)
        _run(app, "a")

        loaded = _load(app, "a")
        assert loaded == value
        assert {key: type(item) for key, item in loaded.items()} == {key: type(item) for key, item in value.items()}

    @pytest.mark.parametrize("value", [[1, 2, 3], "text", 7, 1.5, True, False, 0, 0.0, "", [], {}])
    def test_a_top_level_value_round_trips_with_its_type(self, tmp_path: Path, value):
        app = _app_returning(tmp_path, value)
        _run(app, "a")

        loaded = _load(app, "a")
        assert loaded == value and type(loaded) is type(value)  # True == 1 and 0.0 == 0, so == alone proves little

    def test_the_file_is_json_that_other_tools_can_read(self, tmp_path: Path):
        app = _app_returning(tmp_path, {"b": 1, "a": [1, 2], "u": "é"})
        _run(app, "a")

        text = _written(tmp_path).read_text(encoding="utf-8")
        assert json.loads(text) == {"b": 1, "a": [1, 2], "u": "é"}
        assert text.startswith('{\n  "b": 1,') and text.endswith("}\n")
        assert "é" in text  # not escaped to é

    def test_indent_none_writes_one_line(self, tmp_path: Path):
        app = _app_returning(tmp_path, {"a": 1, "b": [1, 2]}, indent=None)
        _run(app, "a")

        assert _written(tmp_path).read_text() == '{"a": 1, "b": [1, 2]}\n'

    def test_sort_keys(self, tmp_path: Path):
        app = _app_returning(tmp_path, {"b": 1, "a": 2}, indent=None, sort_keys=True)
        _run(app, "a")

        assert _written(tmp_path).read_text() == '{"a": 2, "b": 1}\n'

    def test_a_downstream_asset_receives_the_value(self, tmp_path: Path):
        @asset(storage=StorageType.JSON)
        def params(context: AssetContext) -> dict:
            return {"depth": 6}

        seen = {}

        @asset(dependencies=["params"])
        def used(context: AssetContext, params: dict) -> pl.DataFrame:
            seen["params"] = params
            return pl.DataFrame({"depth": [params["depth"]]})

        app = Repipe(assets=[params, used], base_storage_path=tmp_path)
        _run(app, "params")
        _run(app, "used")

        assert seen["params"] == {"depth": 6}

    def test_a_later_run_replaces_the_value(self, tmp_path: Path):
        runs = iter([{"run": 1}, {"run": 2}])

        @asset(storage=StorageType.JSON)
        def a(context: AssetContext) -> dict:
            return next(runs)

        app = Repipe(assets=[a], base_storage_path=tmp_path)
        _run(app, "a")
        _run(app, "a")

        assert _load(app, "a") == {"run": 2}
        assert len(list((tmp_path / "a").iterdir())) == 1

    def test_a_failed_run_keeps_the_previous_value(self, tmp_path: Path):
        fail = iter([False, True])

        @asset(storage=StorageType.JSON)
        def a(context: AssetContext) -> dict:
            if next(fail):
                raise RuntimeError("boom")
            return {"ok": True}

        app = Repipe(assets=[a], base_storage_path=tmp_path)
        _run(app, "a")
        with pytest.raises(RuntimeError):
            _run(app, "a")

        assert _load(app, "a") == {"ok": True}

    def test_nothing_written_yet_says_so(self, tmp_path: Path):
        @asset(storage=StorageType.JSON)
        def a() -> dict: ...

        app = Repipe(assets=[a], base_storage_path=tmp_path)

        with pytest.raises(FileNotFoundError, match="no object has been written yet for asset 'a'"):
            _load(app, "a")


class TestStrict:
    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ({"when": dt.datetime(2026, 1, 1)}, r"asset 'a''s value has a 'datetime.datetime' at \$\.when, which JSON can't store"),
            ({"rates": [1, np.int64(2)]}, r"a 'numpy.int64' at \$\.rates\[1\]"),
            ({"n": np.bool_(True)}, r"a 'numpy.bool.*' at \$\.n"),
            ({"s": {1, 2}}, r"a 'set' at \$\.s"),
            ({"b": b"x"}, r"a 'bytes' at \$\.b"),
            ({"d": pl.DataFrame({"a": [1]})}, r"a 'polars.*DataFrame' at \$\.d"),
            ({"a b": [object()]}, r"a 'object' at \$\['a b'\]\[0\]"),
        ],
    )
    def test_anything_but_json_types_names_the_asset_and_where(self, tmp_path: Path, value, message):
        app = _app_returning(tmp_path, value)

        with pytest.raises(TypeError, match=message):
            _run(app, "a")
        assert not list(tmp_path.rglob("*.json"))

    def test_a_tuple_is_refused_rather_than_quietly_becoming_a_list(self, tmp_path: Path):
        app = _app_returning(tmp_path, {"rates": (0.1, 0.2)})

        with pytest.raises(TypeError, match=r"a 'tuple' at \$\.rates.*use a list"):
            _run(app, "a")

    def test_a_non_string_key_is_refused_rather_than_quietly_becoming_one(self, tmp_path: Path):
        app = _app_returning(tmp_path, {"a": {1: "x"}})

        with pytest.raises(TypeError, match=r"the 'int' key 1 at \$\.a, but JSON keys are strings - convert it to a str"):
            _run(app, "a")

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_nan_and_infinity_are_refused(self, tmp_path: Path, bad):
        app = _app_returning(tmp_path, {"x": [1.0, bad]})

        with pytest.raises(ValueError, match=r"\$\.x\[1\], which isn't valid JSON"):
            _run(app, "a")

    def test_a_value_that_refers_back_to_itself_is_refused(self, tmp_path: Path):
        value: dict[str, Any] = {"a": []}
        value["a"].append(value)
        app = _app_returning(tmp_path, value)

        with pytest.raises(ValueError, match=r"refers back to itself at \$\.a\[0\]"):
            _run(app, "a")

    def test_the_same_object_twice_is_not_a_cycle(self, tmp_path: Path):
        shared = [1, 2]
        app = _app_returning(tmp_path, {"a": shared, "b": shared})
        _run(app, "a")

        assert _load(app, "a") == {"a": [1, 2], "b": [1, 2]}

    def test_none_is_refused(self, tmp_path: Path):
        @asset(storage=StorageType.JSON)
        def a(context: AssetContext) -> Any:
            return None

        app = Repipe(assets=[a], base_storage_path=tmp_path)

        with pytest.raises(TypeError, match="returned None"):
            _run(app, "a")

    def test_a_file_edited_to_hold_nan_is_refused_on_load(self, tmp_path: Path):
        app = _app_returning(tmp_path, {"x": 1})
        _run(app, "a")
        _written(tmp_path).write_text('{"x": NaN}')

        with pytest.raises(RuntimeError, match=r"asset 'a''s file .* can't be read back.*NaN isn't valid JSON"):
            _load(app, "a")


class TestPartitioned:
    def _app(self, tmp_path: Path, outputs):
        @asset(storage=StorageType.JSON, partition=PartitionSettings(columns={"year": pl.Int64}))
        def per_year(context: AssetContext) -> list[Output]:
            return outputs

        return Repipe(assets=[per_year], base_storage_path=tmp_path)

    def test_load_returns_a_dict_keyed_by_partition(self, tmp_path: Path):
        app = self._app(tmp_path, [Output({"y": 2024}, partition={"year": 2024}), Output({"y": 2025}, partition={"year": 2025})])
        _run(app, "per_year")

        settings = app.assets["per_year"].settings.partition
        assert _load(app, "per_year") == {
            PartitionKey.from_mapping(settings, {"year": 2024}): {"y": 2024},
            PartitionKey.from_mapping(settings, {"year": 2025}): {"y": 2025},
        }
        assert sorted(path.parent.name for path in (tmp_path / "per_year").rglob("*.json")) == ["year=2024", "year=2025"]

    def test_overwrite_replaces_only_the_partitions_it_wrote(self, tmp_path: Path):
        _run(self._app(tmp_path, [Output("a", partition={"year": 2024}), Output("b", partition={"year": 2025})]), "per_year")
        second = self._app(tmp_path, [Output("c", partition={"year": 2025})])
        _run(second, "per_year")

        assert sorted(_load(second, "per_year").values()) == ["a", "c"]

    def test_a_partitioned_asset_needs_a_key(self, tmp_path: Path):
        @asset(storage=StorageType.JSON, partition=PartitionSettings(columns={"year": pl.Int64}))
        def per_year(context: AssetContext) -> Output:
            return Output({"y": 1})

        app = Repipe(assets=[per_year], base_storage_path=tmp_path)

        with pytest.raises(ValueError, match="each object needs a key"):
            _run(app, "per_year")


class TestMultiAsset:
    def test_a_json_output_sits_beside_a_frame_output(self, tmp_path: Path):
        @multi_asset(outputs={"rows": MultiAssetOut(), "stats": MultiAssetOut(storage=StorageType.JSON)})
        def parse(context: AssetContext) -> dict[str, Any]:
            return {"rows": pl.DataFrame({"a": [1, 2]}), "stats": {"n": 2}}

        app = Repipe(assets=[parse], base_storage_path=tmp_path)
        _run(app, "rows")

        assert _load(app, "stats") == {"n": 2}
        assert _load(app, "rows").collect()["a"].to_list() == [1, 2]


class TestExactTypes:
    """ `json` stores a subclass as its base type, so what loads would not be what was written. """

    @pytest.mark.parametrize(
        ("value", "base"),
        [
            (Side.BACK, "str"),
            (Rank.FIRST, "int"),
            (np.float64(0.5), "float"),
            (Tags(), "list"),
            (collections.defaultdict(int, a=1), "dict"),
            (collections.OrderedDict(a=1), "dict"),
            (LoudStr("x"), "str"),
        ],
    )
    def test_a_subclass_of_a_json_type_is_refused_and_says_which(self, tmp_path: Path, value, base):
        app = _app_returning(tmp_path, {"v": value})

        with pytest.raises(TypeError, match=rf"at \$\.v, which JSON can't store - it is a subclass of {base}"):
            _run(app, "a")
        assert not list(tmp_path.rglob("*.json"))

    def test_a_str_subclass_key_is_refused(self, tmp_path: Path):
        app = _app_returning(tmp_path, {LoudStr("a"): 1})

        with pytest.raises(TypeError, match=r"a str subclass would load back as a plain str"):
            _run(app, "a")

    def test_the_numpy_scalars_are_refused_alike(self, tmp_path: Path):
        for scalar in (np.float64(0.5), np.float32(0.5), np.int64(1), np.bool_(True)):
            with pytest.raises(TypeError):
                _run(_app_returning(tmp_path, [scalar]), "a")

    def test_the_message_for_numpy_bool_does_not_call_it_bool(self, tmp_path: Path):
        with pytest.raises(TypeError) as error:
            _run(_app_returning(tmp_path, {"n": np.bool_(True)}), "a")

        assert "'bool'" not in str(error.value)


class TestLimits:
    def test_a_lone_surrogate_names_the_asset_and_where(self, tmp_path: Path):
        app = _app_returning(tmp_path, {"s": ["fine", "bad \udce9"]})

        with pytest.raises(ValueError, match=r"asset 'a''s value has a string at \$\.s\[1\] with the lone surrogate"):
            _run(app, "a")

    def test_a_lone_surrogate_in_a_key_is_named_too(self, tmp_path: Path):
        app = _app_returning(tmp_path, {"k\udce9": 1})

        with pytest.raises(ValueError, match="lone surrogate"):
            _run(app, "a")

    def test_valid_non_ascii_text_is_fine(self, tmp_path: Path):
        app = _app_returning(tmp_path, {"日本": ["é", "😀"]})
        _run(app, "a")

        assert _load(app, "a") == {"日本": ["é", "😀"]}

    def test_nesting_deeper_than_the_interpreter_stack_that_json_still_handles_is_stored(self, tmp_path: Path):
        depth = 3000
        app = _app_returning(tmp_path, _nested(depth), indent=None)
        _run(app, "a")

        loaded = _load(app, "a")
        for _ in range(depth):
            (loaded,) = loaded
        assert loaded == []

    def test_nesting_too_deep_for_json_itself_names_the_asset(self, tmp_path: Path):
        app = _app_returning(tmp_path, _nested(200_000))

        with pytest.raises(ValueError, match="asset 'a''s value is nested too deeply to write as JSON"):
            _run(app, "a")

    def test_an_integer_past_the_interpreters_digit_limit_names_the_asset_and_where(self, tmp_path: Path):
        before = sys.get_int_max_str_digits()
        sys.set_int_max_str_digits(4300)
        try:
            app = _app_returning(tmp_path, {"n": [10 ** 5000]})
            with pytest.raises(ValueError, match=r"an integer at \$\.n\[0\] that is too long"):
                _run(app, "a")
        finally:
            sys.set_int_max_str_digits(before)

    def test_a_big_integer_inside_the_limit_round_trips(self, tmp_path: Path):
        app = _app_returning(tmp_path, [10 ** 1000, -(10 ** 1000)])
        _run(app, "a")

        assert _load(app, "a") == [10 ** 1000, -(10 ** 1000)]


class TestReadingStrictly:
    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ('{"x": 1e999}', "doesn't fit in a float"),
            ("[-1e400]", "doesn't fit in a float"),
            ("null", "holds null"),
            ('{"x": Infinity}', "Infinity isn't valid JSON"),
        ],
    )
    def test_a_file_edited_into_something_this_never_writes_is_refused(self, tmp_path: Path, text, message):
        app = _app_returning(tmp_path, {"x": 1})
        _run(app, "a")
        _written(tmp_path).write_text(text)

        with pytest.raises(RuntimeError, match=rf"asset 'a''s file .* can't be read back.*{message}"):
            _load(app, "a")

    def test_the_largest_float_and_a_tiny_one_still_load(self, tmp_path: Path):
        app = _app_returning(tmp_path, [sys.float_info.max, 5e-324, -0.0])
        _run(app, "a")

        assert _load(app, "a") == [sys.float_info.max, 5e-324, -0.0]


class TestIndent:
    @pytest.mark.parametrize("indent", [-1, True, 1.5, "2"])
    def test_a_bad_indent_is_refused(self, indent):
        with pytest.raises(ValueError):
            JsonStorageConfig(indent=indent)

    def test_zero_is_allowed(self, tmp_path: Path):
        app = _app_returning(tmp_path, {"a": [1]}, indent=0)
        _run(app, "a")

        assert _written(tmp_path).read_text() == '{\n"a": [\n1\n]\n}\n'
