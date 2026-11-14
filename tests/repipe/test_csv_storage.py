from __future__ import annotations

import asyncio
import datetime as dt
import decimal
import random
from pathlib import Path
from typing import Any, Iterator

import polars as pl
import pytest

from repipe.asset import AssetContext, Output, PartitionSettings, asset
from repipe.cli.selector import build_graph
from repipe.project import Repipe
from repipe.runtime import materialize
from repipe.storage import CsvStorageConfig, Mode, get_storage_concrete
from repipe.storage.storage import CsvPolars


def _app(tmp_path: Path, frames: list[Any], *, partition: PartitionSettings | None = None, primary_key=None, **storage: Any) -> Repipe:
    """ One asset, `a`, that yields each of `frames` (an `Output`, or a bare frame) as a chunk. """
    @asset(storage=CsvStorageConfig(**storage), partition=partition, primary_key=primary_key)
    def a(context: AssetContext) -> Iterator[Output | pl.DataFrame]:
        yield from frames

    return Repipe(assets=[a], base_storage_path=tmp_path)


def _run(app: Repipe, name: str = "a") -> None:
    asyncio.run(materialize(app.assets[name], build_graph(app.assets)))


def _load(app: Repipe, name: str = "a") -> pl.DataFrame:
    config = app.assets[name].settings.storage
    return get_storage_concrete(config).load(config, app.assets[name]).collect()


def _roundtrip(tmp_path: Path, frame: pl.DataFrame) -> pl.DataFrame:
    app = _app(tmp_path, [frame])
    _run(app)
    return _load(app)


def _lone(tmp_path: Path, pattern: str = "*.csv") -> Path:
    (path,) = (tmp_path / "a").glob(pattern)
    return path


_FLOAT32 = pl.DataFrame({"a": [random.Random(0).uniform(-1e6, 1e6) for _ in range(500)] + [1e-38, 3.4e38, 1.1]}, schema={"a": pl.Float32})

ROUND_TRIPS = {
    "int": pl.DataFrame({"a": [1, 2, None]}),
    "int64 extremes": pl.DataFrame({"a": [2 ** 63 - 1, -2 ** 63]}),
    "uint8": pl.DataFrame({"a": [0, 255]}, schema={"a": pl.UInt8}),
    "uint64": pl.DataFrame({"a": [2 ** 64 - 1]}, schema={"a": pl.UInt64}),
    "float64": pl.DataFrame({"a": [1.5, None, float("nan"), float("inf"), -0.0, 1e-300, 0.1 + 0.2]}),
    "float32": _FLOAT32,
    "float16": pl.DataFrame({"a": [1.5, 2.25]}, schema={"a": pl.Float16}),
    "string": pl.DataFrame({"a": ["x,y", 'he said "hi"', "line1\nline2", "é日本😀", "", None]}),
    "string that looks numeric": pl.DataFrame({"a": ["007", "1e3", "NaN", "1_000", " 5"]}),
    "string that looks boolean": pl.DataFrame({"a": ["true", "false", "TRUE", "True"]}),
    "boolean": pl.DataFrame({"a": [True, False, None]}),
    "date": pl.DataFrame({"a": [dt.date(2026, 1, 1), dt.date(1, 1, 1), None]}),
    "datetime us": pl.DataFrame({"a": [dt.datetime(2026, 1, 1, 12, 30, 15, 123456), None]}),
    "datetime ms": pl.DataFrame({"a": [dt.datetime(2026, 1, 1, 12, 30, 15, 123000)]}, schema={"a": pl.Datetime("ms")}),
    "datetime ns": pl.DataFrame({"a": pl.Series([dt.datetime(2026, 1, 1, 12, 30, 15, 123456)]).dt.cast_time_unit("ns")}),
    "datetime utc": pl.DataFrame({"a": pl.Series([dt.datetime(2026, 1, 1, 12), None]).dt.replace_time_zone("UTC")}),
    "datetime london across dst": pl.DataFrame({
        "a": pl.Series([dt.datetime(2026, 7, 1, 12), dt.datetime(2026, 1, 1, 12), dt.datetime(2026, 10, 25, 1, 30)])
        .dt.replace_time_zone("Europe/London", ambiguous="earliest"),
    }),
    "time": pl.DataFrame({"a": [dt.time(18, 30), dt.time(0, 0, 0, 1), None]}),
    "categorical": pl.DataFrame({"a": ["x", "y", None]}, schema={"a": pl.Categorical}),
    "enum": pl.DataFrame({"a": ["x", "y", None]}, schema={"a": pl.Enum(["x", "y", "z"])}),
    "decimal": pl.DataFrame({"a": [decimal.Decimal("1.10"), decimal.Decimal("-0.05"), None]}, schema={"a": pl.Decimal(10, 2)}),
    "null typed": pl.DataFrame({"a": [None, None]}),
    "datetime kolkata in 1900 (offset has seconds)": pl.DataFrame({
        "a": pl.Series([dt.datetime(1900, 1, 1, 12), dt.datetime(2026, 1, 1, 12)]).dt.replace_time_zone("Asia/Kolkata"),
    }),
    "datetime monrovia in 1970 (offset has seconds)": pl.DataFrame({
        "a": pl.Series([dt.datetime(1970, 1, 1, 12)]).dt.replace_time_zone("Africa/Monrovia"),
    }),
    "datetime kathmandu": pl.DataFrame({"a": pl.Series([dt.datetime(2026, 6, 1, 12)]).dt.replace_time_zone("Asia/Kathmandu")}),
    "awkward column names": pl.DataFrame({"a,b": [1], "line\nbreak": [2], " pad ": [3], "": [4], "it's": [5]}),
    "empty with a schema": pl.DataFrame({"a": []}, schema={"a": pl.Int64}),
    "many columns": pl.DataFrame({
        "id": [1, 2], "when": [dt.date(2026, 1, 1), dt.date(2026, 1, 2)], "code": ["007", "010"], "ok": [True, False],
        "price": [2.5, None],
    }),
}


class TestRoundTrip:
    @pytest.mark.parametrize("frame", ROUND_TRIPS.values(), ids=ROUND_TRIPS.keys())
    def test_comes_back_exactly(self, tmp_path: Path, frame: pl.DataFrame):
        back = _roundtrip(tmp_path, frame)

        assert back.schema == frame.schema
        assert back.equals(frame, null_equal=True)

    def test_a_date_is_a_date_not_text(self, tmp_path: Path):
        back = _roundtrip(tmp_path, pl.DataFrame({"d": [dt.date(2026, 11, 13)]}))

        assert back.schema["d"] == pl.Date and back["d"][0] == dt.date(2026, 11, 13)

    def test_numeric_looking_text_stays_text(self, tmp_path: Path):
        back = _roundtrip(tmp_path, pl.DataFrame({"code": ["007", "1e3"]}))

        assert back["code"].to_list() == ["007", "1e3"]

    def test_text_that_looks_boolean_stays_text(self, tmp_path: Path):
        back = _roundtrip(tmp_path, pl.DataFrame({"s": ["true", "TRUE"]}))

        assert back.schema["s"] == pl.String and back["s"].to_list() == ["true", "TRUE"]

    def test_the_file_is_an_ordinary_csv(self, tmp_path: Path):
        _roundtrip(tmp_path, pl.DataFrame({"a": [1, 2], "d": [dt.date(2026, 1, 1), dt.date(2026, 1, 2)]}))

        assert _lone(tmp_path).read_text() == "a,d\n1,2026-01-01\n2,2026-01-02\n"

    def test_a_downstream_asset_receives_the_typed_frame(self, tmp_path: Path):
        @asset(storage=CsvStorageConfig())
        def source(context: AssetContext) -> pl.DataFrame:
            return pl.DataFrame({"d": [dt.date(2026, 1, 1)], "code": ["007"]})

        seen = {}

        @asset(dependencies=["source"])
        def used(context: AssetContext, source: pl.LazyFrame) -> pl.DataFrame:
            seen["schema"] = source.collect_schema()
            return source.collect()

        app = Repipe(assets=[source, used], base_storage_path=tmp_path)
        _run(app, "source")
        _run(app, "used")

        assert seen["schema"] == pl.Schema({"d": pl.Date, "code": pl.String})

    def test_an_asset_in_a_schema_directory(self, tmp_path: Path):
        @asset(storage=CsvStorageConfig(), schemas=["mart"])
        def a(context: AssetContext) -> pl.DataFrame:
            return pl.DataFrame({"d": [dt.date(2026, 1, 1)]})

        app = Repipe(assets=[a], base_storage_path=tmp_path)
        _run(app)

        storage = get_storage_concrete(app.assets["a"].settings.storage)
        assert _load(app).schema["d"] == pl.Date
        assert storage.schema_path(app.assets["a"]).parent == storage.asset_dir(app.assets["a"])
        assert storage.schema_path(app.assets["a"]).is_file()


class TestAcrossWrites:
    def test_an_int_column_then_an_all_null_chunk_stays_int(self, tmp_path: Path):
        app = _app(
            tmp_path,
            [pl.DataFrame({"a": [1, 2]}), pl.DataFrame({"a": [None, None]}, schema={"a": pl.Int64})],
            mode=Mode.APPEND,
        )
        _run(app)

        back = _load(app)
        assert back.schema["a"] == pl.Int64 and back["a"].to_list() == [1, 2, None, None]

    def test_append_then_merge_keeps_every_type(self, tmp_path: Path):
        frame = pl.DataFrame({"id": [1, 2], "d": [dt.date(2026, 1, 1), dt.date(2026, 1, 2)], "code": ["007", "010"]})
        second = pl.DataFrame({"id": [2, 3], "d": [dt.date(2026, 2, 2), dt.date(2026, 1, 3)], "code": ["020", "030"]})
        app = _app(tmp_path, [frame], mode=Mode.APPEND_THEN_MERGE, primary_key="id")
        _run(app)
        _run(_app(tmp_path, [second], mode=Mode.APPEND_THEN_MERGE, primary_key="id"))

        back = _load(app).sort("id")
        assert back.schema == frame.schema
        assert back.rows() == [(1, dt.date(2026, 1, 1), "007"), (2, dt.date(2026, 2, 2), "020"), (3, dt.date(2026, 1, 3), "030")]
        assert len(list((tmp_path / "a").glob("*.csv"))) == 1

    def test_overwrite_may_change_the_schema(self, tmp_path: Path):
        _run(_app(tmp_path, [pl.DataFrame({"a": [1]})]))
        app = _app(tmp_path, [pl.DataFrame({"b": ["x"], "d": [dt.date(2026, 1, 1)]})])
        _run(app)

        assert _load(app).schema == pl.Schema({"b": pl.String, "d": pl.Date})

    def test_append_with_a_different_schema_is_refused_naming_the_change(self, tmp_path: Path):
        _run(_app(tmp_path, [pl.DataFrame({"a": [1]})], mode=Mode.APPEND))
        app = _app(tmp_path, [pl.DataFrame({"a": ["x"], "b": [1]})], mode=Mode.APPEND)

        with pytest.raises(ValueError, match=r"different schema.*'b' added.*'a' was Int64, now String.*repipe clear a"):
            _run(app)
        assert _load(app).schema == pl.Schema({"a": pl.Int64})  # the earlier data and its record are as they were

    def test_chunks_of_one_run_must_share_a_schema(self, tmp_path: Path):
        app = _app(tmp_path, [pl.DataFrame({"a": [1]}), pl.DataFrame({"a": [1.5]})])

        with pytest.raises(ValueError, match=r"different schemas.*'a' was Int64, now Float64"):
            _run(app)

    def test_partitioned_overwrite_refuses_a_new_schema_beside_untouched_partitions(self, tmp_path: Path):
        partition = PartitionSettings(columns={"year": pl.Int64})
        _run(_app(tmp_path, [Output(pl.DataFrame({"a": [1]}), partition={"year": 2024})], partition=partition))
        app = _app(tmp_path, [Output(pl.DataFrame({"a": ["x"]}), partition={"year": 2025})], partition=partition)

        with pytest.raises(ValueError, match="different schema"):
            _run(app)

    def test_partitioned_round_trip_keeps_the_partition_column_types(self, tmp_path: Path):
        partition = PartitionSettings(columns={"day": pl.Date, "code": pl.String})
        frames = [
            Output(pl.DataFrame({"x": [1.5]}), partition={"day": dt.date(2026, 1, 1), "code": "007"}),
            Output(pl.DataFrame({"x": [2.5]}), partition={"day": dt.date(2026, 1, 2), "code": "010"}),
        ]
        app = _app(tmp_path, frames, partition=partition)
        _run(app)

        back = _load(app).sort("day")
        assert back.schema == pl.Schema({"x": pl.Float64, "day": pl.Date, "code": pl.String})
        assert back.rows() == [(1.5, dt.date(2026, 1, 1), "007"), (2.5, dt.date(2026, 1, 2), "010")]


class TestUnsupported:
    @pytest.mark.parametrize(
        ("frame", "column", "kind"),
        [
            (pl.DataFrame({"x": [1], "d": [dt.timedelta(seconds=1)]}), "d", "Duration"),
            (pl.DataFrame({"l": [[1, 2]]}), "l", "List"),
            (pl.DataFrame({"s": [{"a": 1}]}), "s", "Struct"),
            (pl.DataFrame({"b": [b"ab"]}), "b", "Binary"),
        ],
    )
    def test_a_type_csv_cannot_hold_names_the_asset_and_column(self, tmp_path: Path, frame, column, kind):
        app = _app(tmp_path, [frame])

        with pytest.raises(TypeError, match=rf"asset 'a''s column '{column}' is {kind}.*which CSV can't store"):
            _run(app)
        assert not list(tmp_path.rglob("*.csv")) and not list(tmp_path.rglob("_schema.json"))

    def test_a_frame_with_no_columns_is_refused_when_written(self, tmp_path: Path):
        app = _app(tmp_path, [pl.DataFrame()])

        with pytest.raises(ValueError, match="asset 'a' has no columns"):
            _run(app)


class TestTheRecord:
    def test_it_sits_beside_the_files(self, tmp_path: Path):
        _roundtrip(tmp_path, pl.DataFrame({"a": [1]}))

        assert sorted(path.name for path in (tmp_path / "a").iterdir()) == sorted([_lone(tmp_path).name, "_schema.json"])

    def test_it_goes_wherever_the_directory_goes(self, tmp_path: Path):
        import shutil
        _roundtrip(tmp_path / "here", pl.DataFrame({"d": [dt.date(2026, 1, 1)], "code": ["007"]}))
        shutil.copytree(tmp_path / "here" / "a", tmp_path / "there" / "a")

        back = _load(_app(tmp_path / "there", []))

        assert back.schema == pl.Schema({"d": pl.Date, "code": pl.String}) and back["code"].to_list() == ["007"]

    def test_an_overwrite_keeps_it(self, tmp_path: Path):
        frame = pl.DataFrame({"d": [dt.date(2026, 1, 1)]})
        _run(_app(tmp_path, [frame]))
        app = _app(tmp_path, [frame])
        _run(app)

        assert _load(app).schema["d"] == pl.Date and (tmp_path / "a" / "_schema.json").is_file()

    def test_a_partitioned_overwrite_keeps_it(self, tmp_path: Path):
        partition = PartitionSettings(columns={"day": pl.Date})
        first = [Output(pl.DataFrame({"x": [1]}), partition={"day": dt.date(2026, 1, 1)}), Output(pl.DataFrame({"x": [2]}), partition={"day": dt.date(2026, 1, 2)})]
        _run(_app(tmp_path, first, partition=partition))
        app = _app(tmp_path, [Output(pl.DataFrame({"x": [3]}), partition={"day": dt.date(2026, 1, 2)})], partition=partition)
        _run(app)

        back = _load(app).sort("day")
        assert back.schema["day"] == pl.Date and back["x"].to_list() == [1, 3]
        assert (tmp_path / "a" / "_schema.json").is_file()

    def test_a_csv_with_no_record_is_inferred_as_it_always_was(self, tmp_path: Path):
        app = _app(tmp_path, [pl.DataFrame({"d": [dt.date(2026, 1, 1)], "code": ["007"]})])
        _run(app)
        (tmp_path / "a" / "_schema.json").unlink()

        back = _load(app)
        assert back.schema == pl.Schema({"d": pl.String, "code": pl.Int64})

    def test_a_hand_placed_csv_loads(self, tmp_path: Path):
        app = _app(tmp_path, [])
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "part-1.csv").write_text("x,y\n1,a\n2,b\n")

        assert _load(app).rows() == [(1, "a"), (2, "b")]

    def test_a_column_changed_by_hand_is_refused_on_load(self, tmp_path: Path):
        app = _app(tmp_path, [pl.DataFrame({"a": [1], "b": [2]})])
        _run(app)
        _lone(tmp_path).write_text("a,c\n1,2\n")

        with pytest.raises(ValueError, match=r"has the columns \['a', 'c'\], but \['a', 'b'\] were recorded"):
            _load(app)

    def test_a_value_that_is_not_its_type_fails_the_read(self, tmp_path: Path):
        app = _app(tmp_path, [pl.DataFrame({"a": [1, 2]})])
        _run(app)
        _lone(tmp_path).write_text("a\n1\nx\n")

        with pytest.raises(pl.exceptions.PolarsError):
            _load(app)

    def test_a_boolean_that_is_not_true_or_false_fails_the_read(self, tmp_path: Path):
        app = _app(tmp_path, [pl.DataFrame({"a": [True, False]})])
        _run(app)
        _lone(tmp_path).write_text("a\ntrue\nyes\n")

        with pytest.raises(pl.exceptions.PolarsError):
            _load(app)

    def test_clearing_the_asset_removes_it(self, tmp_path: Path):
        app = _app(tmp_path, [pl.DataFrame({"a": [1]})])
        _run(app)
        config = app.assets["a"].settings.storage

        get_storage_concrete(config).clear(app.assets["a"])

        assert not list(tmp_path.rglob("_schema.json"))
        _run(_app(tmp_path, [pl.DataFrame({"a": ["new"]})], mode=Mode.APPEND))  # no stale record to clash with

    def test_clearing_one_partition_keeps_it(self, tmp_path: Path):
        partition = PartitionSettings(columns={"year": pl.Int64})
        frames = [Output(pl.DataFrame({"a": [1]}), partition={"year": 2024}), Output(pl.DataFrame({"a": [2]}), partition={"year": 2025})]
        app = _app(tmp_path, frames, partition=partition)
        _run(app)
        config = app.assets["a"].settings.storage
        settings = app.assets["a"].settings.partition

        from repipe.asset import PartitionKey
        get_storage_concrete(config).clear(app.assets["a"], PartitionKey.from_mapping(settings, {"year": 2024}))

        assert (tmp_path / "a" / "_schema.json").is_file()
        assert _load(app)["a"].to_list() == [2]

    def test_a_failed_run_leaves_the_data_and_its_record(self, tmp_path: Path):
        _run(_app(tmp_path, [pl.DataFrame({"a": [1]})]))

        @asset(storage=CsvStorageConfig())
        def a(context: AssetContext) -> Iterator[pl.DataFrame]:
            yield pl.DataFrame({"a": ["x"]})
            raise RuntimeError("boom")

        app = Repipe(assets=[a], base_storage_path=tmp_path)
        with pytest.raises(RuntimeError):
            _run(app)

        # The first chunk was already committed (writes are durable as they are yielded), so the record follows it
        assert _load(app).schema == pl.Schema({"a": pl.String})


def _forget_the_record(tmp_path: Path) -> None:
    """ What an asset written before types were recorded looks like. """
    (tmp_path / "a" / "_schema.json").unlink()


class TestDataWrittenBeforeTypesWereRecorded:
    def test_text_that_would_have_to_become_numbers_is_refused_before_anything_is_committed(self, tmp_path: Path):
        _run(_app(tmp_path, [pl.DataFrame({"x": ["abc"]})], mode=Mode.APPEND_THEN_MERGE, primary_key="x"))
        _forget_the_record(tmp_path)
        app = _app(tmp_path, [pl.DataFrame({"x": [1]})], mode=Mode.APPEND_THEN_MERGE, primary_key="x")

        with pytest.raises(ValueError, match=r"written before its types were recorded.*repipe clear a"):
            _run(app)

        assert not (tmp_path / "a" / "_schema.json").exists()
        assert len(list((tmp_path / "a").glob("*.csv"))) == 1  # no pending part, no half-done merge
        assert _load(app)["x"].to_list() == ["abc"]

    def test_numbers_would_not_be_quietly_turned_into_text(self, tmp_path: Path):
        _run(_app(tmp_path, [pl.DataFrame({"x": [1.5]})], mode=Mode.APPEND))
        _forget_the_record(tmp_path)
        app = _app(tmp_path, [pl.DataFrame({"x": ["007"]})], mode=Mode.APPEND)

        with pytest.raises(ValueError, match=r"\['x'\] would become text"):
            _run(app)

    def test_data_that_fits_is_adopted_and_gets_its_types(self, tmp_path: Path):
        _run(_app(tmp_path, [pl.DataFrame({"d": [dt.date(2026, 1, 1)], "n": [1]})], mode=Mode.APPEND))
        _forget_the_record(tmp_path)
        assert _load(_app(tmp_path, [], mode=Mode.APPEND)).schema["d"] == pl.String  # as it was read before

        app = _app(tmp_path, [pl.DataFrame({"d": [dt.date(2026, 1, 2)], "n": [2]})], mode=Mode.APPEND)
        _run(app)

        back = _load(app).sort("n")
        assert back.schema == pl.Schema({"d": pl.Date, "n": pl.Int64})
        assert back.rows() == [(dt.date(2026, 1, 1), 1), (dt.date(2026, 1, 2), 2)]

    def test_a_merge_onto_data_that_fits_keeps_its_types(self, tmp_path: Path):
        _run(_app(tmp_path, [pl.DataFrame({"id": [1], "d": [dt.date(2026, 1, 1)]})], mode=Mode.APPEND_THEN_MERGE, primary_key="id"))
        _forget_the_record(tmp_path)
        app = _app(tmp_path, [pl.DataFrame({"id": [2], "d": [dt.date(2026, 1, 2)]})], mode=Mode.APPEND_THEN_MERGE, primary_key="id")
        _run(app)

        back = _load(app).sort("id")
        assert back.schema == pl.Schema({"id": pl.Int64, "d": pl.Date}) and back["id"].to_list() == [1, 2]

    def test_an_overwrite_of_everything_needs_no_check(self, tmp_path: Path):
        _run(_app(tmp_path, [pl.DataFrame({"x": ["abc"]})]))
        _forget_the_record(tmp_path)
        app = _app(tmp_path, [pl.DataFrame({"x": [1]})])
        _run(app)

        assert _load(app).schema == pl.Schema({"x": pl.Int64})


class TestAnAssetWithNothingInIt:
    def test_a_downstream_asset_can_load_an_upstream_that_was_overwritten_with_nothing(self, tmp_path: Path):
        runs = iter([[pl.DataFrame({"a": [1]})], []])

        @asset(storage=CsvStorageConfig())
        def up(context: AssetContext) -> Iterator[pl.DataFrame]:
            yield from next(runs)

        loaded = []

        @asset(dependencies=["up"])
        def down(context: AssetContext, up: pl.LazyFrame) -> pl.DataFrame:
            loaded.append(up)  # not collected: it is the loading that used to raise
            return pl.DataFrame({"n": [1]})

        app = Repipe(assets=[up, down], base_storage_path=tmp_path)
        _run(app, "up")
        _run(app, "up")
        _run(app, "down")

        assert len(loaded) == 1
        assert not (tmp_path / "up" / "_schema.json").exists()  # nothing left for it to describe

    def test_a_recorded_asset_whose_files_are_gone_loads_empty_and_typed(self, tmp_path: Path):
        app = _app(tmp_path, [pl.DataFrame({"d": [dt.date(2026, 1, 1)], "n": [1]})])
        _run(app)
        _lone(tmp_path).unlink()

        back = _load(app)
        assert back.height == 0 and back.schema == pl.Schema({"d": pl.Date, "n": pl.Int64})


class TestNullColumns:
    """ `pl.DataFrame({"a": [None]})` has the type `Null`, which says nothing about what the column would hold. """

    def test_an_untyped_all_null_chunk_after_an_int_chunk_stays_int(self, tmp_path: Path):
        app = _app(tmp_path, [pl.DataFrame({"a": [1, 2]}), pl.DataFrame({"a": [None, None]})], mode=Mode.APPEND)
        _run(app)

        back = _load(app)
        assert back.schema["a"] == pl.Int64 and back["a"].to_list() == [1, 2, None, None]

    def test_and_before_one(self, tmp_path: Path):
        app = _app(tmp_path, [pl.DataFrame({"a": [None, None]}), pl.DataFrame({"a": [1, 2]})], mode=Mode.APPEND)
        _run(app)

        back = _load(app)
        assert back.schema["a"] == pl.Int64 and back["a"].to_list() == [None, None, 1, 2]

    def test_across_runs_in_both_orders(self, tmp_path: Path):
        _run(_app(tmp_path / "typed_first", [pl.DataFrame({"a": [1]})], mode=Mode.APPEND))
        typed_first = _app(tmp_path / "typed_first", [pl.DataFrame({"a": [None]})], mode=Mode.APPEND)
        _run(typed_first)
        _run(_app(tmp_path / "null_first", [pl.DataFrame({"a": [None]})], mode=Mode.APPEND))
        null_first = _app(tmp_path / "null_first", [pl.DataFrame({"a": [1]})], mode=Mode.APPEND)
        _run(null_first)

        assert _load(typed_first).schema["a"] == pl.Int64 and _load(typed_first)["a"].to_list() == [1, None]
        assert _load(null_first).schema["a"] == pl.Int64 and _load(null_first)["a"].to_list() == [None, 1]

    def test_a_null_column_next_to_others_still_has_to_match_them(self, tmp_path: Path):
        app = _app(tmp_path, [pl.DataFrame({"a": [1], "b": ["x"]}), pl.DataFrame({"a": [None], "b": [1.5]})], mode=Mode.APPEND)

        with pytest.raises(ValueError, match="'b' was String, now Float64"):
            _run(app)


class TestACommitThatFails:
    def test_an_overwrite_that_fails_leaves_the_old_data_under_its_own_record(self, tmp_path: Path, monkeypatch):
        _run(_app(tmp_path, [pl.DataFrame({"x": [1.5]})]))
        app = _app(tmp_path, [pl.DataFrame({"x": ["abc"]})])

        def fail(self, asset, directory, run_id):
            raise OSError("disk full")

        monkeypatch.setattr(CsvPolars, "promote", fail)
        with pytest.raises(OSError):
            _run(app)
        monkeypatch.undo()

        back = _load(app)
        assert back.schema == pl.Schema({"x": pl.Float64}) and back["x"].to_list() == [1.5]


class TestPartitionedOverwrites:
    def test_rewriting_every_partition_still_cannot_change_the_schema(self, tmp_path: Path):
        # Documented: each chunk commits on its own, so the first still sees the others as kept. `repipe clear` first.
        partition = PartitionSettings(columns={"year": pl.Int64})
        _run(_app(tmp_path, [
            Output(pl.DataFrame({"a": [1]}), partition={"year": 2024}), Output(pl.DataFrame({"a": [2]}), partition={"year": 2025}),
        ], partition=partition))
        app = _app(tmp_path, [
            Output(pl.DataFrame({"a": ["x"]}), partition={"year": 2024}), Output(pl.DataFrame({"a": ["y"]}), partition={"year": 2025}),
        ], partition=partition)

        with pytest.raises(ValueError, match=r"different schema.*repipe clear a"):
            _run(app)


class TestColumnNames:
    @pytest.mark.parametrize("name", ['q"', '"', 'a"b', "\ufeffa"])
    def test_a_name_a_header_cannot_hold_is_refused_before_anything_is_written(self, tmp_path: Path, name):
        app = _app(tmp_path, [pl.DataFrame({name: [1]})])

        with pytest.raises(ValueError, match="quote or a byte-order mark in its name"):
            _run(app)
        assert not list(tmp_path.rglob("*.csv")) and not list(tmp_path.rglob("_schema.json"))


class TestZonesWithSeconds:
    def test_the_instant_is_exact_and_the_zone_comes_back(self, tmp_path: Path):
        frame = pl.DataFrame({
            "a": pl.Series([dt.datetime(1900, 1, 1, 12)]).dt.replace_time_zone("Asia/Kolkata"),
            "b": pl.Series([dt.datetime(1970, 1, 1, 12)]).dt.replace_time_zone("Africa/Monrovia"),
        })

        back = _roundtrip(tmp_path, frame)

        assert back.schema == frame.schema
        assert back["a"][0] == frame["a"][0] and back["a"][0].utcoffset() == frame["a"][0].utcoffset()  # not 12:00:10
        assert back["b"][0] == frame["b"][0]

    def test_the_file_holds_it_in_utc(self, tmp_path: Path):
        _roundtrip(tmp_path, pl.DataFrame({"a": pl.Series([dt.datetime(2026, 7, 1, 12)]).dt.replace_time_zone("Europe/London")}))

        assert _lone(tmp_path).read_text() == "a\n2026-07-01T11:00:00.000000+0000\n"
