from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from typing import Any, Iterator

import polars as pl
import pytest

from repipe.asset import AssetContext, Output, PartitionSettings, asset
from repipe.cli.selector import build_graph
from repipe.project import Repipe
from repipe.runtime import materialize
from repipe.storage import ArrowStorageConfig, Mode, ParquetStorageConfig, get_storage_concrete

DAYS = PartitionSettings(columns={"day": pl.Date})
FORMATS = [ParquetStorageConfig, ArrowStorageConfig]


def _app(
    tmp_path: Path, config: type, runs: list[Any], *, partition: PartitionSettings | None = DAYS, primary_key=None, **storage: Any,
) -> Repipe:
    """ One asset, `a`, which yields `runs` (each an `Output`, or a bare frame when it is not partitioned). """
    @asset(storage=config(**storage), partition=partition, primary_key=primary_key)
    def a(context: AssetContext) -> Iterator[Any]:
        yield from runs

    return Repipe(assets=[a], base_storage_path=tmp_path)


def _run(app: Repipe) -> None:
    asyncio.run(materialize(app.assets["a"], build_graph(app.assets)))


def _load(app: Repipe) -> pl.DataFrame:
    config = app.assets["a"].settings.storage
    return get_storage_concrete(config).load(config, app.assets["a"]).collect()


def _day(number: int, **columns: Any) -> Output:
    day = dt.date(2026, 11, number)
    return Output(pl.DataFrame({"day": [day], **columns}), partition={"day": day})


@pytest.mark.parametrize("config", FORMATS)
class TestFilesWithDifferentColumns:
    def test_a_later_partition_with_an_extra_column_reads_with_every_column(self, tmp_path, config):
        app = _app(tmp_path, config, [_day(10, x=[1]), _day(11, x=[2], y=[20])])
        _run(app)

        frame = _load(app).sort("day")
        assert frame.columns == ["day", "x", "y"]
        assert frame["y"].to_list() == [None, 20]

    def test_an_earlier_partition_with_an_extra_column_reads_with_every_column(self, tmp_path, config):
        app = _app(tmp_path, config, [_day(10, x=[1], y=[10]), _day(11, x=[2])])
        _run(app)

        frame = _load(app).sort("day")
        assert frame.columns == ["day", "x", "y"]
        assert frame["y"].to_list() == [10, None]

    def test_partition_columns_keep_their_declared_type(self, tmp_path, config):
        app = _app(tmp_path, config, [_day(10, x=[1]), _day(11, x=[2], y=[20])])
        _run(app)

        assert _load(app).schema["day"] == pl.Date

    def test_columns_of_different_types_in_different_files_are_widened(self, tmp_path, config):
        runs = [
            Output(pl.DataFrame({"day": [dt.date(2026, 11, 10)], "x": pl.Series([1], dtype=pl.Int32)}), partition={"day": dt.date(2026, 11, 10)}),
            Output(pl.DataFrame({"day": [dt.date(2026, 11, 11)], "x": pl.Series([2], dtype=pl.Int64)}), partition={"day": dt.date(2026, 11, 11)}),
        ]
        app = _app(tmp_path, config, runs)
        _run(app)

        frame = _load(app).sort("day")
        assert frame.schema["x"] == pl.Int64
        assert frame["x"].to_list() == [1, 2]

    def test_a_filter_on_the_partition_still_returns_only_that_partition(self, tmp_path, config):
        app = _app(tmp_path, config, [_day(10, x=[1]), _day(11, x=[2], y=[20]), _day(12, x=[3], y=[30])])
        _run(app)
        asset_ = app.assets["a"]
        storage = get_storage_concrete(asset_.settings.storage)

        frame = storage.load(asset_.settings.storage, asset_).filter(pl.col.day == dt.date(2026, 11, 11)).collect()
        assert frame.to_dicts() == [{"day": dt.date(2026, 11, 11), "x": 2, "y": 20}]

    def test_files_with_the_same_columns_in_another_order_read_as_before(self, tmp_path, config):
        runs = [
            Output(pl.DataFrame({"day": [dt.date(2026, 11, 10)], "x": [1], "y": [10]}), partition={"day": dt.date(2026, 11, 10)}),
            Output(pl.DataFrame({"y": [20], "x": [2], "day": [dt.date(2026, 11, 11)]}), partition={"day": dt.date(2026, 11, 11)}),
        ]
        app = _app(tmp_path, config, runs)
        _run(app)

        frame = _load(app).sort("day")
        assert set(frame.columns) == {"day", "x", "y"}
        assert frame["x"].to_list() == [1, 2] and frame["y"].to_list() == [10, 20]

    def test_a_single_file_reads_as_before(self, tmp_path, config):
        app = _app(tmp_path, config, [_day(10, x=[1])])
        _run(app)

        assert _load(app).to_dicts() == [{"day": dt.date(2026, 11, 10), "x": 1}]

    def test_an_unpartitioned_asset_with_parts_of_different_columns_reads_with_every_column(self, tmp_path, config):
        app = _app(tmp_path, config, [pl.DataFrame({"x": [1]}), pl.DataFrame({"x": [2], "y": [20]})], partition=None, mode=Mode.APPEND)
        _run(app)

        frame = _load(app).sort("x")
        assert frame.columns == ["x", "y"]
        assert frame["y"].to_list() == [None, 20]

    def test_nothing_stored_fails_as_it_did(self, tmp_path, config):
        app = _app(tmp_path, config, [_day(10, x=[1])])  # never run

        with pytest.raises(Exception):
            _load(app)


@pytest.mark.parametrize("config", FORMATS)
class TestAnAssetWhoseOutputGainsAColumn:
    def test_a_merging_run_that_adds_a_column_keeps_the_old_rows_and_the_new_column(self, tmp_path, config):
        monday = dt.date(2026, 11, 10)
        first = Output(pl.DataFrame({"day": [monday, monday], "id": [1, 2], "x": [10, 20]}), partition={"day": monday})
        second = Output(pl.DataFrame({"day": [monday, monday], "id": [2, 3], "x": [21, 30], "y": [210, 300]}), partition={"day": monday})
        _run(_app(tmp_path, config, [first], mode=Mode.APPEND_THEN_MERGE, primary_key=["id"]))

        merging = _app(tmp_path, config, [second], mode=Mode.APPEND_THEN_MERGE, primary_key=["id"])
        _run(merging)

        frame = _load(merging).sort("id")
        assert frame.columns == ["day", "id", "x", "y"]
        assert frame["id"].to_list() == [1, 2, 3]
        assert frame["x"].to_list() == [10, 21, 30]  # the later run replaced id 2
        assert frame["y"].to_list() == [None, 210, 300]

    def test_the_partition_is_left_as_files_that_read_back_after_the_merge(self, tmp_path, config):
        monday = dt.date(2026, 11, 10)
        first = Output(pl.DataFrame({"day": [monday], "id": [1], "x": [10]}), partition={"day": monday})
        second = Output(pl.DataFrame({"day": [monday], "id": [2], "x": [20], "y": [200]}), partition={"day": monday})
        _run(_app(tmp_path, config, [first], mode=Mode.APPEND_THEN_MERGE, primary_key=["id"]))
        merging = _app(tmp_path, config, [second], mode=Mode.APPEND_THEN_MERGE, primary_key=["id"])
        _run(merging)

        storage = get_storage_concrete(merging.assets["a"].settings.storage)
        parts = storage.existing_parts(merging.assets["a"])
        assert len(parts) == 1 and not list((tmp_path / "a").rglob("*.tmp"))  # consolidated into one, nothing orphaned

    def test_consolidating_parts_with_different_columns_keeps_every_column(self, tmp_path, config):
        app = _app(tmp_path, config, [pl.DataFrame({"x": [1]}), pl.DataFrame({"x": [2], "y": [20]})], partition=None, mode=Mode.APPEND)
        _run(app)
        asset_ = app.assets["a"]
        storage = get_storage_concrete(asset_.settings.storage)

        storage.consolidate(asset_.settings.storage, asset_)

        assert len(storage.existing_parts(asset_)) == 1
        assert _load(app).sort("x").to_dicts() == [{"x": 1, "y": None}, {"x": 2, "y": 20}]

    def test_load_previous_reads_parts_with_different_columns(self, tmp_path, config):
        app = _app(tmp_path, config, [pl.DataFrame({"x": [1]}), pl.DataFrame({"x": [2], "y": [20]})], partition=None, mode=Mode.APPEND)
        _run(app)
        asset_ = app.assets["a"]

        previous = get_storage_concrete(asset_.settings.storage).load_previous(asset_.settings.storage, asset_)

        assert previous is not None and previous.collect().sort("x").columns == ["x", "y"]
