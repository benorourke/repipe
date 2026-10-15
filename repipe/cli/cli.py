import asyncio
import shutil
from typing import Iterator, cast

import typer

from repipe.asset.asset import Asset
from repipe.asset.partition import PartitionKey, format_value
from repipe.cli.selector import NodeKind, ResourceQuery, build_graph
from repipe.project import load_app
from repipe.runtime import Runtime
from repipe.runtime import build as build_assets
from repipe.runtime import run as run_assets
from repipe.runtime import test as run_tests
from repipe.storage import consolidate as consolidate_asset, get_storage_concrete
from repipe.storage.storage import staging_root

app = typer.Typer()

_SELECTOR_HELP = (
    "dbt-style selectors, e.g. `asset_1 +asset_2 asset_3+ label:nightly` - see Selector.parse "
    "for the full syntax. Omit to select everything."
)


def _build_runtime(selectors: list[str] | None) -> Runtime:
    query = ResourceQuery.parse(selectors or [])
    root, module, loaded_app = load_app()
    graph = build_graph(loaded_app.assets)
    return Runtime(graph=graph, selection=query)


_PARTITION_HELP = "Only partitions where column=value, e.g. `date=2026-10-12`. Repeatable."


def _selected_assets(runtime: Runtime) -> Iterator[Asset]:
    for node in runtime.selected_nodes():
        if node.kind == NodeKind.ASSET:
            yield cast(Asset, node.obj)


def _matching_partitions(asset: Asset, filters: list[str]) -> list[PartitionKey]:
    """ The asset's existing partitions matching every `column=value` filter. """
    if any("=" not in f for f in filters):
        raise typer.BadParameter(f"--partition takes column=value, got {filters!r}")
    wanted = dict(f.split("=", 1) for f in filters)
    if asset.settings.partition is None:
        raise typer.BadParameter(f"asset {asset.name!r} isn't partitioned")
    return [
        key for key in get_storage_concrete(asset.settings.storage).existing_partitions(asset)
        if all(column in key and format_value(key[column]) == value for column, value in wanted.items())
    ]


@app.callback()
def main():
    """ repipe - an ML + polars native orchestrator """


@app.command()
def run(
        selectors: list[str] = typer.Argument(None, help=_SELECTOR_HELP),
):
    """ Materializes the selected assets, in dependency order. Runs no tests. """
    asyncio.run(run_assets(_build_runtime(selectors)))


@app.command()
def build(
        selectors: list[str] = typer.Argument(None, help=_SELECTOR_HELP),
):
    """ Materializes the selected assets, in dependency order, running each asset's own
    tests immediately after it materializes. """
    asyncio.run(build_assets(_build_runtime(selectors)))


@app.command(name="test")
def test(
        selectors: list[str] = typer.Argument(
            None,
            help="dbt-style test selectors, same syntax as `run`. A reused test's bare name "
                 "must be qualified as `asset_name::test_name` if it's attached to more than "
                 "one asset. Omit to run every test.",
        ),
):
    """ Runs the selected data quality tests directly, independent of materialization. """
    asyncio.run(run_tests(_build_runtime(selectors)))


@app.command(name="consolidate")
def consolidate(
        selectors: list[str] = typer.Argument(None, help=_SELECTOR_HELP),
        max_rows_per_file: int | None = typer.Option(
            None, "--max-rows-per-file", help="Override each asset's configured max_rows_per_file.",
        ),
        partition: list[str] = typer.Option([], "--partition", help=_PARTITION_HELP),
):
    """ Compacts (and dedupes, where configured) every selected asset's part files - the
    same logic `Mode.APPEND_THEN_MERGE` runs automatically, available here on demand. """
    for asset in _selected_assets(_build_runtime(selectors)):
        config = asset.settings.storage
        if max_rows_per_file is not None:
            config = config.model_copy(update={"max_rows_per_file": max_rows_per_file})
        if not partition:
            consolidate_asset(config, asset)
        for key in _matching_partitions(asset, partition) if partition else []:
            consolidate_asset(config, asset, key)


@app.command(name="partitions")
def partitions(
        selectors: list[str] = typer.Argument(None, help=_SELECTOR_HELP),
):
    """ Lists each selected partitioned asset's partitions. """
    for asset in _selected_assets(_build_runtime(selectors)):
        if asset.settings.partition is None:
            continue
        storage = get_storage_concrete(asset.settings.storage)
        typer.echo(asset.name)
        for key in storage.existing_partitions(asset):
            typer.echo(f"  {key.relative_path()}  ({len(storage.existing_parts(asset, key))} files)")


@app.command(name="clear")
def clear(
        selectors: list[str] = typer.Argument(None, help=_SELECTOR_HELP),
        partition: list[str] = typer.Option([], "--partition", help=_PARTITION_HELP),
        staging: bool = typer.Option(False, "--staging", help="Only delete staging left behind by crashed runs."),
):
    """ Deletes the selected assets' data (or some of their partitions). """
    runtime = _build_runtime(selectors)
    if staging:
        for base_storage_path in {asset.base_storage_path for asset in _selected_assets(runtime)}:
            shutil.rmtree(staging_root(base_storage_path), ignore_errors=True)
        return
    if not selectors:
        raise typer.BadParameter("select the assets to clear - this deletes their data")
    for asset in _selected_assets(runtime):
        storage = get_storage_concrete(asset.settings.storage)
        if not partition:
            storage.clear(asset)
        for key in _matching_partitions(asset, partition) if partition else []:
            storage.clear(asset, key)


if __name__ == "__main__":
    app()
