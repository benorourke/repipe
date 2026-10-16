import asyncio
import shutil
from typing import Iterable, Iterator, cast

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


def _targets(assets: Iterable[Asset], filters: list[str]) -> list[tuple[Asset, list[PartitionKey | None]]]:
    """ Each asset with its partitions matching every `column=value` filter, or `[None]` (the whole
    asset) without filters. Checks every asset before returning, so nothing is half done. """
    if any("=" not in f for f in filters):
        raise typer.BadParameter(f"--partition takes column=value, got {filters!r}")
    wanted = dict(f.split("=", 1) for f in filters)
    targets: list[tuple[Asset, list[PartitionKey | None]]] = []
    for asset in assets:
        if not wanted:
            targets.append((asset, [None]))
            continue
        settings = asset.settings.partition
        if settings is None:
            raise typer.BadParameter(f"asset {asset.name!r} isn't partitioned")
        if unknown := set(wanted) - set(settings.columns):
            raise typer.BadParameter(f"asset {asset.name!r} has no partition columns {sorted(unknown)!r}")
        targets.append((asset, [
            key for key in get_storage_concrete(asset.settings.storage).existing_partitions(asset)
            if all(format_value(key[column]) == value for column, value in wanted.items())
        ]))
    return targets


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
    for asset, keys in _targets(_selected_assets(_build_runtime(selectors)), partition):
        config = asset.settings.storage
        if max_rows_per_file is not None:
            config = config.model_copy(update={"max_rows_per_file": max_rows_per_file})
        for key in keys:
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
        staging: bool = typer.Option(False, "--staging", help="Only delete staging left behind by crashed runs. A run in progress will fail."),
):
    """ Deletes the selected assets' data (or some of their partitions). """
    runtime = _build_runtime(selectors)
    if staging:
        for base_storage_path in {asset.base_storage_path for asset in _selected_assets(runtime)}:
            shutil.rmtree(staging_root(base_storage_path), ignore_errors=True)
        return
    if not selectors:
        raise typer.BadParameter("select the assets to clear - this deletes their data")
    for asset, keys in _targets(_selected_assets(runtime), partition):
        for key in keys:
            get_storage_concrete(asset.settings.storage).clear(asset, key)


if __name__ == "__main__":
    app()
