import asyncio
from typing import cast

import typer

from repipe.asset.asset import Asset
from repipe.cli.selector import NodeKind, ResourceQuery, build_graph
from repipe.project import load_app
from repipe.runtime import Runtime
from repipe.runtime import build as build_assets
from repipe.runtime import run as run_assets
from repipe.runtime import test as run_tests
from repipe.storage import consolidate as consolidate_asset

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
):
    """ Compacts (and dedupes, where configured) every selected asset's part files - the
    same logic `Mode.APPEND_THEN_MERGE` runs automatically, available here on demand. """
    runtime = _build_runtime(selectors)
    for node in runtime.selected_nodes():
        if node.kind == NodeKind.ASSET:
            asset = cast(Asset, node.obj)
            config = asset.settings.storage
            if max_rows_per_file is not None:
                config = config.model_copy(update={"max_rows_per_file": max_rows_per_file})
            consolidate_asset(config, asset)


if __name__ == "__main__":
    app()
