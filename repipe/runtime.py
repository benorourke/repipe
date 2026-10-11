import inspect
import itertools
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterator, cast

from pydantic import BaseModel

from repipe.asset.asset import Asset, AssetContext, AssetReturnType
from repipe.cli.selector import GraphNode, NodeKind, ResourceQuery, topologically_sorted
from repipe.storage import consolidate, get_storage_concrete
from repipe.storage.storage import Mode
from repipe.testing.data_quality import RegisteredDataQualityTest

# Disambiguates two `new_run_id()` calls landing in the same millisecond
_run_id_counter = itertools.count(1)


def new_run_id() -> str:
    now = datetime.now(timezone.utc)
    return f"{now:%Y%m%d-%H%M%S}-{now.microsecond // 1000:03d}-{next(_run_id_counter):04d}"


class Runtime(BaseModel):
    """ Everything needed to execute a single `repipe` invocation against an already-built
    asset/test graph: the graph itself, plus what this invocation selected. """
    model_config = {"arbitrary_types_allowed": True}

    graph: dict[str, GraphNode]
    selection: ResourceQuery

    def selected_nodes(self) -> Iterator[GraphNode]:
        """ The nodes matched by `selection` - assets and tests alike - in the combined
        graph's dependency order. """
        return topologically_sorted(self.graph, self.selection.resolve(self.graph))

    def tests_for_asset(self, asset_name: str) -> dict[str, RegisteredDataQualityTest]:
        """ The tests directly attached to this one asset, regardless of `selection` - used by
        `build` to run a materialized asset's own tests right after it runs. """
        return {
            node.id: cast(RegisteredDataQualityTest, node.obj)
            for node in self.graph.values()
            if node.kind == NodeKind.TEST and asset_name in node.upstream
        }


async def run(runtime: Runtime) -> None:
    """ Materializes the selected assets, in dependency order. Runs no tests - repipe `run`. """
    results: dict[str, AssetReturnType] = {}
    for node in runtime.selected_nodes():
        if node.kind == NodeKind.ASSET:
            await materialize(cast(Asset, node.obj), runtime.graph, results)


async def build(runtime: Runtime) -> None:
    """ Materializes the selected assets, in dependency order, running each asset's own
    attached tests immediately after it materializes - repipe `build`. """
    results: dict[str, AssetReturnType] = {}
    for node in runtime.selected_nodes():
        if node.kind != NodeKind.ASSET:
            continue
        await materialize(cast(Asset, node.obj), runtime.graph, results)
        for test_id, test in runtime.tests_for_asset(node.id).items():
            run_test(test_id, test)


async def test(runtime: Runtime) -> None:
    """ Runs the selected data quality tests directly, independent of materialization, in the
    combined graph's dependency order (a test never runs before the asset(s) it's attached
    to) - repipe `test`. """
    for node in runtime.selected_nodes():
        if node.kind == NodeKind.TEST:
            run_test(node.id, cast(RegisteredDataQualityTest, node.obj))


async def materialize(
        asset: Asset,
        graph: dict[str, GraphNode],
        results: dict[str, AssetReturnType],
) -> None:
    """ Persists the asset's result through its configured storage. """
    context = AssetContext(manifest=asset, dependencies=asset.settings.dependency_names)

    storage_config = asset.settings.storage
    storage = get_storage_concrete(storage_config)

    kwargs: dict[str, Any] = {}
    for param_name in inspect.signature(asset.func).parameters:
        if param_name == "context":
            kwargs[param_name] = context
        elif param_name == asset.name:
            # Its own previous output (None on the first run) - snapshotted here, before any write.
            kwargs[param_name] = storage.load_previous(storage_config, asset)
        else:
            kwargs[param_name] = _resolve_dependency(param_name, graph, results)
    func = asset.func

    # Snapshot stale parts for deletion once generating is complete
    stale_parts = storage.existing_parts(asset) if storage_config.mode == Mode.OVERWRITE else set()

    run_id = new_run_id()

    if inspect.isasyncgenfunction(func):
        chunks: AsyncIterator[Any] = func(**kwargs)
        part_number = 1
        async for chunk in chunks:
            storage.save(storage_config, asset, chunk, run_id, part_number)
            part_number += 1
        results[asset.name] = storage.load(storage_config, asset)
    elif inspect.isgeneratorfunction(func):
        part_number = 1
        for chunk in func(**kwargs):
            storage.save(storage_config, asset, chunk, run_id, part_number)
            part_number += 1
        results[asset.name] = storage.load(storage_config, asset)
    else:
        result = await func(**kwargs) if inspect.iscoroutinefunction(func) else func(**kwargs)
        storage.save(storage_config, asset, result, run_id, 1)
        results[asset.name] = result

    for stale_part in stale_parts:
        stale_part.unlink(missing_ok=True)

    if storage_config.mode == Mode.APPEND_THEN_MERGE:
        consolidate(storage_config, asset)

    print(f"materialized {asset.name}")


def _resolve_dependency(
        name: str,
        graph: dict[str, GraphNode],
        results: dict[str, AssetReturnType],
) -> AssetReturnType:
    if name in results:
        return results[name]

    node = graph.get(name)
    if node is None or node.kind != NodeKind.ASSET:
        raise ValueError(f"{name!r} is not a known asset - check its @asset dependencies/parameters")

    upstream = cast(Asset, node.obj)
    return get_storage_concrete(upstream.settings.storage).load(upstream.settings.storage, upstream)


def run_test(test_id: str, test: RegisteredDataQualityTest) -> None:
    print(test_id)  # TODO: actually invoke test.func and handle fail_type
