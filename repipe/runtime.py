import inspect
from typing import Any, Iterator, cast

from pydantic import BaseModel

from repipe.asset.asset import Asset, AssetContext, AssetReturnType
from repipe.cli.selector import GraphNode, NodeKind, ResourceQuery, topologically_sorted
from repipe.storage import get_storage_concrete
from repipe.testing.data_quality import RegisteredDataQualityTest


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


def run(runtime: Runtime) -> None:
    """ Materializes the selected assets, in dependency order. Runs no tests - repipe `run`. """
    results: dict[str, AssetReturnType] = {}
    for node in runtime.selected_nodes():
        if node.kind == NodeKind.ASSET:
            materialize(cast(Asset, node.obj), runtime.graph, results)


def build(runtime: Runtime) -> None:
    """ Materializes the selected assets, in dependency order, running each asset's own
    attached tests immediately after it materializes - repipe `build`. """
    results: dict[str, AssetReturnType] = {}
    for node in runtime.selected_nodes():
        if node.kind != NodeKind.ASSET:
            continue
        materialize(cast(Asset, node.obj), runtime.graph, results)
        for test_id, test in runtime.tests_for_asset(node.id).items():
            run_test(test_id, test)


def test(runtime: Runtime) -> None:
    """ Runs the selected data quality tests directly, independent of materialization, in the
    combined graph's dependency order (a test never runs before the asset(s) it's attached
    to) - repipe `test`. """
    for node in runtime.selected_nodes():
        if node.kind == NodeKind.TEST:
            run_test(node.id, cast(RegisteredDataQualityTest, node.obj))


def materialize(
        asset: Asset,
        graph: dict[str, GraphNode],
        results: dict[str, AssetReturnType],
) -> AssetReturnType:
    """ Calls the asset's function with its context and resolved upstream dependencies,
    persists the result through the asset's configured storage, and caches it in `results`
    so downstream assets materialized later in the same run can reuse it directly. """
    context = AssetContext(manifest=asset, dependencies=asset.settings.dependency_names)

    kwargs: dict[str, Any] = {}
    for param_name in inspect.signature(asset.func).parameters:
        kwargs[param_name] = (
            context if param_name == "context" else _resolve_dependency(param_name, graph, results)
        )

    result = asset.func(**kwargs)

    get_storage_concrete(asset.settings.storage).save(asset.settings.storage, asset, result)
    results[asset.name] = result
    print(f"materialized {asset.name}")
    return result


def _resolve_dependency(
        name: str,
        graph: dict[str, GraphNode],
        results: dict[str, AssetReturnType],
) -> AssetReturnType:
    """ The in-memory result if this dependency was just materialized earlier in the same run,
    otherwise its last persisted output - so `run my_downstream_asset` (no `+`) can skip
    re-running `my_asset` while still reading its existing data. """
    if name in results:
        return results[name]

    node = graph.get(name)
    if node is None or node.kind != NodeKind.ASSET:
        raise ValueError(f"{name!r} is not a known asset - check its @asset dependencies/parameters")

    upstream = cast(Asset, node.obj)
    return get_storage_concrete(upstream.settings.storage).load(upstream.settings.storage, upstream)


def run_test(test_id: str, test: RegisteredDataQualityTest) -> None:
    print(test_id)  # TODO: actually invoke test.func and handle fail_type
