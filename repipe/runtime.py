import inspect
import itertools
import types
import typing
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterable, Iterator, Mapping, cast

import duckdb
import polars as pl
from pydantic import BaseModel

from repipe.asset.asset import Asset, AssetContext, AssetFunction, Output
from repipe.cli.selector import GraphNode, NodeKind, ResourceQuery, topologically_sorted
from repipe.storage import get_storage_concrete
from repipe.storage.session import WriteSession
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
    async for _ in _materialize_selected(runtime):
        pass


async def build(runtime: Runtime) -> None:
    """ Materializes the selected assets, in dependency order, running each asset's own
    attached tests immediately after it materializes - repipe `build`. """
    async for function in _materialize_selected(runtime):
        for asset_name in function.outputs:
            for test_id, test in runtime.tests_for_asset(asset_name).items():
                run_test(test_id, test)


async def test(runtime: Runtime) -> None:
    """ Runs the selected data quality tests directly, independent of materialization, in the
    combined graph's dependency order (a test never runs before the asset(s) it's attached
    to) - repipe `test`. """
    for node in runtime.selected_nodes():
        if node.kind == NodeKind.TEST:
            run_test(node.id, cast(RegisteredDataQualityTest, node.obj))


async def _materialize_selected(runtime: Runtime) -> AsyncIterator[AssetFunction]:
    """ Runs each selected asset's function once, however many of its outputs are selected. """
    nodes = [node for node in runtime.selected_nodes() if node.kind == NodeKind.ASSET]
    selected = {node.id for node in nodes}
    done: set[int] = set()
    for node in nodes:
        function = cast(Asset, node.obj).function
        if id(function) not in done:
            done.add(id(function))
            await materialize_function(function, runtime.graph, selected)
            yield function


async def materialize(asset: Asset, graph: dict[str, GraphNode]) -> None:
    """ Runs `asset`'s function, writing every asset it produces. """
    await materialize_function(asset.function, graph)


@dataclass(frozen=True)
class Write:
    asset: str
    data: Any
    partition: Mapping[str, Any] | None = None


async def materialize_function(
        function: AssetFunction,
        graph: dict[str, GraphNode],
        selected: Iterable[str] | None = None,
) -> None:
    outputs = [cast(Asset, graph[name].obj) for name in function.outputs]
    kwargs = _resolve_inputs(function, outputs, graph)

    session = WriteSession(outputs, new_run_id(), function.is_multi)
    try:
        async for write in _iter_writes(function, kwargs):
            session.stage(write.asset, _duckdb_relation_to_polars(write.data), write.partition)
        session.commit()
    except BaseException:
        session.abort()
        raise

    unselected = [name for name in function.outputs if selected is not None and name not in selected]
    if function.is_multi:
        note = f" ({', '.join(unselected)} not selected)" if unselected else ""
        print(f"materialized {function.name} -> {', '.join(function.outputs)}{note}")
    else:
        print(f"materialized {function.name}")


def _resolve_inputs(function: AssetFunction, outputs: list[Asset], graph: dict[str, GraphNode]) -> dict[str, Any]:
    own_outputs = {asset.name: asset for asset in outputs}
    context = AssetContext(manifest=outputs[0], dependencies=function.dependency_names)

    parameters = inspect.signature(function.func).parameters
    try:
        annotations = typing.get_type_hints(function.func)
    except Exception:  # unresolvable forward reference - fall back to the raw annotations
        annotations = {}

    kwargs: dict[str, Any] = {}
    for param_name in parameters:
        annotation = annotations.get(param_name, parameters[param_name].annotation)
        if param_name == "context":
            kwargs[param_name] = context
        elif param_name in own_outputs:
            # Its own previous output (None on the first run)
            own = own_outputs[param_name]
            previous = get_storage_concrete(own.settings.storage).load_previous(own.settings.storage, own)
            kwargs[param_name] = _as_annotated(previous, annotation)
        else:
            kwargs[param_name] = _as_annotated(_load_dependency(param_name, graph), annotation)
    return kwargs


def _load_dependency(name: str, graph: dict[str, GraphNode]) -> Any:
    node = graph.get(name)
    if node is None or node.kind != NodeKind.ASSET:
        raise ValueError(f"{name!r} is not a known asset - check its @asset dependencies/parameters")

    upstream = cast(Asset, node.obj)
    return get_storage_concrete(upstream.settings.storage).load(upstream.settings.storage, upstream)


async def _iter_writes(function: AssetFunction, kwargs: dict[str, Any]) -> AsyncIterator[Write]:
    func = function.func
    if inspect.isasyncgenfunction(func):
        async for item in func(**kwargs):
            yield _yielded_write(function, item)
    elif inspect.isgeneratorfunction(func):
        for item in func(**kwargs):
            yield _yielded_write(function, item)
    else:
        result = await func(**kwargs) if inspect.iscoroutinefunction(func) else func(**kwargs)
        for write in _returned_writes(function, result):
            yield write


def _returned_writes(function: AssetFunction, result: Any) -> list[Write]:
    if isinstance(result, Output):
        return [_output_write(function, result)]
    if isinstance(result, (list, tuple)):
        return [_output_write(function, _as_output(function, item)) for item in result]
    if not function.is_multi:
        return [Write(function.outputs[0], result)]
    if not isinstance(result, Mapping):
        raise TypeError(
            f"multi_asset {function.name!r} returned {type(result).__name__!r} - return a dict "
            f"keyed by output name, or an Output"
        )
    return [_dict_entry_write(function, name, value) for name, value in result.items()]


def _as_output(function: AssetFunction, item: Any) -> Output:
    if not isinstance(item, Output):
        raise TypeError(
            f"asset {function.name!r} returned a list/tuple, so every item must be an Output - "
            f"got {type(item).__name__!r}"
        )
    return item


def _dict_entry_write(function: AssetFunction, name: str, value: Any) -> Write:
    """ A multi-asset's dict return only needs `Output` per entry once that entry needs a second
    axis - typically a partition, since the asset name is already the dict key. """
    if isinstance(value, Output):
        if value.asset is not None and value.asset != name:
            raise ValueError(
                f"multi_asset {function.name!r} returned {name!r} as a dict key, but its Output "
                f"names asset {value.asset!r}"
            )
        return _output_write(function, replace(value, asset=name))
    return _output_write(function, Output(value, asset=name))


def _yielded_write(function: AssetFunction, item: Any) -> Write:
    if isinstance(item, Output):
        return _output_write(function, item)
    if function.is_multi:
        raise TypeError(f"multi_asset {function.name!r} yielded plain data - yield Output(data, asset=...) instead")
    return Write(function.outputs[0], item)


def _output_write(function: AssetFunction, output: Output) -> Write:
    if output.asset is None:
        if function.is_multi:
            raise TypeError(f"multi_asset {function.name!r} must name the asset each Output is for")
        return Write(function.outputs[0], output.data, output.partition)
    if output.asset not in function.outputs:
        raise ValueError(f"{function.name!r} wrote to {output.asset!r}, which isn't one of its outputs {function.outputs!r}")
    return Write(output.asset, output.data, output.partition)


def _duckdb_relation_to_polars(obj: Any) -> Any:
    return obj.pl() if isinstance(obj, duckdb.DuckDBPyRelation) else obj


def _as_annotated(value: Any, annotation: object) -> Any:
    """ Upstreams load lazily; a parameter annotated as a `DataFrame` or DuckDB relation gets one. """
    if value is None:
        return None
    if _allows(annotation, duckdb.DuckDBPyRelation):
        con = duckdb.connect()
        con.register("value", value)
        return con.table("value")
    if isinstance(value, pl.LazyFrame) and _allows(annotation, pl.DataFrame) and not _allows(annotation, pl.LazyFrame):
        return value.collect()
    return value


def _allows(annotation: object, expected: type) -> bool:
    if annotation is expected:
        return True
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return any(_allows(arg, expected) for arg in typing.get_args(annotation))
    return False


def run_test(test_id: str, test: RegisteredDataQualityTest) -> None:
    print(test_id)  # TODO: actually invoke test.func and handle fail_type
