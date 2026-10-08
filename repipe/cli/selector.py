from collections import defaultdict
from enum import StrEnum
from typing import Iterable, Iterator, Mapping, Self

from pydantic import BaseModel, model_validator

from repipe.asset.asset import Asset
from repipe.testing.data_quality import RegisteredDataQualityTest


class NodeKind(StrEnum):
    ASSET = "asset"
    TEST = "test"

# TODO: make this allow for either {test_name} or {asset_name}::{test_name}

class GraphNode(BaseModel):
    """ A single selectable unit in the combined asset/test graph. """
    model_config = {"arbitrary_types_allowed": True}

    id: str
    """ Unique within the graph: 
        1. the asset's name
        2. the test's name, qualified by ID (e.g. {asset_name}::{test_name}`) """

    kind: NodeKind
    name: str
    """ The asset's or test's own name - not unique for tests, since the same test may be
    attached to multiple assets. Selecting a reused test by this bare name is ambiguous - see
    `Selector.resolve` - so it must be qualified by its full `id` instead. """

    upstream: frozenset[str]
    labels: frozenset[str] = frozenset()
    """ Tests have no labels of their own - they inherit whichever asset they're attached to. """

    obj: Asset | RegisteredDataQualityTest
    """ The underlying object this node represents - the Asset for an ASSET node, or the
    RegisteredDataQualityTest for a TEST node. Kept here so a resolved selection of node ids
    can be turned straight back into real objects, without a second name->object lookup. """


def build_graph(assets: Mapping[str, Asset]) -> dict[str, GraphNode]:
    """ Builds one combined graph of assets and the data quality tests attached to them.
    A test's upstream is the asset(s) it's declared on, and it inherits that asset's labels. """
    nodes: dict[str, GraphNode] = {}

    for asset_name, manifest in assets.items():
        labels = manifest.settings.labels

        nodes[asset_name] = GraphNode(
            id=asset_name,
            kind=NodeKind.ASSET,
            name=asset_name,
            upstream=frozenset(manifest.settings.dependency_names),
            labels=labels,
            obj=manifest,
        )

        for test in manifest.settings.tests or []:
            registered = getattr(test, "__data_quality_test__")
            test_id = f"{asset_name}::{registered.name}"
            nodes[test_id] = GraphNode(
                id=test_id,
                kind=NodeKind.TEST,
                name=registered.name,
                upstream=frozenset({asset_name}),
                labels=labels,
                obj=registered,
            )

    return nodes


class Selection(BaseModel):
    name: str | None = None
    label: str | None = None

    @model_validator(mode='after')
    def validate_target_exists(self) -> Self:
        if not (self.name or self.label):
            raise ValueError("Name or label must be specified")
        if self.name and self.label:
            raise ValueError("Name & label are mutually exclusive")
        return self

    def matches(self, node: GraphNode) -> bool:
        if self.name is not None:
            return node.id == self.name or node.name == self.name
        return self.label in node.labels


class Selector(BaseModel):
    target: Selection

    upstream: bool = False
    downstream: bool = False
    exclude: bool = False

    @classmethod
    def parse(cls, raw: str) -> Self:
        """
        Parses a single dbt-style selector token:
        `my_asset` -> just my_asset
        `+my_asset` -> my_asset and everything upstream of it
        `my_asset+` -> my_asset and everything downstream of it (including its tests)
        `+my_asset+` -> my_asset plus both directions
        `label:nightly` -> assets (and their tests) carrying the `nightly` label
        `!my_asset` -> excludes my_asset (and its own +/- expansion) from the result
        """
        text = raw.strip()
        if not text:
            raise ValueError("Selector cannot be empty")

        exclude = text.startswith('!')
        if exclude:
            text = text[1:]

        upstream = text.startswith('+')
        if upstream:
            text = text[1:]

        downstream = text.endswith('+')
        if downstream:
            text = text[:-1]

        if not text:
            raise ValueError(f"Selector {raw!r} has no target")

        if text.startswith('label:'):
            target = Selection(label=text[len('label:'):])
        else:
            target = Selection(name=text)

        return cls(target=target, upstream=upstream, downstream=downstream, exclude=exclude)

    def resolve(self, graph: dict[str, GraphNode]) -> set[str]:
        """ Resolves this selector against the combined asset/test graph, returning matched
        node ids expanded upstream/downstream as requested. """
        matched = {node.id for node in graph.values() if self.target.matches(node)}
        if not matched:
            raise ValueError(f"No asset or test matched {self.target.name or self.target.label!r}")

        _check_unambiguous(self.target, matched, graph)

        result = set(matched)
        if self.upstream:
            result |= _ancestors(matched, graph)
        if self.downstream:
            result |= _descendants(matched, graph)
        return result


def _check_unambiguous(target: Selection, matched: set[str], graph: dict[str, GraphNode]) -> None:
    """ A bare name matches a node's `id` uniquely by construction, except for a test reused
    across multiple assets, where it also matches each one's `name`. Label selections are
    exempt - matching every test carrying a label is the point, not an ambiguity. """
    if target.name is None:
        return

    test_matches = sorted(node_id for node_id in matched if graph[node_id].kind == NodeKind.TEST)
    if len(test_matches) > 1:
        raise ValueError(
            f"{target.name!r} matches multiple tests ({', '.join(test_matches)}) - "
            f"select one by its full id instead, e.g. {test_matches[0]!r}"
        )


def _ancestors(start: set[str], graph: dict[str, GraphNode]) -> set[str]:
    seen: set[str] = set()
    frontier = set(start)
    while frontier:
        next_frontier: set[str] = set()
        for node_id in frontier:
            for upstream_id in graph[node_id].upstream:
                if upstream_id not in seen:
                    seen.add(upstream_id)
                    next_frontier.add(upstream_id)
        frontier = next_frontier
    return seen


def _descendants(start: set[str], graph: dict[str, GraphNode]) -> set[str]:
    children: dict[str, set[str]] = defaultdict(set)
    for node in graph.values():
        for upstream_id in node.upstream:
            children[upstream_id].add(node.id)

    seen: set[str] = set()
    frontier = set(start)
    while frontier:
        next_frontier: set[str] = set()
        for node_id in frontier:
            for child_id in children[node_id]:
                if child_id not in seen:
                    seen.add(child_id)
                    next_frontier.add(child_id)
        frontier = next_frontier
    return seen


def topologically_sorted(
        graph: Mapping[str, GraphNode],
        node_ids: Iterable[str] | None = None,
) -> Iterator[GraphNode]:
    """ Uses Kahn's algorithm to order nodes - assets and tests alike - by their `upstream`
    dependency graph. `node_ids` restricts the sort to a subset of `graph` (default: all of
    it); dependencies pointing outside that subset are treated as already satisfied, so this
    also works to order a selected subset of a larger graph.

    A test is only ever upstream-dependent on the asset(s) it's attached to, so it's always
    emitted no earlier than those assets - this is what gives `repipe test` a deterministic,
    dependency-respecting run order instead of the arbitrary order of a `set`. """
    selected = set(node_ids) if node_ids is not None else set(graph.keys())

    remaining = {
        node_id: [dep for dep in graph[node_id].upstream if dep in selected]
        for node_id in selected
    }
    emitted: set[str] = set()

    while remaining:
        ready = sorted(
            node_id for node_id, deps in remaining.items()
            if all(dep in emitted for dep in deps)
        )
        if not ready:
            raise ValueError(f"Cycle detected among node(s): {sorted(remaining)}")

        for node_id in ready:
            emitted.add(node_id)
            del remaining[node_id]
            yield graph[node_id]


class ResourceQuery(BaseModel):
    """ A set of dbt-style selector tokens (e.g. CLI `--select` args), resolved together
    against the combined asset/test graph. An empty query selects everything. """
    selectors: list[Selector]

    @classmethod
    def parse(cls, raw: list[str]) -> Self:
        return cls(selectors=[Selector.parse(token) for token in raw])

    def resolve(self, graph: dict[str, GraphNode]) -> set[str]:
        includes = [s for s in self.selectors if not s.exclude]
        excludes = [s for s in self.selectors if s.exclude]

        if includes:
            result: set[str] = set()
            for selector in includes:
                result |= selector.resolve(graph)
        else:
            result = set(graph.keys())

        for selector in excludes:
            result -= selector.resolve(graph)

        return result
