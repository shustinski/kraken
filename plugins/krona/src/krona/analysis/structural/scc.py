from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class StronglyConnectedComponent:
    nodes: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.nodes)


class TarjanSCCDetector:
    """
    Tarjan SCC implementation used by the structural analyzer.

    It operates on a plain adjacency map to avoid coupling graph construction to a
    specific graph library.
    """

    def __init__(self, adjacency: dict[str, set[str]]):
        self._adjacency = adjacency
        self._index = 0
        self._stack: list[str] = []
        self._on_stack: set[str] = set()
        self._indexes: dict[str, int] = {}
        self._lowlinks: dict[str, int] = {}
        self._components: list[StronglyConnectedComponent] = []

    def run(self) -> list[StronglyConnectedComponent]:
        for node in self._adjacency:
            if node not in self._indexes:
                self._strong_connect(node)
        return self._components

    def _strong_connect(self, node: str) -> None:
        self._indexes[node] = self._index
        self._lowlinks[node] = self._index
        self._index += 1
        self._stack.append(node)
        self._on_stack.add(node)

        for neighbor in self._adjacency.get(node, set()):
            if neighbor not in self._indexes:
                self._strong_connect(neighbor)
                self._lowlinks[node] = min(self._lowlinks[node], self._lowlinks[neighbor])
            elif neighbor in self._on_stack:
                self._lowlinks[node] = min(self._lowlinks[node], self._indexes[neighbor])

        if self._lowlinks[node] == self._indexes[node]:
            component_nodes: list[str] = []
            while True:
                current = self._stack.pop()
                self._on_stack.remove(current)
                component_nodes.append(current)
                if current == node:
                    break
            self._components.append(StronglyConnectedComponent(nodes=tuple(component_nodes)))


def adjacency_from_edges(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for source, target in edges:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set())
    return adjacency
