"""Robust translation-only local-block solver. Small custom graph, no NetworkX."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cartograph.domain.coordinates import GridCoordinate, Translation2D
from cartograph.domain.registration import RegistrationStatus
from cartograph.domain.topology import GraphEdge, LocalBlockSolution, LocalGraph


@dataclass(frozen=True, slots=True)
class HuberOptimizerSettings:
    delta_px: float = 2.0
    iterations: int = 8
    min_weight: float = 1e-6


class HuberTranslationOptimizer:
    """min sum w_ij ρ(||(p_j - p_i) - d_ij||²) with p_center = (0, 0)."""

    def __init__(self, settings: HuberOptimizerSettings | None = None) -> None:
        self._settings = settings or HuberOptimizerSettings()

    def optimize(self, graph: LocalGraph) -> LocalBlockSolution:
        nodes = tuple(sorted(set(graph.nodes) | {graph.center}))
        poses = {graph.center: Translation2D(0.0, 0.0)}
        edges = [edge for edge in graph.edges if edge.weight > 0.0 and edge.result.is_usable]
        excluded = [edge for edge in graph.edges if edge not in edges]
        if not edges:
            for node in nodes:
                poses.setdefault(node, Translation2D(0.0, 0.0))
            return LocalBlockSolution(
                center=graph.center,
                poses=poses,
                graph=graph,
                excluded_edges=tuple(excluded),
                status=RegistrationStatus.FAILED,
                message="no usable pairwise edges",
            )

        variables = [node for node in nodes if node != graph.center]
        index = {node: i for i, node in enumerate(variables)}
        weights = np.array([max(edge.weight, self._settings.min_weight) for edge in edges], dtype=np.float64)
        original = weights.copy()
        tree = _maximum_spanning_tree(edges, nodes)
        if tree:
            tree_set = {(edge.source, edge.target) for edge in tree}
            weights = np.array(
                [original[i] if (edge.source, edge.target) in tree_set else self._settings.min_weight for i, edge in enumerate(edges)],
                dtype=np.float64,
            )
        solution = _solve_weighted(edges, index, graph.center, weights)
        for _ in range(max(1, self._settings.iterations)):
            residuals = _edge_residuals(edges, index, graph.center, solution)
            updated = np.array(
                [_huber_weight(residual, self._settings.delta_px) * original[i] for i, residual in enumerate(residuals)],
                dtype=np.float64,
            )
            updated = np.maximum(updated, self._settings.min_weight)
            if np.max(np.abs(updated - weights)) < 1e-6:
                weights = updated
                break
            weights = updated
            solution = _solve_weighted(edges, index, graph.center, weights)

        for node, variable_index in index.items():
            poses[node] = Translation2D(
                float(solution[2 * variable_index]),
                float(solution[2 * variable_index + 1]),
            )
        for node in nodes:
            poses.setdefault(node, Translation2D(0.0, 0.0))

        status = RegistrationStatus.OK
        message = ""
        if any(edge.result.status is not RegistrationStatus.OK for edge in graph.edges):
            status = RegistrationStatus.LOW_CONFIDENCE
            message = "block contains low-confidence or rejected pairs"
        if excluded:
            status = RegistrationStatus.LOW_CONFIDENCE
            message = "one or more pairwise edges were excluded"
        return LocalBlockSolution(
            center=graph.center,
            poses=poses,
            graph=graph,
            excluded_edges=tuple(excluded),
            status=status,
            message=message,
        )


def _huber_weight(residual: float, delta: float) -> float:
    if residual <= delta:
        return 1.0
    if residual <= 0.0:
        return 1.0
    return float(delta / residual)


def _solve_weighted(
    edges: list[GraphEdge],
    index: dict[GridCoordinate, int],
    center: GridCoordinate,
    weights: np.ndarray,
) -> np.ndarray:
    rows = len(edges) * 2
    cols = len(index) * 2
    design = np.zeros((rows, cols), dtype=np.float64)
    rhs = np.zeros(rows, dtype=np.float64)
    for edge_index, edge in enumerate(edges):
        scale = float(np.sqrt(max(weights[edge_index], 0.0)))
        _fill_translation_rows(design, rhs, edge, index, center, 2 * edge_index, scale)
    if cols == 0:
        return np.zeros(0, dtype=np.float64)
    solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    return solution


def _fill_translation_rows(
    design: np.ndarray,
    rhs: np.ndarray,
    edge: GraphEdge,
    index: dict[GridCoordinate, int],
    center: GridCoordinate,
    row: int,
    scale: float,
) -> None:
    # x_j - x_i = dx, y_j - y_i = dy
    for axis, value in ((0, edge.measurement.dx), (1, edge.measurement.dy)):
        target_row = row + axis
        rhs[target_row] = value * scale
        if edge.source != center:
            design[target_row, 2 * index[edge.source] + axis] = -scale
        if edge.target != center:
            design[target_row, 2 * index[edge.target] + axis] = scale


def _edge_residuals(
    edges: list[GraphEdge],
    index: dict[GridCoordinate, int],
    center: GridCoordinate,
    solution: np.ndarray,
) -> list[float]:
    residuals: list[float] = []
    for edge in edges:
        source = _pose(edge.source, index, center, solution)
        target = _pose(edge.target, index, center, solution)
        err_x = (target[0] - source[0]) - edge.measurement.dx
        err_y = (target[1] - source[1]) - edge.measurement.dy
        residuals.append(float(np.hypot(err_x, err_y)))
    return residuals


def _maximum_spanning_tree(edges: list[GraphEdge], nodes: tuple[GridCoordinate, ...]) -> list[GraphEdge]:
    parent = {node: node for node in nodes}

    def find(node: GridCoordinate) -> GridCoordinate:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    tree: list[GraphEdge] = []
    for edge in sorted(edges, key=lambda item: (item.weight, item.result.confidence), reverse=True):
        source_root = find(edge.source)
        target_root = find(edge.target)
        if source_root == target_root:
            continue
        parent[source_root] = target_root
        tree.append(edge)
        if len(tree) >= len(nodes) - 1:
            break
    return tree


def _pose(
    coord: GridCoordinate,
    index: dict[GridCoordinate, int],
    center: GridCoordinate,
    solution: np.ndarray,
) -> tuple[float, float]:
    if coord == center:
        return 0.0, 0.0
    variable = index[coord]
    return float(solution[2 * variable]), float(solution[2 * variable + 1])
