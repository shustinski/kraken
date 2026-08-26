"""GASP, Mutex Watershed, Multicut and Lifted Multicut backends."""

from __future__ import annotations

import heapq
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from ...application.preview_cancellation import raise_if_preview_cancelled
from .features import MetalStructuralFeatures, float_map_to_u8
from .gradient_watershed import analyze_metal_presence
from .material_classifier import classify_partition_material
from .signed_graph import SignedAffinityBuilder, SignedAffinityGraph
from .strategy_contracts import StrategySegmentation


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int32)
        self.size = np.ones(size, dtype=np.int64)

    def find(self, value: int) -> int:
        root = value
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while int(self.parent[value]) != value:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, first: int, second: int) -> tuple[int, int]:
        left = self.find(first)
        right = self.find(second)
        if left == right:
            return left, right
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]
        return left, right


@dataclass(slots=True)
class _EdgeStats:
    attractive_sum: float
    repulsive_sum: float
    count: int
    maximum_attractive: float
    maximum_repulsive: float

    def combine(self, other: _EdgeStats) -> _EdgeStats:
        return _EdgeStats(
            attractive_sum=self.attractive_sum + other.attractive_sum,
            repulsive_sum=self.repulsive_sum + other.repulsive_sum,
            count=self.count + other.count,
            maximum_attractive=max(self.maximum_attractive, other.maximum_attractive),
            maximum_repulsive=max(self.maximum_repulsive, other.maximum_repulsive),
        )


def _linkage_score(stats: _EdgeStats, criterion: str) -> float:
    if criterion == "sum":
        return stats.attractive_sum - stats.repulsive_sum
    if criterion == "mutex_abs_max":
        return (
            stats.maximum_attractive
            if stats.maximum_attractive >= stats.maximum_repulsive
            else -stats.maximum_repulsive
        )
    return (stats.attractive_sum - stats.repulsive_sum) / max(1, stats.count)


def _agglomerate(
    node_count: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    attraction: np.ndarray,
    repulsion: np.ndarray,
    *,
    criterion: str,
    minimum_score: float,
    maximum_repulsive_conflict: float | None,
    maximum_operations: int,
    time_limit_seconds: float | None = None,
    initial_partition: np.ndarray | None = None,
) -> tuple[np.ndarray, int, bool]:
    dsu = _DisjointSet(node_count + 1)
    if initial_partition is not None:
        representatives: dict[int, int] = {}
        for node in range(1, node_count + 1):
            label = int(initial_partition[node])
            representative = representatives.setdefault(label, node)
            dsu.union(representative, node)
    adjacency: dict[int, dict[int, _EdgeStats]] = {index: {} for index in range(1, node_count + 1)}
    heap: list[tuple[float, int, int]] = []
    for first, second, attractive, repulsive in zip(edge_u, edge_v, attraction, repulsion, strict=True):
        left = dsu.find(int(first))
        right = dsu.find(int(second))
        if left == right:
            continue
        stats = _EdgeStats(float(attractive), float(repulsive), 1, float(attractive), float(repulsive))
        existing = adjacency[left].get(right)
        if existing is not None:
            stats = existing.combine(stats)
        adjacency[left][right] = stats
        adjacency[right][left] = stats
        heapq.heappush(heap, (-_linkage_score(stats, criterion), left, right))

    started = perf_counter()
    operations = 0
    time_limited = False
    while heap and operations < maximum_operations:
        if operations % 512 == 0:
            raise_if_preview_cancelled()
        if time_limit_seconds is not None and perf_counter() - started >= time_limit_seconds:
            time_limited = True
            break
        negative_score, first, second = heapq.heappop(heap)
        left = dsu.find(first)
        right = dsu.find(second)
        if left == right:
            continue
        current_stats = adjacency.get(left, {}).get(right)
        if current_stats is None:
            continue
        current_score = _linkage_score(current_stats, criterion)
        if abs(current_score + negative_score) > 1e-7:
            heapq.heappush(heap, (-current_score, left, right))
            continue
        if current_score <= minimum_score:
            break
        conflict = current_stats.repulsive_sum / max(1, current_stats.count)
        if maximum_repulsive_conflict is not None and conflict > maximum_repulsive_conflict:
            adjacency[left].pop(right, None)
            adjacency[right].pop(left, None)
            continue

        root, absorbed = dsu.union(left, right)
        root_edges = adjacency.setdefault(root, {})
        absorbed_edges = adjacency.pop(absorbed, {})
        root_edges.pop(absorbed, None)
        for neighbour, absorbed_stats in list(absorbed_edges.items()):
            neighbour_root = dsu.find(neighbour)
            if neighbour_root in {root, absorbed}:
                continue
            neighbour_edges = adjacency.setdefault(neighbour_root, {})
            neighbour_edges.pop(absorbed, None)
            existing = root_edges.get(neighbour_root)
            combined = absorbed_stats if existing is None else existing.combine(absorbed_stats)
            root_edges[neighbour_root] = combined
            neighbour_edges[root] = combined
            heapq.heappush(heap, (-_linkage_score(combined, criterion), root, neighbour_root))
        operations += 1

    roots = np.zeros(node_count + 1, dtype=np.int32)
    root_to_label: dict[int, int] = {}
    for node in range(1, node_count + 1):
        root = dsu.find(node)
        roots[node] = root_to_label.setdefault(root, len(root_to_label) + 1)
    return roots, operations, time_limited


def _mutex_watershed_partition(
    graph: SignedAffinityGraph,
    parameters: Mapping[str, Any],
    lifted_mutex: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, int]:
    attractive_scale = float(parameters.get("attractive_weight_scale", 1.0))
    mutex_scale = float(parameters.get("mutex_weight_scale", 1.0))
    attractive = graph.attraction * attractive_scale
    repulsive = graph.repulsion * mutex_scale
    if str(parameters.get("attractive_neighborhood_offsets", "local")) == "local":
        attractive = np.where(graph.edge_diagonal_only, 0.0, attractive)
    if str(parameters.get("edge_ordering", "descending_confidence")) == "signed_margin":
        margin = attractive - repulsive
        attractive = np.maximum(margin, 0.0)
        repulsive = np.maximum(-margin, 0.0)
    edge_u = graph.edge_u
    edge_v = graph.edge_v
    if lifted_mutex is not None:
        lifted_u, lifted_v, lifted_weight = lifted_mutex
        edge_u = np.r_[edge_u, lifted_u]
        edge_v = np.r_[edge_v, lifted_v]
        attractive = np.r_[attractive, np.zeros(lifted_weight.shape, dtype=np.float32)]
        repulsive = np.r_[repulsive, lifted_weight * mutex_scale]

    kinds = np.r_[np.zeros(attractive.size, dtype=np.uint8), np.ones(repulsive.size, dtype=np.uint8)]
    weights = np.r_[attractive, repulsive]
    first = np.r_[edge_u, edge_u]
    second = np.r_[edge_v, edge_v]
    valid = weights > 0.0
    kinds = kinds[valid]
    weights = weights[valid]
    first = first[valid]
    second = second[valid]
    # Mutex first on exact ties makes the exclusion semantics deterministic.
    order = np.lexsort((-kinds, -weights))
    dsu = _DisjointSet(graph.node_count + 1)
    mutex: dict[int, set[int]] = {node: set() for node in range(1, graph.node_count + 1)}
    processed = 0
    minimum_mutex = float(parameters.get("minimum_mutex_confidence", 0.55))
    for index in order:
        if processed % 1024 == 0:
            raise_if_preview_cancelled()
        left = dsu.find(int(first[index]))
        right = dsu.find(int(second[index]))
        if left == right:
            continue
        if int(kinds[index]) == 1:
            if float(weights[index]) < minimum_mutex:
                continue
            mutex[left].add(right)
            mutex[right].add(left)
            processed += 1
            continue
        if right in mutex[left] or left in mutex[right]:
            continue
        root, absorbed = dsu.union(left, right)
        combined_mutex = {dsu.find(item) for item in mutex[root] | mutex[absorbed]}
        combined_mutex.discard(root)
        combined_mutex.discard(absorbed)
        mutex[root] = combined_mutex
        mutex.pop(absorbed, None)
        for other in combined_mutex:
            other_root = dsu.find(other)
            values = mutex.setdefault(other_root, set())
            values.discard(absorbed)
            values.discard(left)
            values.discard(right)
            values.add(root)
        processed += 1
    roots = np.zeros(graph.node_count + 1, dtype=np.int32)
    mapping: dict[int, int] = {}
    for node in range(1, graph.node_count + 1):
        if node % 128 == 0:
            raise_if_preview_cancelled()
        root = dsu.find(node)
        roots[node] = mapping.setdefault(root, len(mapping) + 1)
    return roots, processed


def _multicut_costs(
    attraction: np.ndarray,
    repulsion: np.ndarray,
    parameters: Mapping[str, Any],
) -> np.ndarray:
    attraction_scale = float(parameters.get("attraction_cost_scale", 1.0))
    repulsion_scale = float(parameters.get("repulsion_cost_scale", 1.0))
    bias = float(parameters.get("affinity_bias", 0.5))
    if str(parameters.get("cost_transform", "log_odds")) == "signed_linear":
        return (attraction_scale * attraction - repulsion_scale * repulsion - (bias - 0.5)).astype(np.float32)
    epsilon = 1e-5
    join_probability = np.clip(
        (attraction_scale * attraction + epsilon)
        / (attraction_scale * attraction + repulsion_scale * repulsion + 2.0 * epsilon),
        epsilon,
        1.0 - epsilon,
    )
    bias_logit = np.log(bias / max(1.0 - bias, epsilon))
    return (np.log(join_probability / (1.0 - join_probability)) - bias_logit).astype(np.float32)


def _lifted_relations(
    graph: SignedAffinityGraph,
    features: MetalStructuralFeatures,
    parameters: Mapping[str, Any],
    *,
    mutex_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if graph.node_count <= 1:
        empty_i = np.empty(0, dtype=np.int32)
        empty_f = np.empty(0, dtype=np.float32)
        return empty_i, empty_i, empty_f, empty_f, np.zeros(features.gray.shape, dtype=np.float32)
    minimum = int(parameters.get("minimum_lifted_distance", 2 if mutex_only else 4))
    maximum = (
        int(parameters.get("long_range_mutex_distance", 8))
        if mutex_only
        else int(parameters.get("maximum_lifted_distance", 24))
    )
    step = max(1, int(parameters.get("lifted_distance_step", max(1, minimum))))
    threshold = (
        float(parameters.get("minimum_mutex_confidence", 0.55))
        if mutex_only
        else float(parameters.get("lifted_confidence_threshold", 0.6))
    )
    maximum_edges = (
        int(parameters.get("maximum_lifted_edges", 200000)) if not mutex_only else min(200000, graph.node_count * 8)
    )
    orientation_aligned = bool(parameters.get("orientation_aligned_lifted_edges", True)) or mutex_only
    pairs: dict[tuple[int, int], tuple[float, float]] = {}
    labels = graph.pixel_labels
    height, width = labels.shape
    for node in range(1, graph.node_count + 1):
        x_coord, y_coord = graph.node_centroids[node]
        angle = float(graph.node_orientation[node]) + 0.5 * np.pi
        directions = (angle, angle + np.pi) if orientation_aligned else (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0)
        for distance in range(minimum, maximum + 1, step):
            for direction in directions:
                target_x = round(float(x_coord) + np.cos(direction) * distance)
                target_y = round(float(y_coord) + np.sin(direction) * distance)
                if target_x < 0 or target_x >= width or target_y < 0 or target_y >= height:
                    continue
                other = int(labels[target_y, target_x])
                if other <= 0 or other == node:
                    continue
                pair = (min(node, other), max(node, other))
                orientation_similarity = 0.5 + 0.5 * np.cos(
                    2.0 * (graph.node_orientation[node] - graph.node_orientation[other])
                )
                intensity_similarity = np.exp(
                    -abs(float(graph.node_intensity[node] - graph.node_intensity[other])) / 0.12
                )
                midpoint_x = int(np.clip(round((float(x_coord) + target_x) * 0.5), 0, width - 1))
                midpoint_y = int(np.clip(round((float(y_coord) + target_y) * 0.5), 0, height - 1))
                crossed_boundary = float(features.boundary_strength[midpoint_y, midpoint_x])
                attractive = float(orientation_similarity * intensity_similarity * (1.0 - crossed_boundary))
                repulsive = float(
                    crossed_boundary * (0.5 + 0.5 * features.orientation_persistence[midpoint_y, midpoint_x])
                )
                if mutex_only:
                    attractive = 0.0
                else:
                    if not bool(parameters.get("same_trace_lifted_attraction", True)):
                        attractive = 0.0
                    if not bool(parameters.get("cross_boundary_lifted_repulsion", True)):
                        repulsive = 0.0
                if max(attractive, repulsive) < threshold:
                    continue
                previous = pairs.get(pair)
                if previous is None:
                    pairs[pair] = (attractive, repulsive)
                else:
                    pairs[pair] = (max(previous[0], attractive), max(previous[1], repulsive))
                if len(pairs) >= maximum_edges:
                    break
            if len(pairs) >= maximum_edges:
                break
        if len(pairs) >= maximum_edges:
            break
    if not pairs:
        empty_i = np.empty(0, dtype=np.int32)
        empty_f = np.empty(0, dtype=np.float32)
        return empty_i, empty_i, empty_f, empty_f, np.zeros(features.gray.shape, dtype=np.float32)
    sorted_items = sorted(pairs.items())
    edge_u = np.array([pair[0] for pair, _weights in sorted_items], dtype=np.int32)
    edge_v = np.array([pair[1] for pair, _weights in sorted_items], dtype=np.int32)
    attractive_edges = np.array([weights[0] for _pair, weights in sorted_items], dtype=np.float32)
    repulsive_edges = np.array([weights[1] for _pair, weights in sorted_items], dtype=np.float32)
    debug = np.zeros(features.gray.shape, dtype=np.float32)
    for first, second, weight in zip(
        edge_u[:10000],
        edge_v[:10000],
        np.maximum(attractive_edges, repulsive_edges)[:10000],
        strict=True,
    ):
        first_point = tuple(np.rint(graph.node_centroids[int(first)]).astype(int))
        second_point = tuple(np.rint(graph.node_centroids[int(second)]).astype(int))
        cv2.line(debug, first_point, second_point, float(weight), 1, cv2.LINE_AA)
    return edge_u, edge_v, attractive_edges, repulsive_edges, debug


def _partition_labels(graph: SignedAffinityGraph, node_partition: np.ndarray) -> np.ndarray:
    return node_partition[graph.pixel_labels].astype(np.int32)


def _empty_result(image: np.ndarray) -> StrategySegmentation:
    empty = np.zeros(image.shape, dtype=np.uint8)
    return StrategySegmentation(
        empty, np.zeros(image.shape, dtype=np.int32), empty.astype(np.float32), empty.astype(np.float32)
    )


def _finalize_graph_result(
    strategy: str,
    graph: SignedAffinityGraph,
    node_partition: np.ndarray,
    features: MetalStructuralFeatures,
    parameters: Mapping[str, Any],
    *,
    solver_ms: float,
    total_started: float,
    debug_data: dict[str, Any],
    extra_debug: dict[str, np.ndarray] | None = None,
) -> StrategySegmentation:
    labels = _partition_labels(graph, node_partition)
    classification_started = perf_counter()
    material = classify_partition_material(labels, features, parameters)
    strong_separator = graph.repulsion_map >= float(parameters.get("minimum_repulsive_confidence", 0.52))
    consolidated_mask = material.mask.copy()
    consolidated_mask[strong_separator] = 0
    connectivity = 8 if str(parameters.get("connectivity", "4")) == "8" else 4
    consolidated_labels = cv2.connectedComponents(
        (consolidated_mask > 0).astype(np.uint8),
        connectivity=connectivity,
    )[1].astype(np.int32)
    classification_ms = (perf_counter() - classification_started) * 1000.0
    return StrategySegmentation(
        binary_mask=consolidated_mask,
        instance_labels=consolidated_labels,
        boundary_map=graph.repulsion_map,
        confidence_map=material.confidence_map,
        debug_images={
            f"metal_{strategy}_attractive_affinity": float_map_to_u8(graph.attraction_map),
            f"metal_{strategy}_repulsive_affinity": float_map_to_u8(graph.repulsion_map),
            f"metal_{strategy}_final_labels": ((consolidated_labels.astype(np.uint64) * 37) % 251).astype(np.uint8),
            **(extra_debug or {}),
            **material.debug_images,
        },
        debug_data={
            "node_count": graph.node_count,
            "local_edge_count": graph.edge_count,
            "partition_count": int(labels.max()),
            "material_instance_count": int(consolidated_labels.max()),
            **debug_data,
        },
        timings_ms={
            "feature_build": float(features.build_time_ms),
            "graph_construction": float(graph.build_time_ms),
            "solver": solver_ms,
            "material_classification": classification_ms,
            "total": (perf_counter() - total_started) * 1000.0,
        },
    )


def segment_gasp(
    image: np.ndarray,
    features: MetalStructuralFeatures,
    parameters: Mapping[str, Any],
) -> StrategySegmentation:
    total_started = perf_counter()
    if not analyze_metal_presence(image).has_metal:
        return _empty_result(image)
    graph = SignedAffinityBuilder().build(features, parameters)
    solver_started = perf_counter()
    repulsion = graph.repulsion if bool(parameters.get("use_signed_edges", True)) else np.zeros_like(graph.repulsion)
    node_partition, operations, limited = _agglomerate(
        graph.node_count,
        graph.edge_u,
        graph.edge_v,
        graph.attraction,
        repulsion,
        criterion=str(parameters.get("linkage_criterion", "average")),
        minimum_score=max(
            float(parameters.get("minimum_merge_affinity", 0.05)),
            float(parameters.get("merge_stopping_threshold", 0.0)),
        ),
        maximum_repulsive_conflict=float(parameters.get("maximum_repulsive_conflict", 0.45)),
        maximum_operations=int(parameters.get("maximum_operations", 200000)),
    )
    solver_ms = (perf_counter() - solver_started) * 1000.0
    return _finalize_graph_result(
        "gasp",
        graph,
        node_partition,
        features,
        parameters,
        solver_ms=solver_ms,
        total_started=total_started,
        debug_data={"operations": operations, "operation_limit_reached": limited},
    )


def segment_mutex_watershed(
    image: np.ndarray,
    features: MetalStructuralFeatures,
    parameters: Mapping[str, Any],
) -> StrategySegmentation:
    total_started = perf_counter()
    if not analyze_metal_presence(image).has_metal:
        return _empty_result(image)
    graph = SignedAffinityBuilder().build(features, parameters)
    lifted = None
    lifted_debug: dict[str, np.ndarray] = {}
    if str(parameters.get("mutex_neighborhood_offsets", "local_plus_long_range")) == "local_plus_long_range":
        lifted_u, lifted_v, _attr, lifted_repulsion, debug = _lifted_relations(
            graph,
            features,
            parameters,
            mutex_only=True,
        )
        lifted = (lifted_u, lifted_v, lifted_repulsion)
        lifted_debug["metal_mutex_watershed_long_range_mutex"] = float_map_to_u8(debug)
    solver_started = perf_counter()
    node_partition, operations = _mutex_watershed_partition(graph, parameters, lifted)
    solver_ms = (perf_counter() - solver_started) * 1000.0
    return _finalize_graph_result(
        "mutex_watershed",
        graph,
        node_partition,
        features,
        parameters,
        solver_ms=solver_ms,
        total_started=total_started,
        debug_data={
            "operations": operations,
            "lifted_mutex_edges": 0 if lifted is None else int(lifted[0].size),
            "attractive_neighborhood_offsets": str(parameters.get("attractive_neighborhood_offsets", "local")),
            "edge_ordering": str(parameters.get("edge_ordering", "descending_confidence")),
        },
        extra_debug=lifted_debug,
    )


def _run_multicut(
    image: np.ndarray,
    features: MetalStructuralFeatures,
    parameters: Mapping[str, Any],
    *,
    lifted: bool,
) -> StrategySegmentation:
    total_started = perf_counter()
    if not analyze_metal_presence(image).has_metal:
        return _empty_result(image)
    graph = SignedAffinityBuilder().build(features, parameters)
    edge_u = graph.edge_u
    edge_v = graph.edge_v
    attraction = graph.attraction
    repulsion = graph.repulsion
    lifted_count = 0
    extra_debug: dict[str, np.ndarray] = {}
    if lifted and bool(parameters.get("lifted_edges_enabled", True)):
        lifted_u, lifted_v, lifted_attraction, lifted_repulsion, debug = _lifted_relations(
            graph,
            features,
            parameters,
        )
        lifted_count = int(lifted_u.size)
        edge_u = np.r_[edge_u, lifted_u]
        edge_v = np.r_[edge_v, lifted_v]
        attraction = np.r_[
            attraction,
            lifted_attraction * float(parameters.get("lifted_attraction_weight", 0.45)),
        ]
        repulsion = np.r_[
            repulsion,
            lifted_repulsion * float(parameters.get("lifted_repulsion_weight", 0.75)),
        ]
        extra_debug["metal_lifted_multicut_lifted_relations"] = float_map_to_u8(debug)
    costs = _multicut_costs(attraction, repulsion, parameters)
    initial_partition = None
    if str(parameters.get("initialization", "singletons")) == "positive_components":
        initial_dsu = _DisjointSet(graph.node_count + 1)
        for first, second in zip(edge_u[costs > 0.0], edge_v[costs > 0.0], strict=True):
            initial_dsu.union(int(first), int(second))
        initial_partition = np.array(
            [initial_dsu.find(node) for node in range(graph.node_count + 1)],
            dtype=np.int32,
        )
    solver_started = perf_counter()
    node_partition, operations, time_limited = _agglomerate(
        graph.node_count,
        edge_u,
        edge_v,
        np.maximum(costs, 0.0),
        np.maximum(-costs, 0.0),
        criterion="sum",
        minimum_score=float(parameters.get("convergence_tolerance", 0.0)),
        maximum_repulsive_conflict=None,
        maximum_operations=int(parameters.get("maximum_iterations", 200000)),
        time_limit_seconds=float(parameters.get("time_limit_seconds", 30.0)),
        initial_partition=initial_partition,
    )
    solver_ms = (perf_counter() - solver_started) * 1000.0
    name = "lifted_multicut" if lifted else "multicut"
    return _finalize_graph_result(
        name,
        graph,
        node_partition,
        features,
        parameters,
        solver_ms=solver_ms,
        total_started=total_started,
        debug_data={
            "solver": "greedy_additive",
            "initialization": str(parameters.get("initialization", "singletons")),
            "operations": operations,
            "time_limit_reached": time_limited,
            "lifted_edge_count": lifted_count,
        },
        extra_debug=extra_debug,
    )


def segment_multicut(
    image: np.ndarray,
    features: MetalStructuralFeatures,
    parameters: Mapping[str, Any],
) -> StrategySegmentation:
    return _run_multicut(image, features, parameters, lifted=False)


def segment_lifted_multicut(
    image: np.ndarray,
    features: MetalStructuralFeatures,
    parameters: Mapping[str, Any],
) -> StrategySegmentation:
    return _run_multicut(image, features, parameters, lifted=True)


__all__ = [
    "segment_gasp",
    "segment_lifted_multicut",
    "segment_multicut",
    "segment_mutex_watershed",
]
