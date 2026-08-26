"""Adapter for Jannik Irmai et al.'s native Graph Multi-Separator solvers."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from importlib import import_module
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from ...application.preview_cancellation import raise_if_preview_cancelled
from .features import MetalStructuralFeatures, float_map_to_u8
from .gradient_watershed import _estimate_noise_sigma, analyze_metal_presence
from .material_classifier import classify_partition_material
from .segmentation import otsu_segmentation_mask
from .strategy_contracts import StrategySegmentation, StrategyUnavailableError
from .structural_watershed import _RIBBON_HALF_WIDTHS, _non_maximum_suppress, _remap_offset

_UPSTREAM_COMMIT = "437c651ddf1452452cca4cbc3c0eed2065308486"
_PAIRED_RIM_SAMPLE_STEP = 2
_RIBBON_RECOVERY_MIN_AREA = 20
_RIBBON_RECOVERY_MIN_LENGTH = 12.0
_RIBBON_RECOVERY_MAX_WIDTH = 12.0
_RIBBON_RECOVERY_MIN_ASPECT_RATIO = 2.0


def _paired_rim_core_evidence(features: MetalStructuralFeatures) -> np.ndarray:
    """Detect bright ribbons bounded by opposite-polarity gradients.

    The diagnostic is evaluated on a half-resolution sampling grid. It is used
    only to detect when the conservative seed builder missed a coherent class
    of conductors, not as a replacement pixel mask.
    """

    step = _PAIRED_RIM_SAMPLE_STEP
    intensity = features.denoised[::step, ::step].astype(np.float32)
    orientation = features.orientation[::step, ::step]
    across_x = np.cos(orientation).astype(np.float32)
    across_y = np.sin(orientation).astype(np.float32)
    directional_gradient = (
        features.gradient_x[::step, ::step] * across_x + features.gradient_y[::step, ::step] * across_y
    )
    gradient_scale = max(float(np.percentile(np.abs(directional_gradient), 99.0)), 1e-6)
    contrast_scale = max(16.0, 5.0 * _estimate_noise_sigma(features.denoised))

    height, width = intensity.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    best = np.zeros(intensity.shape, dtype=np.float32)
    for full_half_width in _RIBBON_HALF_WIDTHS:
        half_width = full_half_width / step
        offset_x = across_x * half_width
        offset_y = across_y * half_width
        positive_gradient = _remap_offset(
            directional_gradient,
            grid_x,
            grid_y,
            offset_x,
            offset_y,
        )
        negative_gradient = _remap_offset(
            directional_gradient,
            grid_x,
            grid_y,
            -offset_x,
            -offset_y,
        )
        polarity = np.minimum(
            np.maximum(-positive_gradient / gradient_scale, 0.0),
            np.maximum(negative_gradient / gradient_scale, 0.0),
        )

        outside_offset = (full_half_width + 3.0) / step
        outer_positive = _remap_offset(
            intensity,
            grid_x,
            grid_y,
            across_x * outside_offset,
            across_y * outside_offset,
        )
        outer_negative = _remap_offset(
            intensity,
            grid_x,
            grid_y,
            -across_x * outside_offset,
            -across_y * outside_offset,
        )
        bright_fill = np.clip(
            np.minimum(intensity - outer_positive, intensity - outer_negative) / contrast_scale,
            0.0,
            1.0,
        )
        best = np.maximum(best, polarity * bright_fill)
    return np.clip(best, 0.0, 1.0).astype(np.float32)


def _recover_paired_rim_ribbons(
    image: np.ndarray,
    mask: np.ndarray,
    paired_evidence: np.ndarray,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """Recover locally bright narrow ribbons missed by a global threshold."""

    empty = np.zeros(mask.shape, dtype=bool)
    if not bool(parameters.get("paired_rim_recovery_enabled", True)):
        return mask, empty, 0, 0.0, 0.0

    sensitivity = np.clip(float(parameters.get("gradient_field_sensitivity", 1.0)), 0.25, 4.0)
    evidence_limit = float(np.clip(0.05 / sensitivity, 0.02, 0.2))
    contrast_limit = float(max(24.0, 6.0 * _estimate_noise_sigma(image)) / sensitivity)
    candidates = (paired_evidence >= evidence_limit) & (mask == 0)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        candidates.astype(np.uint8),
        connectivity=8,
    )
    recovered = np.zeros(mask.shape, dtype=bool)
    recovered_components = 0
    height, width = mask.shape
    ring_kernel = np.ones((9, 9), dtype=np.uint8)
    for label_id in range(1, count):
        x, y, component_width, component_height, area = (int(value) for value in stats[label_id])
        if area < _RIBBON_RECOVERY_MIN_AREA:
            continue

        component = labels[y : y + component_height, x : x + component_width] == label_id
        component_y, component_x = np.nonzero(component)
        rectangle = cv2.minAreaRect(
            np.column_stack((component_x, component_y)).astype(np.float32),
        )
        ribbon_width, ribbon_length = sorted(float(value) for value in rectangle[1])
        if (
            ribbon_width > _RIBBON_RECOVERY_MAX_WIDTH
            or ribbon_length < _RIBBON_RECOVERY_MIN_LENGTH
            or ribbon_length / max(ribbon_width, 1.0) < _RIBBON_RECOVERY_MIN_ASPECT_RATIO
        ):
            continue

        padding = ring_kernel.shape[0] // 2
        x_start = max(0, x - padding)
        x_end = min(width, x + component_width + padding)
        y_start = max(0, y - padding)
        y_end = min(height, y + component_height + padding)
        local_component = labels[y_start:y_end, x_start:x_end] == label_id
        ring = cv2.dilate(local_component.astype(np.uint8), ring_kernel) > 0
        ring &= ~local_component
        if not np.any(ring):
            continue
        local_image = image[y_start:y_end, x_start:x_end]
        local_contrast = float(np.median(local_image[local_component]) - np.median(local_image[ring]))
        if local_contrast < contrast_limit:
            continue

        recovered[y_start:y_end, x_start:x_end] |= local_component
        recovered_components += 1

    result = np.where((mask > 0) | recovered, 255, 0).astype(np.uint8)
    return result, recovered, recovered_components, evidence_limit, contrast_limit


def _guarded_otsu_fallback(
    image: np.ndarray,
    features: MetalStructuralFeatures,
    parameters: Mapping[str, Any],
    *,
    total_started: float,
) -> StrategySegmentation | None:
    if not bool(parameters.get("paired_rim_fallback_enabled", True)):
        return None

    analysis_started = perf_counter()
    paired_core = _paired_rim_core_evidence(features)
    core = features.core_evidence[::_PAIRED_RIM_SAMPLE_STEP, ::_PAIRED_RIM_SAMPLE_STEP]
    evidence_limit = float(parameters.get("paired_rim_evidence_threshold", 0.25))
    core_fraction = float(np.mean(core >= evidence_limit))
    missing_core_fraction = float(np.mean((paired_core >= evidence_limit) & (core < evidence_limit)))
    analysis_ms = (perf_counter() - analysis_started) * 1000.0
    if core_fraction < float(
        parameters.get("paired_rim_fallback_min_core_fraction", 0.1)
    ) or missing_core_fraction < float(parameters.get("paired_rim_fallback_fraction", 0.04)):
        return None

    fallback_started = perf_counter()
    mask = otsu_segmentation_mask(image, otsu_offset=0.0, dark_foreground=False)
    paired_preview = cv2.resize(
        paired_core,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    (
        mask,
        recovered_ribbons,
        recovered_ribbon_count,
        recovery_evidence_limit,
        recovery_contrast_limit,
    ) = _recover_paired_rim_ribbons(
        image,
        mask,
        paired_preview,
        parameters,
    )
    _count, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    fallback_ms = (perf_counter() - fallback_started) * 1000.0
    return StrategySegmentation(
        binary_mask=mask,
        instance_labels=labels.astype(np.int32),
        boundary_map=features.boundary_strength,
        confidence_map=(mask.astype(np.float32) / 255.0),
        debug_images={
            "metal_msp_paired_rim_core": float_map_to_u8(paired_preview),
            "metal_msp_otsu_fallback": mask,
            "metal_msp_recovered_ribbons": np.where(recovered_ribbons, 255, 0).astype(np.uint8),
        },
        debug_data={
            "backend": "opencv-otsu",
            "requested_backend": "JannikIrmai/multi-separator",
            "fallback": "missing_core_paired_rims",
            "paired_rim_core_fraction": core_fraction,
            "paired_rim_missing_core_fraction": missing_core_fraction,
            "paired_rim_recovered_components": recovered_ribbon_count,
            "paired_rim_recovery_evidence_limit": recovery_evidence_limit,
            "paired_rim_recovery_contrast_limit": recovery_contrast_limit,
            "tile_count": 0,
            "solver_workers": 0,
        },
        timings_ms={
            "feature_build": float(features.build_time_ms),
            "paired_rim_analysis": analysis_ms,
            "graph_construction": 0.0,
            "solver": 0.0,
            "native_tile_phase": 0.0,
            "material_classification": fallback_ms,
            "total": (perf_counter() - total_started) * 1000.0,
        },
    )


def _native_backend() -> Any:
    try:
        multi_separator = import_module("contour._native.multi_separator")
    except ImportError as exc:
        raise StrategyUnavailableError(
            "Graph Multi-Separator native extension is not built. Reinstall/build the Contour package "
            "with its pybind11 build requirements; no fallback strategy was run."
        ) from exc
    return multi_separator


def _directional_support(evidence: np.ndarray, orientation: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    size = 2 * radius + 1
    candidates: list[np.ndarray] = []
    centers = (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0)
    for angle in centers:
        kernel = np.zeros((size, size), dtype=np.float32)
        center = radius
        for offset in range(-radius, radius + 1):
            x_coord = round(center + np.cos(angle) * offset)
            y_coord = round(center + np.sin(angle) * offset)
            kernel[y_coord, x_coord] = 1.0
        kernel /= max(float(kernel.sum()), 1.0)
        candidates.append(cv2.filter2D(evidence, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT))
    tangent = np.mod(orientation + 0.5 * np.pi, np.pi)
    indices = np.rint(tangent / (np.pi / 4.0)).astype(np.int32) % 4
    return np.take_along_axis(np.stack(candidates), indices[None, :, :], axis=0)[0]


def _separator_probability(
    features: MetalStructuralFeatures,
    parameters: Mapping[str, Any],
) -> np.ndarray:
    continuity = _directional_support(
        features.boundary_strength * (0.4 + 0.6 * features.orientation_coherence),
        features.orientation,
        max(1, int(parameters.get("long_range_radius", 7)) // 2),
    )
    gradient_ridge = (_non_maximum_suppress(features.gradient_magnitude, features.orientation) > 0).astype(np.float32)
    separator_positive = (
        float(parameters.get("separator_unary_weight", 1.0)) * features.boundary_strength
        + float(parameters.get("boundary_separator_weight", 1.2)) * features.orientation_persistence
        + float(parameters.get("gradient_repulsion_weight", 0.9))
        * features.gradient_magnitude
        * np.maximum(gradient_ridge, 0.35)
        + float(parameters.get("orientation_consistency_weight", 0.7))
        * features.orientation_coherence
        * features.boundary_strength
        + float(parameters.get("separator_continuity_weight", 0.8)) * continuity
    )
    region_positive = float(parameters.get("region_unary_weight", 0.55)) * (
        features.core_evidence + features.substrate_evidence
    ) + float(parameters.get("intensity_affinity_weight", 0.6)) * (1.0 - features.local_contrast)
    weight_sum = sum(
        float(parameters.get(key, default))
        for key, default in (
            ("separator_unary_weight", 1.0),
            ("boundary_separator_weight", 1.2),
            ("gradient_repulsion_weight", 0.9),
            ("orientation_consistency_weight", 0.7),
            ("separator_continuity_weight", 0.8),
            ("region_unary_weight", 0.55),
            ("intensity_affinity_weight", 0.6),
        )
    )
    probability = np.clip(
        0.5 + (separator_positive - region_positive) / max(weight_sum, 1e-6),
        1e-5,
        1.0 - 1e-5,
    )
    ridge = _non_maximum_suppress(probability.astype(np.float32), features.orientation) > 0
    off_ridge = (~ridge) & (probability > 0.5)
    if np.any(off_ridge):
        probability = probability.copy()
        probability[off_ridge] = 0.5 * (probability[off_ridge] + 0.5)
    return np.clip(probability, 1e-5, 1.0 - 1e-5)


def _interaction_offsets(parameters: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    offsets: list[tuple[int, int]] = [(1, 0), (0, 1)]
    long_range: list[bool] = [False, False]
    if str(parameters.get("connectivity", "8")) == "8":
        offsets.extend(((1, 1), (1, -1)))
        long_range.extend((False, False))
    if bool(parameters.get("long_range_enabled", True)):
        radius = min(
            int(parameters.get("long_range_radius", 7)),
            int(parameters.get("maximum_interaction_distance", 12)),
        )
        if radius > 1:
            for offset in ((radius, 0), (0, radius), (radius, radius), (radius, -radius)):
                if offset not in offsets:
                    offsets.append(offset)
                    long_range.append(True)
    return np.asarray(offsets, dtype=np.int32), np.asarray(long_range, dtype=bool)


def _shift_for_start(values: np.ndarray, dy: int, dx: int, fill: float) -> np.ndarray:
    height, width = values.shape
    shifted = np.full(values.shape, fill, dtype=np.float64)
    destination_y = slice(max(0, -dy), min(height, height - dy))
    destination_x = slice(max(0, -dx), min(width, width - dx))
    source_y = slice(max(0, dy), min(height, height + dy))
    source_x = slice(max(0, dx), min(width, width + dx))
    shifted[destination_y, destination_x] = values[source_y, source_x]
    return shifted


def _line_minimum(values: np.ndarray, offset: np.ndarray) -> np.ndarray:
    dy, dx = (int(offset[0]), int(offset[1]))
    length = max(abs(dy), abs(dx))
    result = np.full(values.shape, np.inf, dtype=np.float64)
    for step in range(length + 1):
        step_dy = round(step * dy / max(length, 1))
        step_dx = round(step * dx / max(length, 1))
        result = np.minimum(result, _shift_for_start(values, step_dy, step_dx, np.inf))
    result[~np.isfinite(result)] = 0.0
    return result


def _native_costs(
    separator_probability: np.ndarray,
    region_support: np.ndarray,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    threshold = np.clip(float(parameters.get("minimum_separator_confidence", 0.56)), 1e-5, 1.0 - 1e-5)
    vertex_costs = np.log((1.0 - separator_probability) / separator_probability)
    vertex_costs -= np.log((1.0 - threshold) / threshold)
    offsets, long_range = _interaction_offsets(parameters)
    interactions = np.empty((len(offsets), *separator_probability.shape), dtype=np.float64)
    for index, offset in enumerate(offsets):
        line_cost = _line_minimum(vertex_costs, offset)
        if long_range[index]:
            line_cost *= float(parameters.get("long_range_repulsion_weight", 0.65))
            line_cost += float(parameters.get("long_range_attraction_weight", 0.35)) * _line_minimum(
                region_support,
                offset,
            )
        else:
            line_cost *= float(parameters.get("separator_continuity_weight", 0.8))
        interactions[index] = line_cost
    return vertex_costs.astype(np.float64), offsets, interactions


def _run_in_chunks(solver: Any, maximum_iterations: int) -> int:
    chunk_size = 4096
    while True:
        raise_if_preview_cancelled()
        previous = int(solver.num_iter())
        remaining = chunk_size if maximum_iterations <= 0 else min(chunk_size, maximum_iterations - previous)
        if remaining <= 0:
            return previous
        solver.run(remaining)
        current = int(solver.num_iter())
        if current - previous < remaining:
            return current


def _solve_native(
    vertex_costs: np.ndarray,
    offsets: np.ndarray,
    interaction_costs: np.ndarray,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, int, float | None, str]:
    native = _native_backend()
    solver_name = str(parameters.get("solver", "greedy_separator_growing"))
    maximum_iterations = int(parameters.get("maximum_iterations", 0))
    shape = np.asarray(vertex_costs.shape, dtype=np.uintp)
    if solver_name == "greedy_separator_growing":
        stacked = np.concatenate((vertex_costs[None, :, :], interaction_costs), axis=0)
        solver = native.GreedySeparatorGrowing2D(
            shape,
            offsets,
            np.asfortranarray(stacked).ravel(order="F"),
        )
        iterations = _run_in_chunks(solver, maximum_iterations)
        labels = np.asarray(solver.vertex_labels(), dtype=np.int32).reshape(vertex_costs.shape, order="F")
        return labels, iterations, None, solver_name

    solver = native.GreedySeparatorShrinking()
    solver.setup_grid(
        shape,
        offsets.ravel(),
        np.ascontiguousarray(vertex_costs).ravel(),
        np.ascontiguousarray(interaction_costs).ravel(),
    )
    iterations = _run_in_chunks(solver, maximum_iterations)
    labels = np.asarray(solver.vertex_labels(), dtype=np.int32).reshape(vertex_costs.shape)
    return labels, iterations, float(solver.objective()), solver_name


def _solve_native_tiles(
    separator_probability: np.ndarray,
    region_support: np.ndarray,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, int, float | None, str, int, float, float, float]:
    """Apply the same upstream solver to overlapping native-resolution tiles."""
    height, width = separator_probability.shape
    tile_size = int(parameters.get("solver_tile_size", 384))
    overlap = int(parameters.get("solver_tile_overlap", 16))
    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        graph_started = perf_counter()
        vertex_costs, offsets, interaction_costs = _native_costs(
            separator_probability,
            region_support,
            parameters,
        )
        graph_ms = (perf_counter() - graph_started) * 1000.0
        solver_started = perf_counter()
        labels, iterations, objective, solver_name = _solve_native(
            vertex_costs,
            offsets,
            interaction_costs,
            parameters,
        )
        solver_ms = (perf_counter() - solver_started) * 1000.0
        return labels == 0, iterations, objective, solver_name, 1, graph_ms, solver_ms, graph_ms + solver_ms

    overlap = min(overlap, max(0, tile_size // 3))
    separator = np.zeros((height, width), dtype=bool)
    total_iterations = 0
    total_objective = 0.0
    has_objective = True
    graph_ms = 0.0
    solver_ms = 0.0
    tile_count = 0
    solver_name = str(parameters.get("solver", "greedy_separator_growing"))
    tiles: list[tuple[int, int, int, int, int, int, int, int]] = []
    for y_start in range(0, height, tile_size):
        y_end = min(height, y_start + tile_size)
        extended_y_start = max(0, y_start - overlap)
        extended_y_end = min(height, y_end + overlap)
        for x_start in range(0, width, tile_size):
            x_end = min(width, x_start + tile_size)
            extended_x_start = max(0, x_start - overlap)
            extended_x_end = min(width, x_end + overlap)
            tiles.append(
                (
                    y_start,
                    y_end,
                    x_start,
                    x_end,
                    extended_y_start,
                    extended_y_end,
                    extended_x_start,
                    extended_x_end,
                )
            )

    def solve_tile(
        tile: tuple[int, int, int, int, int, int, int, int],
    ) -> tuple[np.ndarray, int, float | None, str, float, float]:
        raise_if_preview_cancelled()
        (
            _y_start,
            _y_end,
            _x_start,
            _x_end,
            extended_y_start,
            extended_y_end,
            extended_x_start,
            extended_x_end,
        ) = tile
        tile_probability = separator_probability[
            extended_y_start:extended_y_end,
            extended_x_start:extended_x_end,
        ]
        tile_region_support = region_support[
            extended_y_start:extended_y_end,
            extended_x_start:extended_x_end,
        ]
        graph_started = perf_counter()
        vertex_costs, offsets, interaction_costs = _native_costs(
            tile_probability,
            tile_region_support,
            parameters,
        )
        tile_graph_ms = (perf_counter() - graph_started) * 1000.0
        solver_started = perf_counter()
        tile_labels, tile_iterations, tile_objective, tile_solver_name = _solve_native(
            vertex_costs,
            offsets,
            interaction_costs,
            parameters,
        )
        tile_solver_ms = (perf_counter() - solver_started) * 1000.0
        return tile_labels, tile_iterations, tile_objective, tile_solver_name, tile_graph_ms, tile_solver_ms

    tile_phase_started = perf_counter()
    worker_count = min(max(1, int(parameters.get("solver_workers", 1))), len(tiles))
    if worker_count == 1:
        tile_results = [solve_tile(tile) for tile in tiles]
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="contour-msp") as executor:
            futures = [executor.submit(copy_context().run, solve_tile, tile) for tile in tiles]
            tile_results = [future.result() for future in futures]
    tile_phase_ms = (perf_counter() - tile_phase_started) * 1000.0

    for tile, tile_result in zip(tiles, tile_results, strict=True):
        (
            y_start,
            y_end,
            x_start,
            x_end,
            extended_y_start,
            _extended_y_end,
            extended_x_start,
            _extended_x_end,
        ) = tile
        tile_labels, iterations, objective, solver_name, tile_graph_ms, tile_solver_ms = tile_result
        core_y_start = y_start - extended_y_start
        core_y_end = core_y_start + (y_end - y_start)
        core_x_start = x_start - extended_x_start
        core_x_end = core_x_start + (x_end - x_start)
        separator[y_start:y_end, x_start:x_end] = (
            tile_labels[
                core_y_start:core_y_end,
                core_x_start:core_x_end,
            ]
            == 0
        )
        total_iterations += iterations
        if objective is None:
            has_objective = False
        else:
            total_objective += objective
        graph_ms += tile_graph_ms
        solver_ms += tile_solver_ms
        tile_count += 1
    return (
        separator,
        total_iterations,
        total_objective if has_objective else None,
        solver_name,
        tile_count,
        graph_ms,
        solver_ms,
        tile_phase_ms,
    )


def _assign_peeled_separator_pixels(
    labels: np.ndarray,
    native_separator: np.ndarray,
    thin_separator: np.ndarray,
) -> np.ndarray:
    """Give thick-separator pixels to the region on each side of the 1 px skeleton."""

    owners = labels.copy()
    owners[thin_separator] = 0
    peeled = native_separator & ~thin_separator
    if not np.any(peeled):
        return owners

    maximum_steps = max(1, int(native_separator.shape[0] + native_separator.shape[1]))
    neighbour_offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for _step in range(maximum_steps):
        candidate_min = np.zeros(owners.shape, dtype=np.int32)
        for dy, dx in neighbour_offsets:
            candidate = _shift_labels(owners, dy, dx)
            candidate[candidate < 1] = 0
            replace = (candidate > 0) & ((candidate_min == 0) | (candidate < candidate_min))
            candidate_min[replace] = candidate[replace]
        target = peeled & (owners == 0) & (candidate_min > 0)
        if not np.any(target):
            break
        owners[target] = candidate_min[target]
    owners[thin_separator] = 0
    return owners


def _thin_separator_to_one_pixel(separator: np.ndarray) -> np.ndarray:
    """Reduce separator bands to an 8-connected 1 px skeleton (Zhang–Suen)."""

    binary = (separator > 0).astype(np.uint8)
    if binary.size == 0 or not np.any(binary):
        return np.zeros(separator.shape, dtype=bool)

    image = np.pad(binary, 1, mode="constant")
    maximum_iterations = max(8, 2 * (int(separator.shape[0]) + int(separator.shape[1])))
    for _iteration in range(maximum_iterations):
        changed = False
        for step in (1, 2):
            p2 = image[:-2, 1:-1]
            p3 = image[:-2, 2:]
            p4 = image[1:-1, 2:]
            p5 = image[2:, 2:]
            p6 = image[2:, 1:-1]
            p7 = image[2:, :-2]
            p8 = image[1:-1, :-2]
            p9 = image[:-2, :-2]
            p1 = image[1:-1, 1:-1]
            neighbour_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )
            if step == 1:
                extra = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                extra = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (p1 == 1) & (neighbour_count >= 2) & (neighbour_count <= 6) & (transitions == 1) & extra
            if np.any(remove):
                image[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            break
    return image[1:-1, 1:-1].astype(bool)


def _remove_short_separators(separator: np.ndarray, minimum_length: int) -> np.ndarray:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(separator.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, dtype=bool)
    if count > 1:
        keep[1:] = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) >= max(
            1,
            int(minimum_length),
        )
    return keep[labels]


def _absorb_small_regions(labels: np.ndarray, minimum_area: int) -> np.ndarray:
    if minimum_area <= 1 or int(labels.max()) <= 0:
        return labels
    result = labels.copy()
    areas = np.bincount(result.ravel(), minlength=int(result.max()) + 1)
    for label_id in np.flatnonzero((areas > 0) & (areas < int(minimum_area))):
        if label_id == 0:
            continue
        region = result == int(label_id)
        ring = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
        neighbours = result[ring & ~region]
        neighbours = neighbours[neighbours > 0]
        if neighbours.size:
            result[region] = int(np.bincount(neighbours).argmax())
    unique = np.unique(result)
    unique = unique[unique > 0]
    remap = np.zeros(int(result.max()) + 1, dtype=np.int32)
    remap[unique] = np.arange(1, unique.size + 1, dtype=np.int32)
    return remap[result]


_NEIGHBOUR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _shift_labels(values: np.ndarray, dy: int, dx: int) -> np.ndarray:
    height, width = values.shape
    shifted = np.zeros(values.shape, dtype=np.int32)
    destination_y = slice(max(0, -dy), min(height, height - dy))
    destination_x = slice(max(0, -dx), min(width, width - dx))
    source_y = slice(max(0, dy), min(height, height + dy))
    source_x = slice(max(0, dx), min(width, width + dx))
    shifted[destination_y, destination_x] = values[source_y, source_x]
    return shifted


def _project_material_separators(
    labels: np.ndarray,
    separator: np.ndarray,
    separator_probability: np.ndarray,
    features: MetalStructuralFeatures,
    region_confidence: np.ndarray,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Restore separator nodes that have strong conductor-interior evidence.

    The upstream graph solver models separators as their own nodes. They are not
    intrinsically background pixels, so converting all of them to zero creates
    holes and open contours inside bright conductors. Only separator pixels with
    strong core evidence are projected. A weak separator may also join two
    already-classified metal regions. Substrate-like separator pixels remain
    background, while confident separators keep their distinct region labels.
    """
    if not bool(parameters.get("separator_projection_enabled", True)):
        return labels, np.zeros(labels.shape, dtype=bool), 0

    maximum_label = int(labels.max())
    radius = max(0, int(parameters.get("separator_projection_radius", 1)))
    if maximum_label <= 0 or radius == 0:
        return labels, np.zeros(labels.shape, dtype=bool), 0

    metal_regions = np.zeros(maximum_label + 1, dtype=bool)
    available = min(metal_regions.size, region_confidence.size)
    metal_regions[:available] = region_confidence[:available] >= float(parameters.get("minimum_metal_confidence", 0.52))
    metal_regions[0] = False

    minimum_core = float(parameters.get("separator_projection_min_core_evidence", 0.25))
    core_margin = float(parameters.get("separator_projection_core_margin", 0.25))
    fillable = (
        separator
        & (features.core_evidence >= minimum_core)
        & (features.core_evidence >= features.substrate_evidence + core_margin)
    )
    if not np.any(fillable) or not np.any(metal_regions):
        return labels, np.zeros(labels.shape, dtype=bool), 0

    owners = labels.copy()
    parent = np.arange(maximum_label + 1, dtype=np.int32)
    merged_region_pairs = 0

    def find(label_id: int) -> int:
        while int(parent[label_id]) != label_id:
            parent[label_id] = parent[int(parent[label_id])]
            label_id = int(parent[label_id])
        return label_id

    def union(first: int, second: int) -> None:
        nonlocal merged_region_pairs
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        lower, upper = sorted((first_root, second_root))
        parent[upper] = lower
        merged_region_pairs += 1

    weak_separator = separator_probability <= float(parameters.get("metal_merge_max_separator_confidence", 0.7))
    for _step in range(radius):
        candidate_min = np.zeros(labels.shape, dtype=np.int32)
        for dy, dx in _NEIGHBOUR_OFFSETS:
            candidate = _shift_labels(owners, dy, dx)
            candidate[~metal_regions[candidate]] = 0
            replace = (candidate > 0) & ((candidate_min == 0) | (candidate < candidate_min))
            candidate_min[replace] = candidate[replace]

        target = fillable & (owners == 0) & (candidate_min > 0)
        if not np.any(target):
            break

        merge_target = target & weak_separator
        if np.any(merge_target):
            pair_base = maximum_label + 1
            for dy, dx in _NEIGHBOUR_OFFSETS:
                candidate = _shift_labels(owners, dy, dx)
                candidate[~metal_regions[candidate]] = 0
                paired = merge_target & (candidate > 0) & (candidate != candidate_min)
                if not np.any(paired):
                    continue
                first = np.minimum(candidate_min[paired], candidate[paired]).astype(np.int64)
                second = np.maximum(candidate_min[paired], candidate[paired]).astype(np.int64)
                encoded = np.unique(first * pair_base + second)
                for pair in encoded:
                    union(int(pair // pair_base), int(pair % pair_base))

        owners[target] = candidate_min[target]

    for label_id in range(1, maximum_label + 1):
        parent[label_id] = find(label_id)
    positive = owners > 0
    owners[positive] = parent[owners[positive]]
    unique = np.unique(owners[positive])
    remap = np.zeros(maximum_label + 1, dtype=np.int32)
    remap[unique] = np.arange(1, unique.size + 1, dtype=np.int32)
    owners[positive] = remap[owners[positive]]
    projected = separator & (owners > 0)
    return owners, projected, merged_region_pairs


def segment_graph_multi_separator(
    image: np.ndarray,
    features: MetalStructuralFeatures,
    parameters: Mapping[str, Any],
) -> StrategySegmentation:
    total_started = perf_counter()
    if not analyze_metal_presence(image).has_metal:
        empty = np.zeros(image.shape, dtype=np.uint8)
        return StrategySegmentation(
            empty,
            np.zeros(image.shape, dtype=np.int32),
            empty.astype(np.float32),
            empty.astype(np.float32),
        )

    fallback = _guarded_otsu_fallback(
        image,
        features,
        parameters,
        total_started=total_started,
    )
    if fallback is not None:
        return fallback

    separator_probability = _separator_probability(features, parameters)
    region_support = np.clip(
        0.5 * (features.core_evidence + features.substrate_evidence) + 0.5 * (1.0 - features.local_contrast),
        0.0,
        1.0,
    )
    (
        native_separator,
        iterations,
        objective,
        solver_name,
        tile_count,
        graph_ms,
        solver_ms,
        tile_phase_ms,
    ) = _solve_native_tiles(separator_probability, region_support, parameters)
    native_separator = _remove_short_separators(
        native_separator,
        int(parameters.get("minimum_separator_length", 4)),
    )
    separator = _thin_separator_to_one_pixel(native_separator)
    labels = cv2.connectedComponents((~native_separator).astype(np.uint8), connectivity=4)[1].astype(np.int32)
    labels[native_separator] = 0
    labels = _assign_peeled_separator_pixels(labels, native_separator, separator)
    labels = _absorb_small_regions(labels, int(parameters.get("minimum_region_area", 12)))

    classify_started = perf_counter()
    initial_material = classify_partition_material(labels, features, parameters)
    labels, projected_separators, merged_region_pairs = _project_material_separators(
        labels,
        separator,
        separator_probability,
        features,
        initial_material.region_confidence,
        parameters,
    )
    material = (
        classify_partition_material(labels, features, parameters) if np.any(projected_separators) else initial_material
    )
    classification_ms = (perf_counter() - classify_started) * 1000.0
    return StrategySegmentation(
        binary_mask=material.mask,
        instance_labels=material.instance_labels,
        boundary_map=separator_probability,
        confidence_map=material.confidence_map,
        debug_images={
            "metal_msp_separator_cost": float_map_to_u8(separator_probability),
            "metal_msp_selected_separators": np.where(separator, 255, 0).astype(np.uint8),
            "metal_msp_projected_separators": np.where(projected_separators, 255, 0).astype(np.uint8),
            "metal_msp_remaining_separators": np.where(separator & ~projected_separators, 255, 0).astype(np.uint8),
            "metal_msp_regions": ((labels.astype(np.uint64) * 37) % 251).astype(np.uint8),
            **material.debug_images,
        },
        debug_data={
            "backend": "JannikIrmai/multi-separator",
            "upstream_commit": _UPSTREAM_COMMIT,
            "solver": solver_name,
            "iterations": iterations,
            "objective": objective,
            "separator_pixels": int(np.count_nonzero(separator)),
            "projected_separator_pixels": int(np.count_nonzero(projected_separators)),
            "remaining_separator_pixels": int(np.count_nonzero(separator & ~projected_separators)),
            "merged_metal_region_pairs": merged_region_pairs,
            "region_count": int(labels.max()),
            "tile_count": tile_count,
            "solver_tile_size": int(parameters.get("solver_tile_size", 384)),
            "solver_tile_overlap": int(parameters.get("solver_tile_overlap", 16)),
            "solver_workers": int(parameters.get("solver_workers", 1)),
            "native_tile_phase_ms": tile_phase_ms,
        },
        timings_ms={
            "feature_build": float(features.build_time_ms),
            "graph_construction": graph_ms,
            "solver": solver_ms,
            "native_tile_phase": tile_phase_ms,
            "material_classification": classification_ms,
            "total": (perf_counter() - total_started) * 1000.0,
        },
    )


__all__ = ["segment_graph_multi_separator"]
