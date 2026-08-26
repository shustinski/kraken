"""Adapter from Contour SEM features to the Berkeley OWT-UCM algorithm."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .bsr_owt_ucm import build_bsr_ucm, cut_bsr_hierarchy
from .features import MetalStructuralFeatures, float_map_to_u8
from .gradient_watershed import analyze_metal_presence
from .material_classifier import classify_partition_material
from .strategy_contracts import StrategySegmentation

_BSR_ORIENTATION_BINS = 8


@dataclass(frozen=True, slots=True)
class OrientedWatershedPartition:
    labels: np.ndarray
    boundary_strength: np.ndarray
    dominant_orientation: np.ndarray
    oriented_channels: np.ndarray
    oriented_channels_preview: np.ndarray


def _line_kernel(angle: float, sigma: float) -> np.ndarray:
    radius = max(1, round(2.5 * sigma))
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center = float(radius)
    for offset in range(-radius, radius + 1):
        x_coord = round(center + np.cos(angle) * offset)
        y_coord = round(center + np.sin(angle) * offset)
        if 0 <= x_coord < size and 0 <= y_coord < size:
            kernel[y_coord, x_coord] += np.exp(-0.5 * (offset / max(sigma, 0.1)) ** 2)
    return kernel / max(float(kernel.sum()), 1e-6)


def build_oriented_boundary_channels(
    features: MetalStructuralFeatures,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build BSR's eight tangent-oriented probability-of-boundary channels."""
    sigma = float(parameters.get("orientation_smoothing_sigma", 2.0))
    source_name = str(parameters.get("contour_source", "combined"))
    normal = np.mod(features.orientation, np.pi).astype(np.float32)
    base = features.gradient_magnitude
    if source_name == "structural_gradient":
        base = features.boundary_strength
    elif source_name == "combined":
        base = np.clip(0.5 * features.gradient_magnitude + 0.5 * features.boundary_strength, 0.0, 1.0)

    channels = np.empty((_BSR_ORIENTATION_BINS, *base.shape), dtype=np.float32)
    for index in range(_BSR_ORIENTATION_BINS):
        tangent = np.pi * float(index) / float(_BSR_ORIENTATION_BINS)
        expected_normal = np.mod(tangent - 0.5 * np.pi, np.pi)
        angular_match = np.maximum(0.0, np.cos(2.0 * (normal - expected_normal))).astype(np.float32)
        channel = base * angular_match * (0.35 + 0.65 * features.orientation_coherence)
        channels[index] = cv2.filter2D(
            channel,
            cv2.CV_32F,
            _line_kernel(tangent, sigma),
            borderType=cv2.BORDER_REFLECT,
        )
    channels *= 1.0 + float(parameters.get("contour_continuity_weight", 0.65)) * features.orientation_persistence
    peak = float(np.percentile(channels, 99.5)) if channels.size else 0.0
    if peak > 1e-6:
        channels = np.clip(channels / peak, 0.0, 1.0)
    dominant = np.argmax(channels, axis=0).astype(np.uint8)
    strength = np.max(channels, axis=0).astype(np.float32)
    return channels, strength, dominant


def _filter_marker_components(markers: np.ndarray, minimum_area: int) -> np.ndarray:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(markers.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, dtype=bool)
    if count > 1:
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(1, int(minimum_area))
    filtered = np.where(keep[labels], labels, 0).astype(np.int32)
    unique = np.unique(filtered)
    unique = unique[unique > 0]
    remap = np.zeros(count, dtype=np.int32)
    remap[unique] = np.arange(1, unique.size + 1, dtype=np.int32)
    return remap[filtered]


def oriented_watershed_partition(
    features: MetalStructuralFeatures,
    parameters: Mapping[str, Any],
) -> OrientedWatershedPartition:
    """Create BSR's finest watershed partition from oriented boundaries."""
    channels, strength, dominant = build_oriented_boundary_channels(features, parameters)
    contour_sigma = float(parameters.get("contour_smoothing_sigma", 1.0))
    surface = cv2.GaussianBlur(strength, (0, 0), max(0.1, contour_sigma))
    minimum_strength = float(parameters.get("minimum_contour_strength", 0.12))
    surface = np.where(surface >= minimum_strength, surface, 0.0).astype(np.float32)
    suppression = float(parameters.get("watershed_minima_suppression", 0.06))
    minima_surface = cv2.GaussianBlur(surface, (0, 0), max(0.1, 0.5 + 4.0 * suppression))
    eroded = cv2.erode(minima_surface, np.ones((3, 3), dtype=np.uint8))
    minima = cv2.erode(
        (minima_surface <= eroded + 1e-6).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
    )
    markers = _filter_marker_components(minima, int(parameters.get("minimum_initial_basin_area", 6)))
    if int(markers.max()) < 2:
        spacing = max(3, round(np.sqrt(float(parameters.get("minimum_initial_basin_area", 6))) * 3.0))
        marker_points = np.zeros(surface.shape, dtype=np.uint8)
        marker_points[spacing // 2 :: spacing, spacing // 2 :: spacing] = 1
        markers = cv2.connectedComponents(marker_points, connectivity=8)[1].astype(np.int32)

    watershed_labels = markers.copy()
    cv2.watershed(cv2.cvtColor(float_map_to_u8(surface), cv2.COLOR_GRAY2BGR), watershed_labels)
    labels = np.where(watershed_labels > 0, watershed_labels, 0).astype(np.int32)
    for _iteration in range(2):
        missing = labels == 0
        if not np.any(missing):
            break
        grown = cv2.dilate(labels.astype(np.float32), np.ones((3, 3), dtype=np.uint8)).astype(np.int32)
        labels[missing] = grown[missing]

    channel_strength = float_map_to_u8(strength)
    orientation_hsv = np.zeros((*surface.shape, 3), dtype=np.uint8)
    orientation_hsv[..., 0] = np.rint(dominant.astype(np.float32) * 179.0 / _BSR_ORIENTATION_BINS).astype(np.uint8)
    orientation_hsv[..., 1] = np.where(channel_strength > 0, 255, 0).astype(np.uint8)
    orientation_hsv[..., 2] = channel_strength
    return OrientedWatershedPartition(
        labels=labels,
        boundary_strength=surface,
        dominant_orientation=dominant,
        oriented_channels=channels,
        oriented_channels_preview=cv2.cvtColor(orientation_hsv, cv2.COLOR_HSV2BGR),
    )


def _absorb_small_regions(labels: np.ndarray, minimum_area: int) -> np.ndarray:
    if minimum_area <= 1 or int(labels.max()) <= 0:
        return labels
    result = labels.copy()
    areas = np.bincount(result.ravel(), minlength=int(result.max()) + 1)
    for label_id in np.flatnonzero((areas > 0) & (areas < minimum_area)):
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


def build_ucm_hierarchy(
    partition: OrientedWatershedPartition,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    """Build and cut the original BSR dynamic mean-boundary hierarchy."""
    result = build_bsr_ucm(partition.labels, partition.oriented_channels)
    selected = cut_bsr_hierarchy(
        partition.labels,
        result.hierarchy,
        float(parameters.get("hierarchy_level", 0.2)),
    )
    selected = _absorb_small_regions(selected, int(parameters.get("minimum_output_region_area", 20)))
    return selected, result.ucm, list(result.hierarchy)


def _labels_preview(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0 or int(labels.max()) <= 0:
        return np.zeros(labels.shape, dtype=np.uint8)
    return ((labels.astype(np.uint64) * 37) % 251).astype(np.uint8)


def segment_owt_ucm(
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
    partition_started = perf_counter()
    initial = oriented_watershed_partition(features, parameters)
    partition_ms = (perf_counter() - partition_started) * 1000.0
    hierarchy_started = perf_counter()
    selected, ucm, hierarchy = build_ucm_hierarchy(initial, parameters)
    hierarchy_ms = (perf_counter() - hierarchy_started) * 1000.0
    classify_started = perf_counter()
    material = classify_partition_material(selected, features, parameters)
    classification_ms = (perf_counter() - classify_started) * 1000.0
    return StrategySegmentation(
        binary_mask=material.mask,
        instance_labels=material.instance_labels,
        boundary_map=ucm,
        confidence_map=material.confidence_map,
        debug_images={
            "metal_owt_oriented_boundaries": initial.oriented_channels_preview,
            "metal_owt_initial_watershed": _labels_preview(initial.labels),
            "metal_owt_ucm": float_map_to_u8(ucm),
            "metal_owt_selected_hierarchy": _labels_preview(selected),
            **material.debug_images,
        },
        debug_data={
            "backend": "Berkeley OWT-UCM (AGPL-3.0-or-later)",
            "hierarchy": hierarchy,
            "initial_region_count": int(initial.labels.max()),
            "selected_region_count": int(selected.max()),
            "hierarchy_level": float(parameters.get("hierarchy_level", 0.2)),
        },
        timings_ms={
            "feature_build": float(features.build_time_ms),
            "graph_construction": partition_ms,
            "solver": hierarchy_ms,
            "material_classification": classification_ms,
            "total": (perf_counter() - total_started) * 1000.0,
        },
    )


__all__ = [
    "OrientedWatershedPartition",
    "build_oriented_boundary_channels",
    "build_ucm_hierarchy",
    "oriented_watershed_partition",
    "segment_owt_ucm",
]
