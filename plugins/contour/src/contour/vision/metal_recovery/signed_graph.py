"""Shared signed-affinity graph construction for GASP, MWS and multicuts."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from threading import RLock
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .features import MetalStructuralFeatures
from .strategy_contracts import StrategyConfigurationError

_GRAPH_PARAMETER_KEYS = (
    "graph_domain",
    "connectivity",
    "atomic_segmentation_method",
    "atomic_region_scale",
    "minimum_atomic_region_area",
    "intensity_attraction_weight",
    "local_contrast_attraction_weight",
    "orientation_attraction_weight",
    "core_attraction_weight",
    "boundary_repulsion_weight",
    "rim_repulsion_weight",
    "oriented_boundary_repulsion_weight",
    "affinity_normalization",
    "affinity_temperature",
    "minimum_attractive_confidence",
    "minimum_repulsive_confidence",
)
# UI processing is frame-major, so one graph is enough for all four solvers.
# Keeping older full-resolution graphs would retain their feature tensors and
# can consume gigabytes on large SEM frames without improving interactivity.
_SIGNED_GRAPH_CACHE_MAX = 1
_SIGNED_GRAPH_CACHE: OrderedDict[
    tuple[int, tuple[tuple[str, object], ...]],
    tuple[MetalStructuralFeatures, SignedAffinityGraph],
] = OrderedDict()
_SIGNED_GRAPH_CACHE_LOCK = RLock()


def _graph_cache_key(
    features: MetalStructuralFeatures,
    parameters: Mapping[str, Any],
) -> tuple[int, tuple[tuple[str, object], ...]]:
    return (
        id(features),
        tuple((key, parameters.get(key)) for key in _GRAPH_PARAMETER_KEYS),
    )


@dataclass(frozen=True, slots=True)
class SignedAffinityGraph:
    pixel_labels: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_diagonal_only: np.ndarray
    attraction: np.ndarray
    repulsion: np.ndarray
    node_area: np.ndarray
    node_intensity: np.ndarray
    node_contrast: np.ndarray
    node_orientation: np.ndarray
    node_coherence: np.ndarray
    node_core: np.ndarray
    node_substrate: np.ndarray
    node_centroids: np.ndarray
    attraction_map: np.ndarray
    repulsion_map: np.ndarray
    build_time_ms: float

    @property
    def node_count(self) -> int:
        return int(self.node_area.size - 1)

    @property
    def edge_count(self) -> int:
        return int(self.edge_u.size)


def _region_mean(values: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    area = np.bincount(labels.ravel(), minlength=count).astype(np.float64)
    sums = np.bincount(labels.ravel(), weights=values.ravel(), minlength=count)
    return np.divide(sums, np.maximum(area, 1.0)).astype(np.float32)


def _absorb_small_regions(labels: np.ndarray, minimum_area: int) -> np.ndarray:
    result = labels.copy()
    if minimum_area <= 1:
        return result
    areas = np.bincount(result.ravel(), minlength=int(result.max()) + 1)
    for label_id in np.flatnonzero((areas > 0) & (areas < minimum_area)):
        if label_id <= 0:
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


def _atomic_partition(
    features: MetalStructuralFeatures,
    parameters: Mapping[str, Any],
) -> np.ndarray:
    height, width = features.gray.shape
    scale = max(2, int(parameters.get("atomic_region_scale", 10)))
    method = str(parameters.get("atomic_segmentation_method", "oriented_watershed"))
    y_grid = np.arange(height, dtype=np.int32)[:, None] // scale
    x_grid = np.arange(width, dtype=np.int32)[None, :] // scale
    columns = (width + scale - 1) // scale
    grid_labels = (y_grid * columns + x_grid + 1).astype(np.int32)
    if method == "regular_grid":
        labels = grid_labels
    else:
        # Full-resolution seeded watershed on continuous orientation-aware
        # boundary evidence.  Only marker density changes; the image is never
        # resized and narrow gaps remain represented in the cost surface.
        oriented_surface = np.clip(
            features.boundary_strength
            * (0.5 + 0.5 * features.orientation_coherence)
            * (0.6 + 0.4 * features.orientation_persistence),
            0.0,
            1.0,
        )
        markers = np.zeros((height, width), dtype=np.int32)
        center = scale // 2
        markers[center::scale, center::scale] = grid_labels[center::scale, center::scale]
        surface_u8 = np.clip(oriented_surface * 255.0, 0.0, 255.0).astype(np.uint8)
        cv2.watershed(cv2.cvtColor(surface_u8, cv2.COLOR_GRAY2BGR), markers)
        labels = np.where(markers > 0, markers, 0).astype(np.int32)
        for _iteration in range(2):
            missing = labels == 0
            if not np.any(missing):
                break
            grown = cv2.dilate(labels.astype(np.float32), np.ones((3, 3), dtype=np.uint8)).astype(np.int32)
            labels[missing] = grown[missing]
        unique = np.unique(labels)
        unique = unique[unique > 0]
        remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
        remap[unique] = np.arange(1, unique.size + 1, dtype=np.int32)
        labels = remap[labels]
    return _absorb_small_regions(labels, int(parameters.get("minimum_atomic_region_area", 6)))


def _pixel_partition(features: MetalStructuralFeatures) -> np.ndarray:
    pixel_count = int(features.gray.size)
    if pixel_count > 1_000_000:
        raise StrategyConfigurationError(
            "Pixel graph would exceed the deterministic in-process safety limit "
            f"({pixel_count:,} nodes). Select Atomic regions; the full-resolution image is still preserved."
        )
    return np.arange(1, pixel_count + 1, dtype=np.int32).reshape(features.gray.shape)


def _edge_samples(
    labels: np.ndarray,
    features: MetalStructuralFeatures,
    connectivity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pairs: list[np.ndarray] = []
    boundaries: list[np.ndarray] = []
    rims: list[np.ndarray] = []
    persistences: list[np.ndarray] = []
    diagonal_flags: list[np.ndarray] = []
    slices = [
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ]
    if connectivity == 8:
        slices.extend(
            [
                ((slice(None, -1), slice(None, -1)), (slice(1, None), slice(1, None))),
                ((slice(None, -1), slice(1, None)), (slice(1, None), slice(None, -1))),
            ]
        )
    for offset_index, (first_slice, second_slice) in enumerate(slices):
        first = labels[first_slice]
        second = labels[second_slice]
        valid = (first > 0) & (second > 0) & (first != second)
        if not np.any(valid):
            continue
        low = np.minimum(first[valid], second[valid]).astype(np.int64)
        high = np.maximum(first[valid], second[valid]).astype(np.int64)
        pairs.append(np.column_stack((low, high)))
        boundaries.append(
            np.maximum(features.boundary_strength[first_slice], features.boundary_strength[second_slice])[valid]
        )
        rims.append(np.maximum(features.rim_evidence[first_slice], features.rim_evidence[second_slice])[valid])
        persistences.append(
            np.maximum(
                features.orientation_persistence[first_slice],
                features.orientation_persistence[second_slice],
            )[valid]
        )
        diagonal_flags.append(np.full(low.shape, offset_index >= 2, dtype=np.bool_))
    if not pairs:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.bool_),
        )
    all_pairs = np.concatenate(pairs)
    boundary = np.concatenate(boundaries).astype(np.float32)
    rim = np.concatenate(rims).astype(np.float32)
    persistence = np.concatenate(persistences).astype(np.float32)
    diagonal = np.concatenate(diagonal_flags)
    order = np.lexsort((all_pairs[:, 1], all_pairs[:, 0]))
    return all_pairs[order], boundary[order], rim[order], persistence[order], diagonal[order]


def _aggregate_edges(
    pairs: np.ndarray,
    boundary: np.ndarray,
    rim: np.ndarray,
    persistence: np.ndarray,
    diagonal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if pairs.size == 0:
        return pairs, boundary, rim, persistence, diagonal
    starts = np.r_[0, np.flatnonzero(np.any(pairs[1:] != pairs[:-1], axis=1)) + 1]
    counts = np.diff(np.r_[starts, len(pairs)]).astype(np.float32)
    unique = pairs[starts]
    boundary_mean = np.add.reduceat(boundary, starts) / counts
    rim_mean = np.add.reduceat(rim, starts) / counts
    persistence_mean = np.add.reduceat(persistence, starts) / counts
    diagonal_only = np.minimum.reduceat(diagonal.astype(np.uint8), starts).astype(np.bool_)
    return unique, boundary_mean, rim_mean, persistence_mean, diagonal_only


def _edge_debug_map(
    labels: np.ndarray,
    edges: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    if edges.size == 0:
        return np.zeros(labels.shape, dtype=np.float32)
    result = np.zeros(labels.shape, dtype=np.float32)
    maximum_label = int(labels.max())
    keys = edges[:, 0].astype(np.int64) * (maximum_label + 1) + edges[:, 1].astype(np.int64)

    def _assign(first: np.ndarray, second: np.ndarray, target: np.ndarray) -> None:
        valid = (first > 0) & (second > 0) & (first != second)
        if not np.any(valid):
            return
        low = np.minimum(first[valid], second[valid]).astype(np.int64)
        high = np.maximum(first[valid], second[valid]).astype(np.int64)
        sample_keys = low * (maximum_label + 1) + high
        positions = np.searchsorted(keys, sample_keys)
        matched = (positions < keys.size) & (keys[np.minimum(positions, keys.size - 1)] == sample_keys)
        samples = np.zeros(sample_keys.shape, dtype=np.float32)
        samples[matched] = values[positions[matched]]
        target[valid] = samples

    for first_slice, second_slice in (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        first = labels[first_slice]
        second = labels[second_slice]
        target = result[first_slice]
        assigned = np.zeros_like(target)
        _assign(first, second, assigned)
        target[:] = np.maximum(target, assigned)
    return result


class SignedAffinityBuilder:
    """Build one normalized signed graph reused unchanged by all graph solvers."""

    def build(
        self,
        features: MetalStructuralFeatures,
        parameters: Mapping[str, Any],
    ) -> SignedAffinityGraph:
        cache_key = _graph_cache_key(features, parameters)
        with _SIGNED_GRAPH_CACHE_LOCK:
            cached = _SIGNED_GRAPH_CACHE.get(cache_key)
            if cached is not None and cached[0] is features:
                _SIGNED_GRAPH_CACHE.move_to_end(cache_key)
                return replace(cached[1], build_time_ms=0.0)

        started = perf_counter()
        domain = str(parameters.get("graph_domain", "atomic_regions"))
        labels = _pixel_partition(features) if domain == "pixels" else _atomic_partition(features, parameters)
        count = int(labels.max()) + 1
        node_area = np.bincount(labels.ravel(), minlength=count).astype(np.int64)
        gray = features.denoised.astype(np.float32) / 255.0
        intensity = _region_mean(gray, labels, count)
        contrast = _region_mean(features.local_contrast, labels, count)
        coherence = _region_mean(features.orientation_coherence, labels, count)
        core = _region_mean(features.core_evidence, labels, count)
        substrate = _region_mean(features.substrate_evidence, labels, count)
        cosine = _region_mean(np.cos(2.0 * features.orientation), labels, count)
        sine = _region_mean(np.sin(2.0 * features.orientation), labels, count)
        orientation = 0.5 * np.arctan2(sine, cosine).astype(np.float32)
        coordinates = np.indices(labels.shape, dtype=np.float32)
        yy: np.ndarray = coordinates[0]
        xx: np.ndarray = coordinates[1]
        centroid_x = _region_mean(xx, labels, count)
        centroid_y = _region_mean(yy, labels, count)
        centroids = np.column_stack((centroid_x, centroid_y)).astype(np.float32)

        pairs, boundary, rim, persistence, diagonal_only = _edge_samples(
            labels,
            features,
            8 if str(parameters.get("connectivity", "4")) == "8" else 4,
        )
        pairs, boundary, rim, persistence, diagonal_only = _aggregate_edges(
            pairs,
            boundary,
            rim,
            persistence,
            diagonal_only,
        )
        edge_u = pairs[:, 0].astype(np.int32) if pairs.size else np.empty(0, dtype=np.int32)
        edge_v = pairs[:, 1].astype(np.int32) if pairs.size else np.empty(0, dtype=np.int32)
        temperature = max(0.05, float(parameters.get("affinity_temperature", 1.0)))
        intensity_similarity = np.exp(-np.abs(intensity[edge_u] - intensity[edge_v]) / (0.12 * temperature))
        contrast_similarity = np.exp(-np.abs(contrast[edge_u] - contrast[edge_v]) / (0.15 * temperature))
        orientation_similarity = 0.5 + 0.5 * np.cos(2.0 * (orientation[edge_u] - orientation[edge_v]))
        orientation_similarity *= np.minimum(coherence[edge_u], coherence[edge_v])
        core_continuity = np.maximum(core[edge_u], core[edge_v]) * (1.0 - np.abs(core[edge_u] - core[edge_v]))
        material_polarity = core - substrate
        material_conflict = np.clip(
            np.abs(material_polarity[edge_u] - material_polarity[edge_v]) * 0.5,
            0.0,
            1.0,
        )

        attraction_terms = np.vstack(
            (
                float(parameters.get("intensity_attraction_weight", 1.0)) * intensity_similarity,
                float(parameters.get("local_contrast_attraction_weight", 0.55)) * contrast_similarity,
                float(parameters.get("orientation_attraction_weight", 0.5)) * orientation_similarity,
                float(parameters.get("core_attraction_weight", 0.8)) * core_continuity,
            )
        )
        repulsion_terms = np.vstack(
            (
                float(parameters.get("boundary_repulsion_weight", 1.25)) * np.maximum(boundary, material_conflict),
                float(parameters.get("rim_repulsion_weight", 0.8)) * rim,
                float(parameters.get("oriented_boundary_repulsion_weight", 0.7)) * persistence,
            )
        )
        if str(parameters.get("affinity_normalization", "weighted_mean")) == "weighted_sum":
            attraction = attraction_terms.sum(axis=0)
            repulsion = repulsion_terms.sum(axis=0)
            scale = max(float(np.percentile(np.r_[attraction, repulsion], 99.0)), 1e-6)
            attraction = np.clip(attraction / scale, 0.0, 1.0)
            repulsion = np.clip(repulsion / scale, 0.0, 1.0)
        else:
            attraction_weight = sum(
                float(parameters.get(key, default))
                for key, default in (
                    ("intensity_attraction_weight", 1.0),
                    ("local_contrast_attraction_weight", 0.55),
                    ("orientation_attraction_weight", 0.5),
                    ("core_attraction_weight", 0.8),
                )
            )
            repulsion_weight = sum(
                float(parameters.get(key, default))
                for key, default in (
                    ("boundary_repulsion_weight", 1.25),
                    ("rim_repulsion_weight", 0.8),
                    ("oriented_boundary_repulsion_weight", 0.7),
                )
            )
            attraction = attraction_terms.sum(axis=0) / max(attraction_weight, 1e-6)
            repulsion = repulsion_terms.sum(axis=0) / max(repulsion_weight, 1e-6)
        attraction = np.where(
            attraction >= float(parameters.get("minimum_attractive_confidence", 0.52)),
            attraction,
            0.0,
        ).astype(np.float32)
        repulsion = np.where(
            repulsion >= float(parameters.get("minimum_repulsive_confidence", 0.52)),
            repulsion,
            0.0,
        ).astype(np.float32)
        attraction_map = _edge_debug_map(labels, pairs, attraction)
        repulsion_map = _edge_debug_map(labels, pairs, repulsion)
        graph = SignedAffinityGraph(
            pixel_labels=labels,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_diagonal_only=diagonal_only,
            attraction=attraction,
            repulsion=repulsion,
            node_area=node_area,
            node_intensity=intensity,
            node_contrast=contrast,
            node_orientation=orientation,
            node_coherence=coherence,
            node_core=core,
            node_substrate=substrate,
            node_centroids=centroids,
            attraction_map=attraction_map,
            repulsion_map=repulsion_map,
            build_time_ms=(perf_counter() - started) * 1000.0,
        )
        with _SIGNED_GRAPH_CACHE_LOCK:
            _SIGNED_GRAPH_CACHE[cache_key] = (features, graph)
            _SIGNED_GRAPH_CACHE.move_to_end(cache_key)
            while len(_SIGNED_GRAPH_CACHE) > _SIGNED_GRAPH_CACHE_MAX:
                _SIGNED_GRAPH_CACHE.popitem(last=False)
        return graph


__all__ = ["SignedAffinityBuilder", "SignedAffinityGraph"]
