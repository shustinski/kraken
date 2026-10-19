"""structural_watershed: geodesic label competition."""
from __future__ import annotations
from typing import TYPE_CHECKING
import cv2
import numpy as np
from contour.utils import ensure_binary_mask

from contour.vision.metal_recovery.structural_watershed_parts.ridge_markers import _shift_no_wrap

from contour.vision.metal_recovery.structural_watershed_parts.wide_interior_markers import _confident_substrate

def _boundary_cost_map(features: _StructuralFeatures, config: StructuralWatershedConfig) -> np.ndarray:
    gradient_term = _unit_range(features.magnitude)
    if config.variant in {"s1", "s2", "s5", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14"}:
        return gradient_term
    cost = (
        float(config.boundary_gradient_weight) * gradient_term
        + float(config.boundary_continuity_weight) * _unit_range(features.persistent_edge)
        + float(config.boundary_rim_weight) * _unit_range(features.rim_response)
        + float(config.boundary_orientation_weight)
        * features.coherence
        * _unit_range(features.oriented_gradient)
    )
    return _unit_range(cost)

def _segment_existing_markers(
    features: _StructuralFeatures,
    core_seeds: np.ndarray,
    groove_seeds: np.ndarray,
    *,
    metal_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    empty = np.zeros(features.denoised.shape, dtype=np.uint8)
    cost = _unit_range(features.magnitude)
    core = ensure_binary_mask(core_seeds)
    groove = ensure_binary_mask(groove_seeds)
    groove = cv2.bitwise_and(groove, cv2.bitwise_not(core))
    if not np.any(core) or not np.any(groove):
        fallback = np.where(features.denoised >= max(float(metal_limit), 1.0), 255, 0).astype(np.uint8)
        markers = np.zeros(features.denoised.shape, dtype=np.int32)
        return fallback, markers, cost, core, groove, empty, empty
    markers = np.zeros(features.denoised.shape, dtype=np.int32)
    markers[core > 0] = 2
    markers[groove > 0] = 1
    terrain = _cost_to_terrain(cost)
    cv2.watershed(cv2.cvtColor(terrain, cv2.COLOR_GRAY2BGR), markers)
    mask = ensure_binary_mask((markers == 2).astype(np.uint8) * 255)
    return _restore_fov_edge(mask), markers, cost, core, groove, empty, empty

def _sparse_background_markers(features: _StructuralFeatures, foreground_markers: np.ndarray) -> np.ndarray:
    """High-confidence substrate only. Narrow gaps stay unlabeled for competition."""

    return _confident_substrate(features, foreground_markers)

def _instance_watershed(
    foreground_markers: np.ndarray,
    background_markers: np.ndarray,
    cost: np.ndarray,
    *,
    grow: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    fg = ensure_binary_mask(foreground_markers)
    bg = cv2.bitwise_and(ensure_binary_mask(background_markers), cv2.bitwise_not(fg))
    markers = _seed_markers(fg, bg)
    if not np.any(fg):
        return np.zeros(fg.shape, dtype=np.uint8), markers
    if (not np.any(bg)) and (not grow):
        return fg, markers
    terrain = _cost_to_terrain(cost)
    cv2.watershed(cv2.cvtColor(terrain, cv2.COLOR_GRAY2BGR), markers)
    mask = ensure_binary_mask((markers > 1).astype(np.uint8) * 255)
    return _restore_fov_edge(mask), markers

def _geodesic_label_competition(
    foreground_markers: np.ndarray,
    background_markers: np.ndarray,
    cost: np.ndarray,
    *,
    foreground_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Multi-source competition: path cost ≈ distance × (1 + boundary)."""

    fg = ensure_binary_mask(foreground_markers)
    bg = cv2.bitwise_and(ensure_binary_mask(background_markers), cv2.bitwise_not(fg))
    if foreground_labels is not None and np.any(foreground_labels > 0):
        markers = _seed_labeled_markers(foreground_labels, bg)
        distance_source = np.where(foreground_labels > 0, 255, 0).astype(np.uint8)
    else:
        markers = _seed_markers(fg, bg)
        distance_source = fg
    if not np.any(distance_source):
        return markers
    distance = cv2.distanceTransform(
        cv2.bitwise_not(distance_source),
        cv2.DIST_L2,
        3,
    ).astype(np.float32)
    safe_cost = np.nan_to_num(np.clip(cost.astype(np.float32), 0.0, 1.0), nan=0.0)
    terrain = _cost_to_terrain(_unit_range(distance * (1.0 + 2.0 * safe_cost)))
    cv2.watershed(cv2.cvtColor(terrain, cv2.COLOR_GRAY2BGR), markers)
    return markers

def _seed_labeled_markers(foreground_labels: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Seed watershed with pre-assigned logical IDs, including disconnected fragments."""

    markers = np.zeros(foreground_labels.shape, dtype=np.int32)
    if np.any(background):
        markers[background > 0] = 1
    positive = foreground_labels > 0
    if np.any(positive):
        markers[positive] = foreground_labels[positive] + 1
    return markers

def _seed_markers(foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
    markers = np.zeros(foreground.shape, dtype=np.int32)
    if np.any(background):
        markers[background > 0] = 1
    if np.any(foreground):
        _count, labels = cv2.connectedComponents((foreground > 0).astype(np.uint8), connectivity=8)
        markers[foreground > 0] = labels[foreground > 0] + 1
    return markers

def _finalize_instance_labels(markers: np.ndarray) -> np.ndarray:
    """Map watershed markers to 0=background, 1..N=instances. Cuts join a neighbor."""

    restored = _restore_instance_fov_edge(markers)
    labels = np.zeros(restored.shape, dtype=np.int32)
    labels[restored > 1] = restored[restored > 1] - 1
    cuts = restored < 0
    if not np.any(cuts) or not np.any(labels):
        labels[cuts] = 0
        return labels
    neighbor = np.zeros_like(labels)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        shifted = _shift_no_wrap(labels, dx, dy)
        take = cuts & (neighbor == 0) & (shifted > 0)
        neighbor[take] = shifted[take]
    labels[cuts] = neighbor[cuts]
    return labels

def _restore_instance_fov_edge(labels: np.ndarray) -> np.ndarray:
    restored = labels.copy()
    inward = (
        (restored[1, :], restored[-2, :], restored[:, 1], restored[:, -2])
    )
    restored[0, :] = np.where(restored[0, :] <= 0, inward[0], restored[0, :])
    restored[-1, :] = np.where(restored[-1, :] <= 0, inward[1], restored[-1, :])
    restored[:, 0] = np.where(restored[:, 0] <= 0, inward[2], restored[:, 0])
    restored[:, -1] = np.where(restored[:, -1] <= 0, inward[3], restored[:, -1])
    return restored

def _binary_components_as_labels(mask: np.ndarray) -> np.ndarray:
    _count, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return labels.astype(np.int32)

def _instance_fragment_stats(labels: np.ndarray) -> tuple[int, int]:
    positive = labels > 0
    if not np.any(positive):
        return 0, 0
    cc_count, components = cv2.connectedComponents(positive.astype(np.uint8), connectivity=8)
    encoded = labels.astype(np.int64) * (int(cc_count) + 1) + components.astype(np.int64)
    unique_pairs = np.unique(encoded[positive])
    instance_ids = (unique_pairs // (int(cc_count) + 1)).astype(np.int32)
    per_instance = np.bincount(instance_ids)
    instance_count = int(np.unique(labels[positive]).size)
    fragmented = int(np.count_nonzero(per_instance > 1))
    return instance_count, fragmented

def _label_boundary_map(labels: np.ndarray) -> np.ndarray:
    fg = np.maximum(labels, 0)
    right = _shift_no_wrap(fg, 1, 0)
    down = _shift_no_wrap(fg, 0, 1)
    boundary = ((fg > 0) & (right > 0) & (fg != right)) | ((fg > 0) & (down > 0) & (fg != down))
    return np.where(boundary, 255, 0).astype(np.uint8)

def _restore_fov_edge(mask: np.ndarray) -> np.ndarray:
    """OpenCV watershed paints the image border as a cut; restore FOV-touching metal."""

    restored = ensure_binary_mask(mask)
    restored[0, :] = np.maximum(restored[0, :], restored[1, :])
    restored[-1, :] = np.maximum(restored[-1, :], restored[-2, :])
    restored[:, 0] = np.maximum(restored[:, 0], restored[:, 1])
    restored[:, -1] = np.maximum(restored[:, -1], restored[:, -2])
    return restored

def _cost_to_terrain(cost: np.ndarray) -> np.ndarray:
    return np.clip(cost * 255.0, 0.0, 255.0).astype(np.uint8)

def _unit_range(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    low, high = np.percentile(values, (1.0, 99.5))
    span = float(high - low)
    if span <= 1e-6:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values.astype(np.float32) - float(low)) / span, 0.0, 1.0)

def _to_u8(values: np.ndarray) -> np.ndarray:
    if values.dtype == np.uint8:
        return values
    return _cost_to_terrain(_unit_range(values.astype(np.float32)))

if TYPE_CHECKING:
    from contour.vision.metal_recovery.structural_watershed_parts.clamped_structural_watershed_config import StructuralWatershedConfig
    from contour.vision.metal_recovery.structural_watershed_parts.extract_structural_features import _StructuralFeatures
