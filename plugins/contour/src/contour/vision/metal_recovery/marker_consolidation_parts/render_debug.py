"""marker_consolidation: render debug."""
from __future__ import annotations
import cv2
import numpy as np
from contour.utils import ensure_binary_mask, ensure_uint8

def _connected_with_stats(
    mask: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    binary = ensure_binary_mask(mask)
    empty_stats = np.zeros((1, 5), dtype=np.int32)
    empty_centroids = np.zeros((1, 2), dtype=np.float64)
    if not np.any(binary):
        return 0, np.zeros(binary.shape, dtype=np.int32), empty_stats, empty_centroids
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return int(count - 1), labels.astype(np.int32), stats, centroids

def _positive_id_count(labels: np.ndarray) -> int:
    if labels.size == 0 or not np.any(labels):
        return 0
    ids = np.unique(labels)
    return int(np.count_nonzero(ids > 0))

def _compact_labels(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if ids.size == 0:
        return np.zeros(labels.shape, dtype=np.int32)
    lookup = np.zeros(int(ids.max()) + 1, dtype=np.int32)
    lookup[ids] = np.arange(1, int(ids.size) + 1, dtype=np.int32)
    out = np.zeros(labels.shape, dtype=np.int32)
    positive = labels > 0
    out[positive] = lookup[labels[positive]]
    return out

def _empty_band_images(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    empty = np.zeros((*shape, 3), dtype=np.uint8)
    return {
        "metal_structural_conductor_bands": empty,
        "metal_structural_transverse_samples": empty,
        "metal_structural_band_groups_accepted": empty,
        "metal_structural_band_groups_rejected": empty,
    }

def _find(parent: np.ndarray, item: int) -> int:
    root = item
    while parent[root] != root:
        root = int(parent[root])
    while parent[item] != item:
        nxt = int(parent[item])
        parent[item] = root
        item = nxt
    return int(root)

def _union(parent: np.ndarray, left: int, right: int) -> None:
    root_left = _find(parent, left)
    root_right = _find(parent, right)
    if root_left != root_right:
        parent[root_right] = root_left

def _remap_by_parent(labels: np.ndarray, parent: np.ndarray) -> np.ndarray:
    roots = np.arange(parent.size, dtype=np.int32)
    for index in range(1, parent.size):
        roots[index] = _find(parent, index)
    remapped = np.zeros(labels.shape, dtype=np.int32)
    positive = labels > 0
    remapped[positive] = roots[labels[positive]]
    return _compact_labels(remapped)

def _acute_angle_delta(first: float, second: float) -> float:
    delta = abs((first - second + 0.5 * np.pi) % np.pi - 0.5 * np.pi)
    return float(delta)

def _label_overlay(labels: np.ndarray) -> np.ndarray:
    color = np.zeros((*labels.shape, 3), dtype=np.uint8)
    positive = labels > 0
    if not np.any(positive):
        return color
    hsv = np.zeros((*labels.shape, 3), dtype=np.uint8)
    hsv[:, :, 0] = ((labels.astype(np.int32) * 37) % 180).astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.where(positive, 220, 0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def _draw_links(
    base: np.ndarray,
    links: tuple[RidgeLink, ...],
    color: tuple[int, int, int],
    *,
    reason: str | None = None,
) -> np.ndarray:
    canvas = cv2.cvtColor(ensure_uint8(base), cv2.COLOR_GRAY2BGR)
    canvas = (canvas.astype(np.float32) * 0.55).astype(np.uint8)
    for link in links:
        if reason is not None and link.reason != reason:
            continue
        cv2.line(canvas, link.point_a, link.point_b, color, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, link.point_a, 1, color, -1)
        cv2.circle(canvas, link.point_b, 1, color, -1)
    return canvas

def _render_debug(
    intensity: np.ndarray,
    *,
    ridge_raw: np.ndarray,
    ridge_logical: np.ndarray,
    wide_raw: np.ndarray,
    wide_logical: np.ndarray,
    combined: np.ndarray,
    accepted: tuple[RidgeLink, ...],
    rejected: tuple[RidgeLink, ...],
) -> dict[str, np.ndarray]:
    return {
        "metal_structural_ridge_fragments": _label_overlay(ridge_raw),
        "metal_structural_wide_fragments": _label_overlay(wide_raw),
        "metal_structural_ridge_links_accepted": _draw_links(
            intensity, accepted, (0, 220, 0)
        ),
        "metal_structural_ridge_links_rejected": _draw_links(
            intensity, rejected, (0, 200, 255)
        ),
        "metal_structural_ridge_links_boundary_veto": _draw_links(
            intensity,
            rejected,
            (0, 0, 255),
            reason="boundary_veto",
        ),
        "metal_structural_logical_ridge": _label_overlay(ridge_logical),
        "metal_structural_logical_wide": _label_overlay(wide_logical),
        "metal_structural_logical_markers": _label_overlay(combined),
    }
