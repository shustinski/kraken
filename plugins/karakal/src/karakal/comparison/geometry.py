"""Geometry and boundary metrics for mask comparison."""
from __future__ import annotations

import numpy as np

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

_CKDTREE_UNSET = object()
_cached_ckdtree: object = _CKDTREE_UNSET


def _get_ckdtree():
    """Import scipy.spatial only when boundary distance metrics need it."""

    global _cached_ckdtree
    if _cached_ckdtree is _CKDTREE_UNSET:
        try:
            from scipy.spatial import cKDTree
        except Exception:  # pragma: no cover
            cKDTree = None
        _cached_ckdtree = cKDTree
    return _cached_ckdtree


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or binary.size == 0:
        return np.zeros_like(binary, dtype=bool)
    if ndi is None or not hasattr(ndi, "binary_erosion"):
        eroded = binary.copy()
        eroded[1:-1, 1:-1] = (
            binary[1:-1, 1:-1]
            & binary[:-2, 1:-1]
            & binary[2:, 1:-1]
            & binary[1:-1, :-2]
            & binary[1:-1, 2:]
        )
    else:
        eroded = ndi.binary_erosion(binary, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return np.asarray(binary & ~eroded, dtype=bool)


def boundary_distance_metrics(mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, float | np.ndarray | None]:
    boundary_a = boundary_mask(mask_a)
    boundary_b = boundary_mask(mask_b)
    intersection = int(np.count_nonzero(boundary_a & boundary_b))
    union = int(np.count_nonzero(boundary_a | boundary_b))
    boundary_iou = 1.0 if union <= 0 else float(intersection / union)
    distances = _symmetric_boundary_distances(boundary_a, boundary_b)
    if distances.size <= 0:
        assd = 0.0
        hd95 = 0.0
    else:
        assd = float(np.mean(distances, dtype=np.float64))
        hd95 = float(np.percentile(distances, 95.0))
    return {
        "boundary_a": boundary_a,
        "boundary_b": boundary_b,
        "boundary_iou_ab": boundary_iou,
        "assd_ab": assd,
        "hd95_ab": hd95,
        "centroid_distance_ab": centroid_distance(mask_a, mask_b),
        "perimeter_delta": abs(int(np.count_nonzero(boundary_a)) - int(np.count_nonzero(boundary_b))) / max(1, max(int(np.count_nonzero(boundary_a)), int(np.count_nonzero(boundary_b)))),
    }


def centroid_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    centroid_a = _centroid(mask_a)
    centroid_b = _centroid(mask_b)
    if centroid_a is None and centroid_b is None:
        return 0.0
    if centroid_a is None or centroid_b is None:
        return float("inf")
    return float(np.hypot(centroid_a[0] - centroid_b[0], centroid_a[1] - centroid_b[1]))


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if ys.size <= 0:
        return None
    return float(np.mean(xs, dtype=np.float64)), float(np.mean(ys, dtype=np.float64))


def _symmetric_boundary_distances(boundary_a: np.ndarray, boundary_b: np.ndarray) -> np.ndarray:
    points_a = np.column_stack(np.nonzero(boundary_a))
    points_b = np.column_stack(np.nonzero(boundary_b))
    if points_a.size <= 0 and points_b.size <= 0:
        return np.zeros((0,), dtype=np.float32)
    if points_a.size <= 0 or points_b.size <= 0:
        max_distance = float(np.hypot(boundary_a.shape[0], boundary_a.shape[1]))
        return np.full((max(len(points_a), len(points_b), 1),), max_distance, dtype=np.float32)
    cKDTree = _get_ckdtree()
    if cKDTree is None:
        distances = []
        for source, target in ((points_a, points_b), (points_b, points_a)):
            diff = source[:, None, :] - target[None, :, :]
            distances.extend(np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1).tolist())
        return np.asarray(distances, dtype=np.float32)
    tree_a = cKDTree(points_a.astype(np.float32))
    tree_b = cKDTree(points_b.astype(np.float32))
    dist_ab = tree_b.query(points_a.astype(np.float32), k=1)[0]
    dist_ba = tree_a.query(points_b.astype(np.float32), k=1)[0]
    return np.concatenate([np.asarray(dist_ab, dtype=np.float32), np.asarray(dist_ba, dtype=np.float32)])
