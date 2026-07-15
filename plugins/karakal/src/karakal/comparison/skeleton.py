"""Skeleton metrics for line-network masks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None


@dataclass(frozen=True, slots=True)
class SkeletonStats:
    skeleton: np.ndarray
    endpoint_count: int
    junction_count: int
    branch_count: int
    length: int


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or binary.size == 0:
        return np.zeros_like(binary, dtype=bool)
    if ndi is None or not hasattr(ndi, "distance_transform_edt") or not hasattr(ndi, "maximum_filter"):
        return binary.copy()
    current = binary.copy()
    previous = np.zeros_like(current, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    # A compact, dependency-light thinning approximation: keep border-normalized medial pixels.
    distance = ndi.distance_transform_edt(current)
    maximum = ndi.maximum_filter(distance, footprint=structure)
    skeleton = current & (distance >= maximum) & (distance > 0)
    if not np.any(skeleton):
        skeleton = current & ~previous
    return np.asarray(skeleton, dtype=bool)


def skeleton_stats(mask: np.ndarray) -> SkeletonStats:
    skeleton = skeletonize_mask(mask)
    neighbors = _neighbor_count(skeleton)
    endpoints = int(np.count_nonzero(skeleton & (neighbors == 1)))
    junctions = int(np.count_nonzero(skeleton & (neighbors >= 3)))
    branch_count = int(max(0, endpoints + junctions))
    return SkeletonStats(
        skeleton=skeleton,
        endpoint_count=endpoints,
        junction_count=junctions,
        branch_count=branch_count,
        length=int(np.count_nonzero(skeleton)),
    )


def skeleton_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    skel_a = skeletonize_mask(mask_a)
    skel_b = skeletonize_mask(mask_b)
    total = int(np.count_nonzero(skel_a)) + int(np.count_nonzero(skel_b))
    if total <= 0:
        return 1.0
    intersection = int(np.count_nonzero(skel_a & skel_b))
    return float((2.0 * intersection) / total)


def skeleton_f1_at_radius(mask_a: np.ndarray, mask_b: np.ndarray, *, radius: int = 1) -> float:
    skel_a = skeletonize_mask(mask_a)
    skel_b = skeletonize_mask(mask_b)
    count_a = int(np.count_nonzero(skel_a))
    count_b = int(np.count_nonzero(skel_b))
    if count_a <= 0 and count_b <= 0:
        return 1.0
    if count_a <= 0 or count_b <= 0:
        return 0.0
    structure = np.ones((2 * max(0, int(radius)) + 1, 2 * max(0, int(radius)) + 1), dtype=bool)
    if ndi is None or not hasattr(ndi, "binary_dilation"):
        dilated_a = _binary_dilation_fallback(skel_a, structure)
        dilated_b = _binary_dilation_fallback(skel_b, structure)
    else:
        dilated_a = ndi.binary_dilation(skel_a, structure=structure)
        dilated_b = ndi.binary_dilation(skel_b, structure=structure)
    precision = float(np.count_nonzero(skel_b & dilated_a) / max(1, count_b))
    recall = float(np.count_nonzero(skel_a & dilated_b) / max(1, count_a))
    if precision + recall <= 0.0:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def cldice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    skel_a = skeletonize_mask(mask_a)
    skel_b = skeletonize_mask(mask_b)
    mask_a_bool = np.asarray(mask_a, dtype=bool)
    mask_b_bool = np.asarray(mask_b, dtype=bool)
    tprec = float(np.count_nonzero(skel_a & mask_b_bool) / max(1, np.count_nonzero(skel_a)))
    tsens = float(np.count_nonzero(skel_b & mask_a_bool) / max(1, np.count_nonzero(skel_b)))
    if tprec + tsens <= 0.0:
        return 1.0 if not np.any(mask_a_bool) and not np.any(mask_b_bool) else 0.0
    return float((2.0 * tprec * tsens) / (tprec + tsens))


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool).astype(np.uint8)
    if ndi is None or not hasattr(ndi, "convolve"):
        padded = np.pad(binary, 1)
        result = np.zeros_like(binary, dtype=np.int16)
        for dy in range(3):
            for dx in range(3):
                if dy == 1 and dx == 1:
                    continue
                result += padded[dy : dy + binary.shape[0], dx : dx + binary.shape[1]]
        return result
    kernel = np.ones((3, 3), dtype=np.int16)
    kernel[1, 1] = 0
    return np.asarray(ndi.convolve(binary, kernel, mode="constant", cval=0), dtype=np.int16)


def _binary_dilation_fallback(mask: np.ndarray, structure: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    footprint = np.asarray(structure, dtype=bool)
    if footprint.ndim != 2 or not np.any(footprint):
        return binary.copy()
    pad_y = footprint.shape[0] // 2
    pad_x = footprint.shape[1] // 2
    padded = np.pad(binary, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant", constant_values=False)
    result = np.zeros_like(binary, dtype=bool)
    for y in range(footprint.shape[0]):
        for x in range(footprint.shape[1]):
            if footprint[y, x]:
                result |= padded[y : y + binary.shape[0], x : x + binary.shape[1]]
    return result
