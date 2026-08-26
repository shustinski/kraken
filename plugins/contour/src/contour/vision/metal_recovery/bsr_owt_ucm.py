"""Windows-compatible adaptation of the Berkeley OWT-UCM hierarchy.

Derived from ``BSR/grouping/lib/contours2ucm.m`` and
``BSR/grouping/source/ucm/ucm_mean_pb.cpp`` by Pablo Arbelaez and
contributors (2009-2010), licensed under AGPL-3.0-or-later. See the
Contour ``THIRD_PARTY_NOTICES.md`` file.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import cv2
import numpy as np

from ...application.preview_cancellation import raise_if_preview_cancelled


@dataclass(frozen=True, slots=True)
class BsrUcmResult:
    """BSR UCM at image resolution plus its complete merge hierarchy."""

    ucm: np.ndarray
    hierarchy: tuple[dict[str, float | int], ...]


@dataclass(slots=True)
class _NeighbourBoundary:
    total: float
    length: float
    edge_ids: list[int]


def _bsr_normalize(values: np.ndarray) -> np.ndarray:
    """Apply the sigmoid calibration shipped in BSR ``contours2ucm.m``."""
    calibrated = 1.0 / (1.0 + np.exp(-(-2.7487 + 11.1189 * values)))
    return np.clip((calibrated - 0.0602) / (1.0 - 0.0602), 0.0, 1.0).astype(np.float32)


def _boundary_tangent_bins(labels: np.ndarray, bins: int) -> np.ndarray:
    """Estimate the local watershed-contour tangent, as BSR ``fit_contour`` does."""
    boundary = np.zeros(labels.shape, dtype=np.float32)
    boundary[:, :-1] = np.maximum(boundary[:, :-1], labels[:, :-1] != labels[:, 1:])
    boundary[:, 1:] = np.maximum(boundary[:, 1:], labels[:, :-1] != labels[:, 1:])
    boundary[:-1, :] = np.maximum(boundary[:-1, :], labels[:-1, :] != labels[1:, :])
    boundary[1:, :] = np.maximum(boundary[1:, :], labels[:-1, :] != labels[1:, :])
    smooth = cv2.GaussianBlur(boundary, (0, 0), 1.0)
    gradient_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    tensor_xx = cv2.GaussianBlur(gradient_x * gradient_x, (0, 0), 1.0)
    tensor_xy = cv2.GaussianBlur(gradient_x * gradient_y, (0, 0), 1.0)
    tensor_yy = cv2.GaussianBlur(gradient_y * gradient_y, (0, 0), 1.0)
    normal = 0.5 * np.arctan2(2.0 * tensor_xy, tensor_xx - tensor_yy)
    tangent = np.mod(normal + 0.5 * np.pi, np.pi)
    best_score = np.full(labels.shape, -np.inf, dtype=np.float32)
    best_bin = np.zeros(labels.shape, dtype=np.int32)
    for index in range(bins):
        center = np.pi * float(index) / float(bins)
        score = np.cos(2.0 * (tangent - center)).astype(np.float32)
        better = score > best_score
        best_score[better] = score[better]
        best_bin[better] = index
    return best_bin


def _boundary_records(
    labels: np.ndarray,
    oriented_channels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Collect BSR oriented strengths for every initial-region boundary."""
    bins, height, width = oriented_channels.shape
    tangent_bins = _boundary_tangent_bins(labels, bins)
    pair_chunks: list[np.ndarray] = []
    strength_chunks: list[np.ndarray] = []
    pixel_chunks: list[np.ndarray] = []

    def collect(
        first: np.ndarray,
        second: np.ndarray,
        first_index: np.ndarray,
        second_index: np.ndarray,
    ) -> None:
        valid = (first > 0) & (second > 0) & (first != second)
        if not np.any(valid):
            return
        low = np.minimum(first[valid], second[valid]).astype(np.int64)
        high = np.maximum(first[valid], second[valid]).astype(np.int64)
        flat_first = first_index[valid]
        flat_second = second_index[valid]
        rows = flat_first // width
        cols = flat_first % width
        orientation = tangent_bins[rows, cols]
        first_strength = oriented_channels[orientation, rows, cols]
        second_rows = flat_second // width
        second_cols = flat_second % width
        second_strength = oriented_channels[orientation, second_rows, second_cols]
        pair_chunks.append(np.column_stack((low, high)))
        strength_chunks.append(np.maximum(first_strength, second_strength).astype(np.float64))
        pixel_chunks.append(np.column_stack((flat_first, flat_second)))

    flat = np.arange(height * width, dtype=np.int64).reshape(height, width)
    collect(labels[:, :-1], labels[:, 1:], flat[:, :-1], flat[:, 1:])
    collect(labels[:-1, :], labels[1:, :], flat[:-1, :], flat[1:, :])
    if not pair_chunks:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64), []

    pairs = np.concatenate(pair_chunks)
    strengths = np.concatenate(strength_chunks)
    pixels = np.concatenate(pixel_chunks)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    strengths = strengths[order]
    pixels = pixels[order]
    starts = np.r_[0, np.flatnonzero(np.any(pairs[1:] != pairs[:-1], axis=1)) + 1]
    ends = np.r_[starts[1:], len(pairs)]
    unique_pairs = pairs[starts]
    totals = np.add.reduceat(strengths, starts)
    pixel_groups = [pixels[start:end].reshape(-1) for start, end in zip(starts, ends, strict=True)]
    lengths = (ends - starts).astype(np.float64)
    return unique_pairs, np.column_stack((totals, lengths)), pixel_groups


def build_bsr_ucm(labels: np.ndarray, oriented_channels: np.ndarray) -> BsrUcmResult:
    """Run BSR dynamic mean-boundary agglomeration on an OWT partition."""
    labels = np.asarray(labels, dtype=np.int32)
    channels = np.asarray(oriented_channels, dtype=np.float32)
    if channels.ndim != 3 or channels.shape[1:] != labels.shape:
        raise ValueError("oriented_channels must have shape (orientation, height, width)")
    pairs, totals_lengths, pixel_groups = _boundary_records(labels, channels)
    if pairs.size == 0:
        return BsrUcmResult(np.zeros(labels.shape, dtype=np.float32), ())

    maximum_label = int(labels.max())
    active = np.ones(maximum_label + 1, dtype=bool)
    active[0] = False
    areas = np.bincount(labels.ravel(), minlength=maximum_label + 1).astype(np.int64)
    neighbours: list[dict[int, _NeighbourBoundary]] = [dict() for _ in range(maximum_label + 1)]
    queue: list[tuple[float, int, int]] = []
    for edge_id, ((first, second), (total, length)) in enumerate(zip(pairs, totals_lengths, strict=True)):
        record = _NeighbourBoundary(float(total), float(length), [edge_id])
        neighbours[int(first)][int(second)] = record
        neighbours[int(second)][int(first)] = record
        heapq.heappush(queue, (float(total / length), int(first), int(second)))

    raw_levels = np.zeros(len(pairs), dtype=np.float64)
    raw_hierarchy: list[tuple[int, int, int, float, int]] = []
    current_energy = 0.0
    while queue:
        if len(raw_hierarchy) % 1024 == 0:
            raise_if_preview_cancelled()
        queued_energy, first, second = heapq.heappop(queue)
        if not active[first] or not active[second]:
            continue
        queued_record = neighbours[first].get(second)
        if queued_record is None:
            continue
        total = queued_record.total
        length = queued_record.length
        energy = total / max(length, 1.0)
        if not np.isclose(queued_energy, energy, rtol=0.0, atol=1e-12):
            continue
        current_energy = max(current_energy, energy)
        father = max(first, second)
        son = min(first, second)
        raw_levels[np.asarray(queued_record.edge_ids, dtype=np.int64)] = current_energy
        raw_hierarchy.append((first, second, father, current_energy, int(areas[first] + areas[second])))

        neighbours[father].pop(son, None)
        neighbours[son].pop(father, None)
        for neighbour, son_record in list(neighbours[son].items()):
            neighbours[neighbour].pop(son, None)
            if not active[neighbour] or neighbour == father:
                continue
            father_record = neighbours[father].get(neighbour)
            if father_record is None:
                merged_record = _NeighbourBoundary(
                    son_record.total,
                    son_record.length,
                    list(son_record.edge_ids),
                )
            else:
                merged_ids = list(father_record.edge_ids)
                merged_ids.extend(son_record.edge_ids)
                merged_record = _NeighbourBoundary(
                    father_record.total + son_record.total,
                    father_record.length + son_record.length,
                    merged_ids,
                )
            neighbours[father][neighbour] = merged_record
            neighbours[neighbour][father] = merged_record
            merged_energy = merged_record.total / max(merged_record.length, 1.0)
            heapq.heappush(queue, (merged_energy, min(father, neighbour), max(father, neighbour)))
        neighbours[son].clear()
        active[son] = False
        areas[father] += areas[son]

    normalized_levels = _bsr_normalize(raw_levels)
    ucm_flat = np.zeros(labels.size, dtype=np.float32)
    for level, pixels in zip(normalized_levels, pixel_groups, strict=True):
        np.maximum.at(ucm_flat, pixels, level)
    ucm = ucm_flat.reshape(labels.shape)
    hierarchy_values = _bsr_normalize(np.asarray([item[3] for item in raw_hierarchy], dtype=np.float64))
    hierarchy = tuple(
        {
            "left": first,
            "right": second,
            "merged": merged,
            "saliency": float(level),
            "area": area,
        }
        for (first, second, merged, _raw_level, area), level in zip(
            raw_hierarchy,
            hierarchy_values,
            strict=True,
        )
    )
    return BsrUcmResult(ucm=ucm, hierarchy=hierarchy)


def cut_bsr_hierarchy(
    labels: np.ndarray,
    hierarchy: tuple[dict[str, float | int], ...],
    level: float,
) -> np.ndarray:
    """Cut the BSR ultrametric hierarchy at ``level`` and relabel consecutively."""
    maximum_label = int(labels.max())
    parent = np.arange(maximum_label + 1, dtype=np.int32)

    def find(value: int) -> int:
        root = value
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[value]) != value:
            previous = int(parent[value])
            parent[value] = root
            value = previous
        return root

    for merge in hierarchy:
        if float(merge["saliency"]) > level:
            break
        left = find(int(merge["left"]))
        right = find(int(merge["right"]))
        if left != right:
            parent[min(left, right)] = max(left, right)
    roots = np.zeros(maximum_label + 1, dtype=np.int32)
    for label_id in range(1, maximum_label + 1):
        roots[label_id] = find(label_id)
    unique = np.unique(roots[1:])
    remap = np.zeros(maximum_label + 1, dtype=np.int32)
    remap[unique] = np.arange(1, unique.size + 1, dtype=np.int32)
    return remap[roots[np.maximum(labels, 0)]]


__all__ = ["BsrUcmResult", "build_bsr_ucm", "cut_bsr_hierarchy"]
