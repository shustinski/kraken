"""Connected-component utilities for comparison events."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover - scipy is optional at runtime
    ndi = None


@dataclass(frozen=True, slots=True)
class Component:
    component_id: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    touches_border: bool


def structure_for_connectivity(connectivity: int) -> np.ndarray:
    if int(connectivity) == 4:
        return np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    return np.ones((3, 3), dtype=np.uint8)


def label_components(mask: np.ndarray, *, connectivity: int = 8) -> tuple[np.ndarray, tuple[Component, ...]]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or binary.size == 0:
        return np.zeros_like(binary, dtype=np.int32), ()
    if ndi is None or not hasattr(ndi, "label"):
        labels = _label_components_fallback(binary, connectivity=connectivity)
        count = int(labels.max()) if labels.size else 0
    else:
        labels, count = ndi.label(binary, structure=structure_for_connectivity(connectivity))
        labels = np.asarray(labels, dtype=np.int32)
    components: list[Component] = []
    height, width = binary.shape
    for component_id in range(1, int(count) + 1):
        ys, xs = np.nonzero(labels == component_id)
        if ys.size <= 0:
            continue
        x0 = int(xs.min())
        y0 = int(ys.min())
        x1 = int(xs.max()) + 1
        y1 = int(ys.max()) + 1
        touches = bool(x0 == 0 or y0 == 0 or x1 >= width or y1 >= height)
        components.append(
            Component(
                component_id=component_id,
                area=int(ys.size),
                bbox=(x0, y0, x1 - x0, y1 - y0),
                centroid=(float(np.mean(xs, dtype=np.float64)), float(np.mean(ys, dtype=np.float64))),
                touches_border=touches,
            )
        )
    return labels, tuple(components)


def component_overlap_matrix(labels_a: np.ndarray, count_a: int, labels_b: np.ndarray, count_b: int) -> np.ndarray:
    overlaps = np.zeros((int(count_a), int(count_b)), dtype=np.int32)
    if labels_a.shape != labels_b.shape or count_a <= 0 or count_b <= 0:
        return overlaps
    for component_id in range(1, int(count_a) + 1):
        values, counts = np.unique(labels_b[labels_a == component_id], return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            target_id = int(value)
            if target_id > 0:
                overlaps[component_id - 1, target_id - 1] = int(count)
    return overlaps


def _label_components_fallback(mask: np.ndarray, *, connectivity: int) -> np.ndarray:
    labels = np.zeros(mask.shape, dtype=np.int32)
    next_id = 0
    height, width = mask.shape
    neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    if int(connectivity) == 8:
        neighbors.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            next_id += 1
            stack = [(y, x)]
            labels[y, x] = next_id
            while stack:
                cy, cx = stack.pop()
                for dy, dx in neighbors:
                    ny = cy + dy
                    nx = cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = next_id
                        stack.append((ny, nx))
    return labels
