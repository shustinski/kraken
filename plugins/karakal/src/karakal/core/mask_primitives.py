"""Low-level binary-mask and morphology primitives."""

from __future__ import annotations

from .repository_shared import (
    EPS,
    POLYGON_CONFIDENCE_COMPLETION_AXIS_RATIO,
    POLYGON_CONFIDENCE_COMPLETION_MAJOR_SCALE,
    cv2,
    ndi,
    np,
)


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, float(value))))


def _normalize_ratio(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return 0.0
    return float(value / (1.0 + value))


def _weighted_mean(pairs: list[tuple[float, float]]) -> float:
    valid = [(float(v), float(w)) for v, w in pairs if np.isfinite(v) and np.isfinite(w) and w > 0.0]
    if not valid:
        return 0.0
    numerator = sum(v * w for v, w in valid)
    denominator = sum(w for _v, w in valid)
    return float(numerator / max(EPS, denominator))


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.size == 0 or not np.any(mask_bool):
        return np.zeros_like(mask_bool, dtype=np.int32), 0
    if ndi is not None and hasattr(ndi, "label"):
        labels, count = ndi.label(mask_bool, structure=np.ones((3, 3), dtype=np.uint8))
        return np.asarray(labels, dtype=np.int32), int(count)
    if cv2 is not None:
        count, labels = cv2.connectedComponents(np.asarray(mask_bool, dtype=np.uint8), connectivity=8)
        return np.asarray(labels, dtype=np.int32), max(0, int(count) - 1)
    height, width = mask_bool.shape
    labels = np.zeros((height, width), dtype=np.int32)
    next_label = 1
    for row in range(height):
        for column in range(width):
            if not mask_bool[row, column] or labels[row, column] != 0:
                continue
            queue = [(row, column)]
            labels[row, column] = next_label
            while queue:
                current_row, current_column = queue.pop()
                for neighbor_row in range(max(0, current_row - 1), min(height, current_row + 2)):
                    for neighbor_column in range(max(0, current_column - 1), min(width, current_column + 2)):
                        if not mask_bool[neighbor_row, neighbor_column] or labels[neighbor_row, neighbor_column] != 0:
                            continue
                        labels[neighbor_row, neighbor_column] = next_label
                        queue.append((neighbor_row, neighbor_column))
            next_label += 1
    return labels, next_label - 1


def _has_fast_component_label_backend() -> bool:
    return bool((ndi is not None and hasattr(ndi, "label")) or cv2 is not None)


def _has_distance_transform_backend() -> bool:
    return bool((ndi is not None and hasattr(ndi, "distance_transform_edt")) or cv2 is not None)


def _binary_erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask_bool.copy()
    if ndi is not None and hasattr(ndi, "binary_erosion"):
        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        return np.asarray(ndi.binary_erosion(mask_bool, structure=structure), dtype=bool)
    padded = np.pad(mask_bool, radius, mode="constant", constant_values=False)
    result = np.ones_like(mask_bool, dtype=bool)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            result &= padded[
                row_offset : row_offset + mask_bool.shape[0], column_offset : column_offset + mask_bool.shape[1]
            ]
    return result


def _binary_dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask_bool.copy()
    if ndi is not None and hasattr(ndi, "binary_dilation"):
        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        return np.asarray(ndi.binary_dilation(mask_bool, structure=structure), dtype=bool)
    padded = np.pad(mask_bool, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask_bool)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            result |= padded[
                row_offset : row_offset + mask_bool.shape[0], column_offset : column_offset + mask_bool.shape[1]
            ]
    return result


def _binary_dilate_rect(mask: np.ndarray, radius_y: int = 1, radius_x: int = 1) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    ry = max(0, int(radius_y))
    rx = max(0, int(radius_x))
    if ry <= 0 and rx <= 0:
        return mask_bool.copy()
    if ndi is not None and hasattr(ndi, "binary_dilation"):
        structure = np.ones((2 * ry + 1, 2 * rx + 1), dtype=bool)
        return np.asarray(ndi.binary_dilation(mask_bool, structure=structure), dtype=bool)
    padded = np.pad(mask_bool, ((ry, ry), (rx, rx)), mode="constant", constant_values=False)
    result = np.zeros_like(mask_bool)
    for row_offset in range(2 * ry + 1):
        for column_offset in range(2 * rx + 1):
            result |= padded[
                row_offset : row_offset + mask_bool.shape[0], column_offset : column_offset + mask_bool.shape[1]
            ]
    return result


def _completion_radii_for_mask(mask: np.ndarray, base_radius: int) -> tuple[int, int]:
    radius = max(0, int(base_radius))
    if radius <= 0:
        return 0, 0
    mask_bool = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0 or xs.size == 0:
        return radius, radius
    height = int(np.max(ys) - np.min(ys) + 1)
    width = int(np.max(xs) - np.min(xs) + 1)
    axis_ratio = float(max(height, width) / max(1, min(height, width)))
    major_scale = max(1, int(POLYGON_CONFIDENCE_COMPLETION_MAJOR_SCALE))
    if axis_ratio >= float(POLYGON_CONFIDENCE_COMPLETION_AXIS_RATIO):
        if height >= width:
            return radius * major_scale, radius
        return radius, radius * major_scale
    return radius, radius


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.size == 0:
        return np.zeros_like(mask_bool)
    padded = np.pad(mask_bool, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    interior = center & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    return center & np.logical_not(interior)


def _distance_transform(mask: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if ndi is not None and hasattr(ndi, "distance_transform_edt"):
        return np.asarray(ndi.distance_transform_edt(mask_bool), dtype=np.float32)
    if cv2 is not None:
        distances = cv2.distanceTransform(np.asarray(mask_bool, dtype=np.uint8), cv2.DIST_L2, 5)
        return np.asarray(distances, dtype=np.float32)
    raise RuntimeError("Distance transform backend is unavailable")


_NEIGHBOR_SLICE_ORDER = (
    (slice(0, -2), slice(0, -2)),
    (slice(0, -2), slice(1, -1)),
    (slice(0, -2), slice(2, None)),
    (slice(1, -1), slice(0, -2)),
    (slice(1, -1), slice(2, None)),
    (slice(2, None), slice(0, -2)),
    (slice(2, None), slice(1, -1)),
    (slice(2, None), slice(2, None)),
)


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=np.uint8), 1, mode="constant", constant_values=0)
    neighbors = np.zeros_like(mask, dtype=np.uint8)
    for row_slice, column_slice in _NEIGHBOR_SLICE_ORDER:
        neighbors += padded[row_slice, column_slice]
    return neighbors


def _transition_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=np.uint8), 1, mode="constant", constant_values=0)
    p2 = padded[:-2, 1:-1]
    p3 = padded[:-2, 2:]
    p4 = padded[1:-1, 2:]
    p5 = padded[2:, 2:]
    p6 = padded[2:, 1:-1]
    p7 = padded[2:, :-2]
    p8 = padded[1:-1, :-2]
    p9 = padded[:-2, :-2]
    sequence = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
    transitions = np.zeros_like(mask, dtype=np.uint8)
    for left, right in zip(sequence[:-1], sequence[1:]):
        transitions += ((left == 0) & (right == 1)).astype(np.uint8)
    return transitions


def skeletonize(mask: np.ndarray) -> np.ndarray:
    image = np.asarray(mask, dtype=np.uint8).copy()
    if image.size == 0:
        return image.astype(bool)
    changed = True
    while changed:
        changed = False
        neighbors = _neighbor_count(image)
        transitions = _transition_count(image)
        padded = np.pad(image, 1, mode="constant", constant_values=0)
        p2 = padded[:-2, 1:-1]
        p4 = padded[1:-1, 2:]
        p6 = padded[2:, 1:-1]
        p8 = padded[1:-1, :-2]
        remove = (
            (image == 1)
            & (neighbors >= 2)
            & (neighbors <= 6)
            & (transitions == 1)
            & ((p2 * p4 * p6) == 0)
            & ((p4 * p6 * p8) == 0)
        )
        if np.any(remove):
            image[remove] = 0
            changed = True
        neighbors = _neighbor_count(image)
        transitions = _transition_count(image)
        padded = np.pad(image, 1, mode="constant", constant_values=0)
        p2 = padded[:-2, 1:-1]
        p4 = padded[1:-1, 2:]
        p6 = padded[2:, 1:-1]
        p8 = padded[1:-1, :-2]
        remove = (
            (image == 1)
            & (neighbors >= 2)
            & (neighbors <= 6)
            & (transitions == 1)
            & ((p2 * p4 * p8) == 0)
            & ((p2 * p6 * p8) == 0)
        )
        if np.any(remove):
            image[remove] = 0
            changed = True
    return image.astype(bool)


def _endpoint_count(skeleton: np.ndarray) -> int:
    if not np.any(skeleton):
        return 0
    neighbors = _neighbor_count(skeleton)
    return int(np.count_nonzero(skeleton & (neighbors == 1)))


def _branchpoint_count(skeleton: np.ndarray) -> int:
    if not np.any(skeleton):
        return 0
    neighbors = _neighbor_count(skeleton)
    return int(np.count_nonzero(skeleton & (neighbors >= 3)))


def _component_area_stats(labels: np.ndarray, count: int) -> tuple[list[float], float]:
    if count <= 0:
        return [], 0.0
    area_counts = np.bincount(np.asarray(labels, dtype=np.int32).ravel(), minlength=count + 1)
    component_areas = [float(value) for value in area_counts[1 : count + 1]]
    mean_component_area = float(np.mean(component_areas, dtype=np.float64)) if component_areas else 0.0
    return component_areas, mean_component_area


def _mask_structure(
    mask: np.ndarray,
    *,
    include_skeleton: bool = True,
    include_boundary_distance: bool = False,
    include_component_labels: bool = True,
) -> dict[str, object]:
    mask_bool = np.asarray(mask, dtype=bool)
    if include_component_labels:
        labels, count = _label_components(mask_bool)
    else:
        labels = np.zeros(mask_bool.shape, dtype=np.int32)
        count = 0
    boundary = _boundary_mask(mask_bool)
    boundary_dist = (
        _distance_transform(~boundary)
        if include_boundary_distance and np.any(boundary) and _has_distance_transform_backend()
        else None
    )
    if include_skeleton:
        skeleton = skeletonize(mask_bool)
        if np.any(skeleton):
            skeleton_neighbors = _neighbor_count(skeleton)
            endpoint_count = int(np.count_nonzero(skeleton & (skeleton_neighbors == 1)))
            branchpoint_count = int(np.count_nonzero(skeleton & (skeleton_neighbors >= 3)))
        else:
            skeleton_neighbors = np.zeros_like(mask_bool, dtype=np.uint8)
            endpoint_count = 0
            branchpoint_count = 0
    else:
        skeleton = np.zeros_like(mask_bool, dtype=bool)
        skeleton_neighbors = np.zeros_like(mask_bool, dtype=np.uint8)
        endpoint_count = 0
        branchpoint_count = 0
    _component_areas, mean_component_area = _component_area_stats(labels, count)
    area = float(np.count_nonzero(mask_bool))
    return {
        "labels": labels,
        "component_count": int(count),
        "area_fraction": float(area / max(1, mask_bool.size)),
        "mean_component_area": mean_component_area,
        "has_skeleton": bool(include_skeleton),
        "boundary": boundary,
        "boundary_dist": boundary_dist,
        "skeleton": skeleton,
        "skeleton_neighbors": skeleton_neighbors,
        "skeleton_length": float(np.count_nonzero(skeleton)),
        "endpoint_count": endpoint_count,
        "branchpoint_count": branchpoint_count,
    }


def _tiny_component_map_from_structure(structure: dict[str, object], area_threshold: int = 12) -> np.ndarray:
    labels = np.asarray(structure["labels"], dtype=np.int32)
    count = int(structure["component_count"])
    result = np.zeros(labels.shape, dtype=np.float32)
    if count <= 0:
        return result
    area_counts = np.bincount(labels.ravel(), minlength=count + 1)
    small_labels = np.flatnonzero(area_counts[1 : count + 1] <= int(area_threshold)) + 1
    for label_id in small_labels:
        result[labels == int(label_id)] = 1.0
    return result


def _tiny_component_map(mask: np.ndarray, area_threshold: int = 12) -> np.ndarray:
    return _tiny_component_map_from_structure(_mask_structure(mask), area_threshold=area_threshold)


def _thin_bridge_map_from_structure(structure: dict[str, object]) -> np.ndarray:
    skeleton = np.asarray(structure["skeleton"], dtype=bool)
    if not np.any(skeleton):
        return np.zeros_like(skeleton, dtype=np.float32)
    neighbors = np.asarray(structure["skeleton_neighbors"], dtype=np.uint8)
    bridge = skeleton & (neighbors == 2)
    return np.asarray(_binary_dilate(bridge, radius=1), dtype=np.float32)


def _thin_bridge_map(mask: np.ndarray) -> np.ndarray:
    return _thin_bridge_map_from_structure(_mask_structure(mask))


def _branchpoint_map_from_structure(structure: dict[str, object]) -> np.ndarray:
    skeleton = np.asarray(structure["skeleton"], dtype=bool)
    neighbors = np.asarray(structure["skeleton_neighbors"], dtype=np.uint8)
    return np.asarray(_binary_dilate(skeleton & (neighbors >= 3), radius=1), dtype=np.float32)


def _branchpoint_map(mask: np.ndarray) -> np.ndarray:
    return _branchpoint_map_from_structure(_mask_structure(mask))


def _endpoint_map_from_structure(structure: dict[str, object]) -> np.ndarray:
    skeleton = np.asarray(structure["skeleton"], dtype=bool)
    neighbors = np.asarray(structure["skeleton_neighbors"], dtype=np.uint8)
    return np.asarray(_binary_dilate(skeleton & (neighbors == 1), radius=1), dtype=np.float32)


def _endpoint_map(mask: np.ndarray) -> np.ndarray:
    return _endpoint_map_from_structure(_mask_structure(mask))
