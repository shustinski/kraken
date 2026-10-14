"""repository: label components."""
from __future__ import annotations
import numpy as np
from skimage.morphology import skeletonize as skimage_skeletonize
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    POLYGON_CONFIDENCE_COMPLETION_AXIS_RATIO,
    POLYGON_CONFIDENCE_COMPLETION_MAJOR_SCALE,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import _NEIGHBOR_SLICE_ORDER

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.mean_filter_3x3 import resize_grayscale_image

def _resize_like(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if tuple(int(v) for v in array.shape) == tuple(int(v) for v in target_shape):
        return np.asarray(array)
    return np.asarray(resize_grayscale_image(np.asarray(array, dtype=np.uint8), target_shape))

def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.size == 0 or not np.any(mask_bool):
        return np.zeros_like(mask_bool, dtype=np.int32), 0
    if ndi is not None:
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

def _binary_erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask_bool.copy()
    if ndi is not None:
        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        return np.asarray(ndi.binary_erosion(mask_bool, structure=structure), dtype=bool)
    padded = np.pad(mask_bool, radius, mode="constant", constant_values=False)
    result = np.ones_like(mask_bool, dtype=bool)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            result &= padded[row_offset:row_offset + mask_bool.shape[0], column_offset:column_offset + mask_bool.shape[1]]
    return result

def _binary_dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask_bool.copy()
    if ndi is not None:
        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        return np.asarray(ndi.binary_dilation(mask_bool, structure=structure), dtype=bool)
    padded = np.pad(mask_bool, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask_bool)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            result |= padded[row_offset:row_offset + mask_bool.shape[0], column_offset:column_offset + mask_bool.shape[1]]
    return result

def _binary_dilate_rect(mask: np.ndarray, radius_y: int = 1, radius_x: int = 1) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    ry = max(0, int(radius_y))
    rx = max(0, int(radius_x))
    if ry <= 0 and rx <= 0:
        return mask_bool.copy()
    if ndi is not None:
        structure = np.ones((2 * ry + 1, 2 * rx + 1), dtype=bool)
        return np.asarray(ndi.binary_dilation(mask_bool, structure=structure), dtype=bool)
    padded = np.pad(mask_bool, ((ry, ry), (rx, rx)), mode="constant", constant_values=False)
    result = np.zeros_like(mask_bool)
    for row_offset in range(2 * ry + 1):
        for column_offset in range(2 * rx + 1):
            result |= padded[row_offset:row_offset + mask_bool.shape[0], column_offset:column_offset + mask_bool.shape[1]]
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
    if ndi is not None:
        return np.asarray(ndi.distance_transform_edt(mask_bool), dtype=np.float32)
    if cv2 is not None:
        distances = cv2.distanceTransform(np.asarray(mask_bool, dtype=np.uint8), cv2.DIST_L2, 5)
        return np.asarray(distances, dtype=np.float32)
    raise RuntimeError("Distance transform backend is unavailable")

def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    image = np.asarray(mask, dtype=np.uint8)
    if cv2 is not None:
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        return cv2.filter2D(image, cv2.CV_8U, kernel, borderType=cv2.BORDER_CONSTANT)
    padded = np.pad(image, 1, mode="constant", constant_values=0)
    neighbors = np.zeros_like(mask, dtype=np.uint8)
    for row_slice, column_slice in _NEIGHBOR_SLICE_ORDER:
        neighbors += padded[row_slice, column_slice]
    return neighbors

def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Return the canonical scikit-image Zhang skeleton.

    The wrapper keeps the validation module's existing local API while avoiding
    a second, subtly different implementation of the thinning algorithm.
    """
    image = np.asarray(mask, dtype=bool)
    if image.size == 0:
        return image.copy()
    return np.asarray(skimage_skeletonize(image, method="zhang"), dtype=bool)

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
    component_areas = [float(value) for value in area_counts[1:count + 1]]
    mean_component_area = float(np.mean(component_areas, dtype=np.float64)) if component_areas else 0.0
    return component_areas, mean_component_area
