"""pipeline: watershed split."""
from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from contour.utils import ensure_binary_mask, ensure_uint8

def _odd(value: int, minimum: int = 1) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1

def _operation_timing_bucket(operation: str) -> str:
    if operation in {"threshold", "adaptive_threshold", "otsu_threshold", "edge_guided_threshold", "color_binarize"}:
        return "threshold_ms"
    if operation in {"morph_open", "morph_close", "erode", "dilate", "gradient", "tophat", "blackhat"}:
        return "morphology_ms"
    return "postprocessing_ms"

def _threshold_mode(name: str) -> int:
    mapping = {
        "binary": cv2.THRESH_BINARY,
        "binary_inv": cv2.THRESH_BINARY_INV,
        "trunc": cv2.THRESH_TRUNC,
        "tozero": cv2.THRESH_TOZERO,
        "tozero_inv": cv2.THRESH_TOZERO_INV,
    }
    return mapping.get(name, cv2.THRESH_BINARY)

def _adaptive_mode(name: str) -> int:
    return cv2.ADAPTIVE_THRESH_GAUSSIAN_C if name == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C

def _morph_shape(name: str) -> int:
    mapping = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    return mapping.get(name, cv2.MORPH_RECT)

def _resize_interpolation(name: str) -> int:
    mapping = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    return mapping.get(name, cv2.INTER_LINEAR)

def _as_gray(image: np.ndarray) -> np.ndarray:
    data = ensure_uint8(image)
    if data.ndim == 3 and data.shape[2] == 4:
        return cv2.cvtColor(data, cv2.COLOR_BGRA2GRAY)
    if data.ndim == 3:
        return cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    return data

def _as_bgr(image: np.ndarray) -> np.ndarray:
    data = ensure_uint8(image)
    if data.ndim == 2:
        return cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    if data.ndim == 3 and data.shape[2] == 4:
        return cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
    return data

def _color_selection_entries(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    raw_entries = parameters.get("selected_colors", [])
    if not isinstance(raw_entries, list):
        return []
    result: list[dict[str, Any]] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        rgb = entry.get("rgb")
        if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
            continue
        try:
            parsed_rgb = [max(0, min(255, int(channel))) for channel in rgb]
        except (TypeError, ValueError):
            continue
        result.append(
            {
                "rgb": parsed_rgb,
                "enabled": bool(entry.get("enabled", True)),
            }
        )
    return result

def _color_binarize(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    bgr = _as_bgr(image)
    delta = max(0, min(255, int(parameters.get("delta", 10))))
    color_entries = [entry for entry in _color_selection_entries(parameters) if entry.get("enabled", True)]
    if not color_entries:
        return np.zeros(bgr.shape[:2], dtype=np.uint8)
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    rgb_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    for entry in color_entries:
        rgb = np.asarray(entry["rgb"], dtype=np.int16)
        lower = np.clip(rgb - delta, 0, 255).astype(np.uint8)
        upper = np.clip(rgb + delta, 0, 255).astype(np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(rgb_image, lower, upper))
    return mask

def _binary_fill_holes(image: np.ndarray, _parameters: dict[str, Any]) -> np.ndarray:
    mask = ensure_binary_mask(image)
    flood = mask.copy()
    h, w = mask.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)

def _binary_filter_components(image: np.ndarray, parameters: dict[str, Any], *, metric: str) -> np.ndarray:
    mask = ensure_binary_mask(image)
    num_labels, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    result = np.zeros_like(mask)
    minimum = float(parameters.get(f"min_component_{metric}", 0.0) or 0.0)
    maximum_raw = float(parameters.get(f"max_component_{metric}", 0.0) or 0.0)
    maximum = maximum_raw if maximum_raw > 0 else None
    for label_index in range(1, num_labels):
        component_mask = np.where(labels == label_index, 255, 0).astype(np.uint8)
        if metric == "area":
            value = float(cv2.countNonZero(component_mask))
        else:
            contours, _hierarchy = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            value = float(sum(cv2.arcLength(contour, True) for contour in contours))
        if value < minimum:
            continue
        if maximum is not None and value > maximum:
            continue
        result[labels == label_index] = 255
    return result

def _binary_filter_area(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    return _binary_filter_components(image, parameters, metric="area")

def _binary_filter_perimeter(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    return _binary_filter_components(image, parameters, metric="perimeter")

def _watershed_split(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    mask = ensure_binary_mask(image)
    if cv2.countNonZero(mask) == 0:
        return mask

    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    if max_distance <= 0.0:
        return mask

    distance_ratio = min(0.95, max(0.05, float(parameters.get("distance_ratio", 0.35))))
    local_max_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    local_max = distance >= cv2.dilate(distance, local_max_kernel, iterations=1) - 1e-6
    sure_fg = np.where(local_max & (distance >= (max_distance * distance_ratio)), 255, 0).astype(np.uint8)

    min_peak_area = max(0, int(parameters.get("min_peak_area", 0) or 0))
    if min_peak_area > 0:
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            (sure_fg > 0).astype(np.uint8), connectivity=8
        )
        filtered = np.zeros_like(sure_fg)
        for label_index in range(1, count):
            if int(stats[label_index, cv2.CC_STAT_AREA]) >= min_peak_area:
                filtered[labels == label_index] = 255
        sure_fg = filtered

    marker_count, _marker_labels = cv2.connectedComponents((sure_fg > 0).astype(np.uint8), connectivity=8)
    if marker_count <= 2:
        return mask

    kernel = _kernel(parameters)
    sure_bg = cv2.dilate(mask, kernel, iterations=max(1, int(parameters.get("background_iterations", 1))))
    unknown = cv2.subtract(sure_bg, sure_fg)

    _marker_count, markers = cv2.connectedComponents((sure_fg > 0).astype(np.uint8), connectivity=8)
    markers = markers.astype(np.int32) + 1
    markers[unknown > 0] = 0

    marker_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(marker_image, markers)

    result = np.zeros_like(mask)
    result[markers > 1] = 255
    return result

def _kernel(parameters: dict[str, Any]) -> np.ndarray:
    size = _odd(int(parameters.get("kernel_size", 3)), minimum=1)
    shape = _morph_shape(str(parameters.get("shape", "rect")))
    return cv2.getStructuringElement(shape, (size, size))

def _gaussian_blur(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    kernel_size = _odd(int(parameters.get("kernel_size", 5)), minimum=1)
    sigma_x = float(parameters.get("sigma_x", 0.0))
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x)

def _median_blur(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    return cv2.medianBlur(image, _odd(int(parameters.get("kernel_size", 5)), minimum=3))

def _bilateral(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    return cv2.bilateralFilter(
        image,
        d=max(1, int(parameters.get("diameter", 9))),
        sigmaColor=float(parameters.get("sigma_color", 75.0)),
        sigmaSpace=float(parameters.get("sigma_space", 75.0)),
    )
