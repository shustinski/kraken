"""pipeline: edge contour refined mask."""
from __future__ import annotations
from typing import Any
import cv2
import numpy as np

from contour.pipeline_parts.watershed_split import (
    _adaptive_mode,
    _as_gray,
    _odd,
    _threshold_mode,
)

def _clahe(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    image = _as_gray(image)
    tile_grid = max(1, int(parameters.get("tile_grid_size", 8)))
    clahe = cv2.createCLAHE(
        clipLimit=max(0.1, float(parameters.get("clip_limit", 2.0))),
        tileGridSize=(tile_grid, tile_grid),
    )
    return clahe.apply(image)

def _histogram_equalization(image: np.ndarray, _: dict[str, Any]) -> np.ndarray:
    return cv2.equalizeHist(_as_gray(image))

def _brightness_contrast(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    return cv2.convertScaleAbs(
        image,
        alpha=float(parameters.get("alpha", 1.0)),
        beta=float(parameters.get("beta", 0.0)),
    )

def _brightness(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    return cv2.convertScaleAbs(
        image,
        alpha=1.0,
        beta=float(parameters.get("brightness", 0.0)),
    )

def _contrast(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    return cv2.convertScaleAbs(
        image,
        alpha=float(parameters.get("contrast", 1.0)),
        beta=0.0,
    )

def _gamma_correction(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    gamma = max(0.01, float(parameters.get("gamma", 1.0)))
    table = np.array([((value / 255.0) ** (1.0 / gamma)) * 255 for value in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)

def _threshold(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    image = _as_gray(image)
    _, result = cv2.threshold(
        image,
        float(parameters.get("threshold", 127)),
        float(parameters.get("max_value", 255)),
        _threshold_mode(str(parameters.get("threshold_type", "binary"))),
    )
    return result

def _adaptive_threshold(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    image = _as_gray(image)
    return cv2.adaptiveThreshold(
        image,
        float(parameters.get("max_value", 255)),
        _adaptive_mode(str(parameters.get("adaptive_method", "gaussian"))),
        _threshold_mode(str(parameters.get("threshold_type", "binary"))),
        _odd(int(parameters.get("block_size", 11)), minimum=3),
        float(parameters.get("c_value", 2.0)),
    )

def _otsu_threshold(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    image = _as_gray(image)
    _, result = cv2.threshold(
        image,
        0,
        255,
        _threshold_mode(str(parameters.get("threshold_type", "binary"))) | cv2.THRESH_OTSU,
    )
    return result

def _binary_threshold_mode(parameters: dict[str, Any]) -> int:
    threshold_type = str(parameters.get("threshold_type", "binary"))
    if threshold_type == "binary_inv":
        return cv2.THRESH_BINARY_INV
    return cv2.THRESH_BINARY

def _base_threshold_mask(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    threshold_mode = str(parameters.get("threshold_mode", "otsu"))
    threshold_type = _binary_threshold_mode(parameters)
    max_value = float(parameters.get("max_value", 255))
    if threshold_mode == "adaptive":
        return cv2.adaptiveThreshold(
            image,
            max_value,
            _adaptive_mode(str(parameters.get("adaptive_method", "gaussian"))),
            threshold_type,
            _odd(int(parameters.get("block_size", 11)), minimum=3),
            float(parameters.get("c_value", 2.0)),
        )
    if threshold_mode == "manual":
        _, result = cv2.threshold(
            image,
            float(parameters.get("threshold", 127)),
            max_value,
            threshold_type,
        )
        return result
    _, result = cv2.threshold(image, 0, max_value, threshold_type | cv2.THRESH_OTSU)
    return result

def _edge_elevation(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    detector = str(parameters.get("edge_detector", "canny"))
    aperture_size = max(3, min(7, _odd(int(parameters.get("aperture_size", 3)), minimum=3)))
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    if detector == "canny":
        edges = cv2.Canny(
            blurred,
            float(parameters.get("threshold1", 50)),
            float(parameters.get("threshold2", 150)),
            apertureSize=aperture_size,
            L2gradient=bool(parameters.get("l2gradient", False)),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        return cv2.GaussianBlur(edges, (3, 3), 0)

    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=aperture_size)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=aperture_size)
    magnitude = cv2.magnitude(grad_x, grad_y)
    if float(magnitude.max()) <= 0.0:
        return np.zeros_like(image, dtype=np.uint8)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def _edge_mask(image: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    detector = str(parameters.get("edge_detector", "canny"))
    if detector == "canny":
        aperture_size = max(3, min(7, _odd(int(parameters.get("aperture_size", 3)), minimum=3)))
        return cv2.Canny(
            cv2.GaussianBlur(image, (3, 3), 0),
            float(parameters.get("threshold1", 50)),
            float(parameters.get("threshold2", 150)),
            apertureSize=aperture_size,
            L2gradient=bool(parameters.get("l2gradient", False)),
        )

    elevation = _edge_elevation(image, parameters)
    if cv2.countNonZero(elevation) == 0:
        return np.zeros_like(image, dtype=np.uint8)
    nonzero = elevation[elevation > 0]
    percentile = max(1.0, min(99.0, float(parameters.get("edge_percentile", 80.0))))
    threshold_value = float(np.percentile(nonzero, percentile)) if nonzero.size else 255.0
    return np.where(elevation >= threshold_value, 255, 0).astype(np.uint8)

def _edge_contour_refined_mask(
    image: np.ndarray,
    base_mask: np.ndarray,
    correction_band: np.ndarray,
    correction_radius: int,
    parameters: dict[str, Any],
) -> np.ndarray | None:
    edges = _edge_mask(image, parameters)
    if cv2.countNonZero(edges) == 0:
        return None

    close_radius = max(1, correction_radius)
    kernel_size = close_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _hierarchy = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    edge_regions = np.zeros_like(base_mask)
    base_area = float(cv2.countNonZero(base_mask))
    min_overlap = max(1.0, base_area * 0.02)
    for contour in contours:
        if contour is None or len(contour) < 3:
            continue
        candidate = np.zeros_like(base_mask)
        cv2.fillPoly(candidate, [contour], 255)
        overlap = float(cv2.countNonZero(cv2.bitwise_and(candidate, base_mask)))
        if overlap < min_overlap:
            continue
        edge_regions = cv2.bitwise_or(edge_regions, candidate)

    if cv2.countNonZero(edge_regions) == 0:
        return None

    result = base_mask.copy()
    result[correction_band > 0] = edge_regions[correction_band > 0]
    if cv2.countNonZero(result) == 0:
        return None
    return result
