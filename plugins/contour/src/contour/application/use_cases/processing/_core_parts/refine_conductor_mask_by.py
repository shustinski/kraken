"""_core: refine conductor mask by."""
from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from contour.edge_detection import (
    build_gradient_elevation,
    normalize_edge_method,
)
from contour.utils import ensure_binary_mask, ensure_uint8
from contour.application.preview_cancellation import raise_if_preview_cancelled
from contour.application.processing import (
    ContourExtractionSettings,
)

from contour.application.use_cases.processing._core_parts.extract_response_peaks import _log_blob_response

def _refine_conductor_mask_by_gradient(
    source_image: Any, base_mask: np.ndarray, settings: ContourExtractionSettings
) -> np.ndarray:
    binary = ensure_binary_mask(base_mask)
    correction_radius = max(0, int(settings.conductor_gradient_band_radius))
    if correction_radius <= 0 or cv2.countNonZero(binary) == 0:
        return binary

    gray = _via_grayscale(source_image)
    elevation = _conductor_gradient_elevation(gray, settings)
    if cv2.countNonZero(elevation) == 0:
        return binary
    raise_if_preview_cancelled()

    kernel_size = correction_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    correction_band = cv2.bitwise_xor(dilated, eroded)
    if cv2.countNonZero(cv2.bitwise_and(elevation, correction_band)) == 0:
        return binary

    foreground_seed = cv2.erode(binary, kernel, iterations=1)
    background_seed = cv2.erode(cv2.bitwise_not(binary), kernel, iterations=1)
    if cv2.countNonZero(foreground_seed) == 0 or cv2.countNonZero(background_seed) == 0:
        return binary

    markers = np.zeros(binary.shape[:2], dtype=np.int32)
    markers[background_seed > 0] = 1
    _component_count, foreground_labels = cv2.connectedComponents(
        (foreground_seed > 0).astype(np.uint8), connectivity=8
    )
    markers[foreground_labels > 0] = foreground_labels[foreground_labels > 0] + 1

    marker_image = cv2.cvtColor(elevation, cv2.COLOR_GRAY2BGR)
    raise_if_preview_cancelled()
    markers = cv2.watershed(marker_image, markers)
    raise_if_preview_cancelled()
    refined = np.where(markers > 1, 255, 0).astype(np.uint8)
    result = binary.copy()
    result[correction_band > 0] = refined[correction_band > 0]
    return ensure_binary_mask(result)

def _conductor_gradient_elevation(image: np.ndarray, settings: ContourExtractionSettings) -> np.ndarray:
    gray = ensure_uint8(image)
    if gray.ndim != 2:
        gray = _via_grayscale(gray)
    method = _resolve_conductor_edge_method(settings)
    elevation = build_gradient_elevation(gray, method)
    if elevation.size == 0 or int(elevation.max()) <= 0:
        return np.zeros_like(gray, dtype=np.uint8)
    min_strength = max(0.0, float(settings.conductor_gradient_min_strength))
    if min_strength <= 0.0:
        return elevation
    elevation = elevation.copy()
    weak = elevation < min_strength
    elevation[weak] = np.clip(elevation[weak].astype(np.float32) * 0.25, 0.0, 255.0).astype(np.uint8)
    return elevation

def _resolve_conductor_edge_method(settings: ContourExtractionSettings) -> str:
    preferred = settings.conductor_gradient_edge_method or settings.edge_method
    return normalize_edge_method(preferred)

def _resolve_via_edge_method(settings: ContourExtractionSettings) -> str:
    preferred = settings.via_gradient_edge_method or settings.edge_method
    return normalize_edge_method(preferred)

def _via_grayscale(image: Any) -> np.ndarray:
    data = ensure_uint8(image)
    if data.ndim == 3 and data.shape[2] == 4:
        return cv2.cvtColor(data, cv2.COLOR_BGRA2GRAY)
    if data.ndim == 3:
        return cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    return data

def _range_mask(gray: np.ndarray, low: int, high: int) -> np.ndarray:
    return np.where((gray >= low) & (gray <= high), 255, 0).astype(np.uint8)

def _odd_kernel_size(value: float, *, minimum: int = 3) -> int:
    size = max(minimum, round(value))
    if size % 2 == 0:
        size += 1
    return size

def _expected_via_span(settings: ContourExtractionSettings) -> int:
    sizes: list[int] = []
    sizes.extend(int(value) for value in settings.fixed_via_widths if int(value) > 0)
    sizes.extend(int(value) for value in settings.fixed_via_heights if int(value) > 0)
    for value in (settings.min_via_width, settings.min_via_height):
        if int(value) > 0:
            sizes.append(int(value))
    for value in (settings.max_via_width, settings.max_via_height):
        if value is not None and int(value) > 0:
            sizes.append(int(value))
    if not sizes:
        return 7
    return max(3, round(float(np.median(np.asarray(sizes, dtype=np.float32)))))

def _clahe_gray(gray: np.ndarray) -> np.ndarray:
    data = ensure_uint8(gray)
    tile = max(4, min(16, _odd_kernel_size(min(data.shape[:2]) / 8.0, minimum=5)))
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile)).apply(data)

def _normalize_columns(gray: np.ndarray) -> np.ndarray:
    """Column-wise min-max normalization (vectorized)."""

    data = ensure_uint8(gray)
    if data.size == 0 or data.ndim != 2:
        return data
    columns = data.astype(np.float32)
    column_min = columns.min(axis=0, keepdims=True)
    column_max = columns.max(axis=0, keepdims=True)
    span = column_max - column_min
    safe = span > 1e-6
    normalized = columns.copy()
    if np.any(safe):
        scale = np.where(safe, 255.0 / np.where(safe, span, 1.0), 0.0)
        normalized = (columns - column_min) * scale
    return np.clip(normalized, 0.0, 255.0).astype(np.uint8)

def _via_polarity_scans(settings: ContourExtractionSettings) -> list[tuple[int, int, bool]]:
    scans: list[tuple[int, int, bool]] = []
    if settings.via_white_range_enabled:
        low = max(0, min(255, int(settings.via_white_range_min)))
        high = max(0, min(255, int(settings.via_white_range_max)))
        if low > high:
            low, high = high, low
        scans.append((low, high, True))
    if settings.via_black_range_enabled:
        low = max(0, min(255, int(settings.via_black_range_min)))
        high = max(0, min(255, int(settings.via_black_range_max)))
        if low > high:
            low, high = high, low
        scans.append((low, high, False))
    if not scans:
        scans = [(0, 255, True), (0, 255, False)]
    return scans

def _intensity_mask(images: list[np.ndarray], low: int, high: int) -> np.ndarray:
    if not images:
        return np.zeros((1, 1), dtype=np.uint8)
    mask = np.zeros_like(images[0], dtype=np.uint8)
    for image in images:
        mask = cv2.bitwise_or(mask, _range_mask(image, low, high))
    return mask

def _intensity_gate(intensity_mask: np.ndarray, expected_span: int) -> np.ndarray:
    if cv2.countNonZero(intensity_mask) == 0:
        return np.full_like(intensity_mask, 255, dtype=np.uint8)
    open_size = _odd_kernel_size(max(3, expected_span * 0.35), minimum=3)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    cleaned = cv2.morphologyEx(intensity_mask, cv2.MORPH_OPEN, open_kernel)
    if cv2.countNonZero(cleaned) == 0:
        # All-noise intensity mask would otherwise collapse to nothing and the
        # detector would return no candidates. Fall back to the raw mask.
        cleaned = intensity_mask
    dilate_size = _odd_kernel_size(max(3, expected_span * 0.6), minimum=3)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    return cv2.dilate(cleaned, dilate_kernel, iterations=1)

def _multiscale_blob_response(
    gray: np.ndarray,
    expected_span: int,
    *,
    bright: bool,
    line_suppression: float,
) -> np.ndarray:
    """Combined TopHat/BlackHat (3 scales) and LoG (3 sigmas) response, ``uint8``."""

    data = ensure_uint8(gray)
    if data.size == 0:
        return np.zeros_like(data, dtype=np.uint8)
    smoothed = cv2.GaussianBlur(data, (3, 3), 0)
    operation = cv2.MORPH_TOPHAT if bright else cv2.MORPH_BLACKHAT
    response = np.zeros_like(data, dtype=np.uint8)
    for scale in (1.6, 2.4, 3.2):
        kernel_size = _odd_kernel_size(expected_span * scale, minimum=5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        local = cv2.morphologyEx(smoothed, operation, kernel)
        response = cv2.max(response, local)
    response = cv2.max(response, _log_blob_response(smoothed, expected_span, bright=bright))

    if line_suppression > 0.0:
        line_length = _odd_kernel_size(max(9, expected_span * 4.0), minimum=9)
        line_width = _odd_kernel_size(max(3, expected_span * 0.45), minimum=3)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_length, line_width))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_width, line_length))
        horizontal_lines = cv2.morphologyEx(response, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(response, cv2.MORPH_OPEN, vertical_kernel)
        line_response = cv2.max(horizontal_lines, vertical_lines)
        response = cv2.subtract(response, cv2.convertScaleAbs(line_response, alpha=line_suppression))
    return ensure_uint8(response)
