from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from ..processing import BatchImageResult, ContourExtractionSettings, DisplaySettings, SaveOptions
from ...contour_extractor import extract_polygons, extract_polygons_with_debug
from ...pipeline import PreprocessingPipeline
from ...serializers import save_result_bundle
from ...utils import ensure_binary_mask, ensure_uint8, load_image_color


@dataclass(frozen=True, slots=True)
class PreviewProcessingRequest:
    image_path: str
    pipeline_config: dict[str, Any]
    contour_settings: ContourExtractionSettings
    source_image: Any | None = None
    preprocessed_image: Any | None = None


@dataclass(frozen=True, slots=True)
class PreparedImageRequest:
    image_path: str
    source_image: Any
    pipeline_config: dict[str, Any]


def build_preview_request_signature(request: PreviewProcessingRequest) -> tuple[str, str, str]:
    return (
        request.image_path,
        json.dumps(request.pipeline_config, ensure_ascii=False, sort_keys=True),
        json.dumps(request.contour_settings.to_dict(), ensure_ascii=False, sort_keys=True),
    )


def build_prepared_image_signature(request: PreparedImageRequest) -> tuple[str, str]:
    return (
        request.image_path,
        json.dumps(request.pipeline_config, ensure_ascii=False, sort_keys=True),
    )


def prepare_image_for_preview(source_image: Any, pipeline_config: dict[str, Any]) -> Any:
    return PreprocessingPipeline.from_dict(pipeline_config).apply(source_image)


def apply_via_vectorization_mask(image: Any, settings: ContourExtractionSettings) -> Any:
    if settings.object_type != "via" and settings.output_mode != "box":
        return ensure_binary_mask(image)

    gray = _via_grayscale(image)
    masks: list[np.ndarray] = []
    if settings.via_white_range_enabled:
        low = max(0, min(255, int(settings.via_white_range_min)))
        high = max(0, min(255, int(settings.via_white_range_max)))
        if low > high:
            low, high = high, low
        masks.append(_via_local_range_mask(gray, low, high, settings, bright=True))
    if settings.via_black_range_enabled:
        low = max(0, min(255, int(settings.via_black_range_min)))
        high = max(0, min(255, int(settings.via_black_range_max)))
        if low > high:
            low, high = high, low
        masks.append(_via_local_range_mask(gray, low, high, settings, bright=False))
    if not masks:
        return ensure_binary_mask(image)

    result = np.zeros_like(gray, dtype=np.uint8)
    for mask in masks:
        result = cv2.bitwise_or(result, ensure_binary_mask(mask))
    return result


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
    size = max(minimum, int(round(value)))
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
    return max(3, int(round(float(np.median(np.asarray(sizes, dtype=np.float32))))))


def _via_local_range_mask(gray: np.ndarray, low: int, high: int, settings: ContourExtractionSettings, *, bright: bool) -> np.ndarray:
    expected_span = _expected_via_span(settings)
    column_normalized = _normalize_columns(gray)
    row_normalized = _normalize_rows(gray)
    clahe_gray = _clahe_gray(gray)
    clahe_columns = _clahe_gray(column_normalized)

    intensity_mask = np.zeros_like(gray, dtype=np.uint8)
    for candidate in (gray, column_normalized):
        intensity_mask = cv2.bitwise_or(intensity_mask, _range_mask(candidate, low, high))

    response = _via_multiscale_response(
        [gray, column_normalized, row_normalized, clahe_gray, clahe_columns],
        expected_span,
        bright=bright,
    )
    response_mask = _response_threshold_mask(response, intensity_mask)

    seed_mask = cv2.bitwise_and(response_mask, cv2.dilate(intensity_mask, np.ones((3, 3), dtype=np.uint8), iterations=1))
    if cv2.countNonZero(seed_mask) == 0:
        seed_mask = intensity_mask

    support_size = _odd_kernel_size(max(3, expected_span * 1.35), minimum=3)
    support_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (support_size, support_size))
    support_mask = cv2.bitwise_and(intensity_mask, cv2.dilate(seed_mask, support_kernel, iterations=1))
    if cv2.countNonZero(support_mask) > 0:
        seed_mask = support_mask

    object_size = _odd_kernel_size(max(3, expected_span * 0.35), minimum=3)
    object_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (object_size, object_size))
    mask = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, object_kernel, iterations=1)
    return ensure_binary_mask(mask)


def _clahe_gray(gray: np.ndarray) -> np.ndarray:
    data = ensure_uint8(gray)
    tile = max(4, min(16, _odd_kernel_size(min(data.shape[:2]) / 8.0, minimum=5)))
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile)).apply(data)


def _normalize_rows(gray: np.ndarray) -> np.ndarray:
    data = ensure_uint8(gray)
    normalized = np.zeros_like(data, dtype=np.uint8)
    for row_index in range(data.shape[0]):
        row = data[row_index, :].astype(np.float32)
        minimum = float(row.min())
        maximum = float(row.max())
        if maximum - minimum < 1e-6:
            normalized[row_index, :] = data[row_index, :]
            continue
        normalized[row_index, :] = np.clip((row - minimum) * (255.0 / (maximum - minimum)), 0, 255).astype(np.uint8)
    return normalized


def _via_multiscale_response(images: list[np.ndarray], expected_span: int, *, bright: bool) -> np.ndarray:
    response = np.zeros_like(images[0], dtype=np.uint8)
    kernel_scales = (1.7, 2.5, 3.4)
    operation = cv2.MORPH_TOPHAT if bright else cv2.MORPH_BLACKHAT
    for image in images:
        smoothed = cv2.medianBlur(ensure_uint8(image), 3)
        for scale in kernel_scales:
            kernel_size = _odd_kernel_size(expected_span * scale, minimum=7)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            local = cv2.morphologyEx(smoothed, operation, kernel)
            response = cv2.max(response, local)
        response = cv2.max(response, _local_zscore_response(smoothed, expected_span, bright=bright))
    return response


def _local_zscore_response(gray: np.ndarray, expected_span: int, *, bright: bool) -> np.ndarray:
    data = ensure_uint8(gray).astype(np.float32)
    window = _odd_kernel_size(expected_span * 3.0, minimum=9)
    mean = cv2.blur(data, (window, window))
    mean_square = cv2.blur(data * data, (window, window))
    variance = np.maximum(mean_square - mean * mean, 1.0)
    std = np.sqrt(variance)
    delta = data - mean if bright else mean - data
    zscore = np.clip((delta / std) * 42.0, 0.0, 255.0)
    return zscore.astype(np.uint8)


def _response_threshold_mask(response: np.ndarray, intensity_mask: np.ndarray) -> np.ndarray:
    response_values = response[response > 0]
    if response_values.size == 0:
        return np.zeros_like(response, dtype=np.uint8)
    in_range_values = response[intensity_mask > 0]
    sample = in_range_values[in_range_values > 0]
    if sample.size == 0:
        sample = response_values
    cutoff = max(5.0, min(52.0, float(np.percentile(sample, 82)) * 0.55))
    return np.where(response >= cutoff, 255, 0).astype(np.uint8)


def process_image_path(
    image_path: str,
    pipeline_config: dict[str, Any],
    contour_settings: ContourExtractionSettings,
    output_directory: str | None = None,
    save_options: SaveOptions | None = None,
    display_settings: DisplaySettings | None = None,
    *,
    source_image: Any | None = None,
    preprocessed_image: Any | None = None,
    image_loader: Callable[[str], Any] = load_image_color,
    save_bundle: Callable[..., dict[str, str]] = save_result_bundle,
    include_images_in_result: bool = True,
) -> BatchImageResult:
    pipeline = PreprocessingPipeline.from_dict(pipeline_config)
    source = source_image if source_image is not None else image_loader(image_path)
    preprocessed = preprocessed_image if preprocessed_image is not None else pipeline.apply(source)
    mask = apply_via_vectorization_mask(preprocessed, contour_settings)
    if contour_settings.debug_enabled:
        polygons, debug_candidates = extract_polygons_with_debug(mask, contour_settings)
    else:
        polygons = extract_polygons(mask, contour_settings)
        debug_candidates = []
    saved_files: dict[str, str] = {}
    if output_directory:
        saved_files = save_bundle(
            output_directory=output_directory,
            image_path=image_path,
            polygons=polygons,
            source_image=source,
            display_settings=display_settings or DisplaySettings(),
            save_options=save_options or SaveOptions(),
            metadata={
                "contour_settings": contour_settings.to_dict(),
                "pipeline": pipeline_config,
            },
        )
    result_source = source if include_images_in_result else None
    result_preprocessed = preprocessed if include_images_in_result else None
    result_mask = mask if include_images_in_result else None
    return BatchImageResult(
        image_path=image_path,
        source_image=result_source,
        preprocessed_image=result_preprocessed,
        pipeline_config=dict(pipeline_config),
        mask_image=result_mask,
        polygons=polygons,
        debug_candidates=debug_candidates,
        saved_files=saved_files,
    )


def _via_binarization_channel(image: Any, channel_mode: str) -> np.ndarray:
    data = ensure_uint8(image)
    if data.ndim == 3 and data.shape[2] == 4:
        bgr = cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
    elif data.ndim == 3:
        bgr = data
    else:
        return _normalize_columns(data) if channel_mode == "columns" else data

    if channel_mode == "red_blue":
        blue = bgr[:, :, 0].astype(np.int16)
        red = bgr[:, :, 2].astype(np.int16)
        return np.abs(red - blue).astype(np.uint8)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return _normalize_columns(gray) if channel_mode == "columns" else gray


def _normalize_columns(gray: np.ndarray) -> np.ndarray:
    data = ensure_uint8(gray)
    normalized = np.zeros_like(data, dtype=np.uint8)
    for column_index in range(data.shape[1]):
        column = data[:, column_index].astype(np.float32)
        minimum = float(column.min())
        maximum = float(column.max())
        if maximum - minimum < 1e-6:
            normalized[:, column_index] = data[:, column_index]
            continue
        normalized[:, column_index] = np.clip((column - minimum) * (255.0 / (maximum - minimum)), 0, 255).astype(np.uint8)
    return normalized
