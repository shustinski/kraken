from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from ..processing import BatchImageResult, ContourExtractionSettings, DisplaySettings, SaveOptions
from ...contour_extractor import extract_polygons
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
    column_normalized = _normalize_columns(gray)
    intensity_mask = cv2.bitwise_or(_range_mask(gray, low, high), _range_mask(column_normalized, low, high))

    expected_span = _expected_via_span(settings)
    background_size = _odd_kernel_size(expected_span * 2.5, minimum=9)
    background_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (background_size, background_size))
    smoothed = cv2.medianBlur(column_normalized, 3)
    operation = cv2.MORPH_TOPHAT if bright else cv2.MORPH_BLACKHAT
    response = cv2.morphologyEx(smoothed, operation, background_kernel)

    nonzero = response[response > 0]
    if nonzero.size:
        cutoff = max(6.0, min(45.0, float(np.percentile(nonzero, 92)) * 0.45))
        response_mask = np.where(response >= cutoff, 255, 0).astype(np.uint8)
    else:
        response_mask = np.zeros_like(gray, dtype=np.uint8)

    seed_mask = cv2.bitwise_and(response_mask, cv2.dilate(intensity_mask, np.ones((3, 3), dtype=np.uint8), iterations=1))
    if cv2.countNonZero(seed_mask) == 0:
        seed_mask = intensity_mask

    support_size = _odd_kernel_size(max(3, expected_span * 1.2), minimum=3)
    support_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (support_size, support_size))
    support_mask = cv2.bitwise_and(intensity_mask, cv2.dilate(seed_mask, support_kernel, iterations=1))
    if cv2.countNonZero(support_mask) > 0:
        seed_mask = support_mask

    object_size = _odd_kernel_size(max(3, expected_span * 0.35), minimum=3)
    object_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (object_size, object_size))
    mask = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, object_kernel, iterations=1)
    return ensure_binary_mask(mask)


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
    polygons = extract_polygons(mask, contour_settings)
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
