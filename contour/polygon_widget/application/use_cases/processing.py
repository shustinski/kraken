from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ...contour_extractor import extract_polygons
from ...edge_detection import (
    build_gradient_elevation,
    normalize_edge_method,
    phase_congruency,
    ridge_response,
    scharr_magnitude,
    structured_edges,
)
from ...pipeline import PreprocessingPipeline
from ...serializers import save_result_bundle
from ...utils import ensure_binary_mask, ensure_uint8, load_image_color
from ..processing import (
    BatchImageResult,
    ContourDebugCandidate,
    ContourExtractionSettings,
    DisplaySettings,
    SaveOptions,
)


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


@dataclass(frozen=True, slots=True)
class _ViaCandidate:
    center_x: float
    center_y: float
    width: float
    height: float
    score: float
    source: str
    roundness: float = 100.0
    reason: str = ""


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
    mask, _debug_candidates = build_via_vectorization_mask(image, settings)
    return mask


def build_detection_debug_maps(
    source_image: Any,
    preprocessed_image: Any,
    settings: ContourExtractionSettings,
    *,
    include_color_maps: bool = True,
) -> dict[str, np.ndarray]:
    """Return a dictionary of debug heatmaps produced by the detector.

    Keys always include:

    * ``"source_gray"`` – grayscale view of the source image.
    * ``"gradient_elevation"`` – gradient magnitude map used for detection.
    * ``"gradient_color"`` – coloured TURBO heatmap (if ``include_color_maps``).
    * ``"scharr"``, ``"phase_congruency"``, ``"structured"``, ``"ridge"`` –
      outputs of individual modern edge detectors for side-by-side
      inspection.
    * ``"mask"`` – final binary mask actually fed into ``extract_polygons``.
    * ``"spot_response"`` / ``"spot_response_dark"`` – multiscale top-hat /
      blob response used by the via detector (only populated for via flow).
    """

    maps: dict[str, np.ndarray] = {}
    if source_image is None:
        return maps

    source_gray = _via_grayscale(source_image)
    maps["source_gray"] = source_gray

    if settings.object_type == "via" or settings.output_mode == "box":
        edge_method = _resolve_via_edge_method(settings)
    else:
        edge_method = _resolve_conductor_edge_method(settings)

    elevation = build_gradient_elevation(source_gray, edge_method)
    maps["gradient_elevation"] = elevation
    if include_color_maps and elevation.size:
        maps["gradient_color"] = cv2.applyColorMap(elevation, cv2.COLORMAP_TURBO)

    maps["scharr"] = scharr_magnitude(source_gray)
    try:
        maps["phase_congruency"] = phase_congruency(source_gray)
    except Exception:  # pragma: no cover - numerical fallback
        maps["phase_congruency"] = np.zeros_like(source_gray, dtype=np.uint8)
    maps["structured"] = structured_edges(source_gray)
    maps["ridge"] = ridge_response(source_gray)

    if settings.object_type == "via" or settings.output_mode == "box":
        expected_span = _expected_via_span(settings)
        try:
            maps["spot_response"] = _via_spot_response(source_gray, expected_span, settings, bright=True)
            maps["spot_response_dark"] = _via_spot_response(source_gray, expected_span, settings, bright=False)
        except Exception:  # pragma: no cover - defensive
            pass
        mask, _candidates = build_via_vectorization_mask(preprocessed_image, settings)
        maps["mask"] = ensure_binary_mask(mask)
    else:
        mask = build_conductor_vectorization_mask(source_image, preprocessed_image, settings)
        maps["mask"] = ensure_binary_mask(mask)
        if include_color_maps:
            maps["conductor_gradient_elevation"] = _conductor_gradient_elevation(source_gray, settings)
    return maps


def build_conductor_vectorization_mask(
    source_image: Any,
    preprocessed_image: Any,
    settings: ContourExtractionSettings,
) -> np.ndarray:
    base_mask = ensure_binary_mask(preprocessed_image)
    if settings.object_type == "via" or settings.output_mode == "box" or not settings.conductor_gradient_enabled:
        return base_mask
    return _refine_conductor_mask_by_gradient(source_image, base_mask, settings)


def build_via_vectorization_mask(
    image: Any, settings: ContourExtractionSettings
) -> tuple[Any, list[ContourDebugCandidate]]:
    if settings.object_type != "via" and settings.output_mode != "box":
        return ensure_binary_mask(image), []

    gray = _via_grayscale(image)
    candidates: list[_ViaCandidate] = []
    rejected_candidates: list[_ViaCandidate] = []
    algorithm_methods_enabled = _via_algorithm_methods_enabled(settings)

    def add_detection_results(accepted: list[_ViaCandidate], rejected: list[_ViaCandidate]) -> None:
        candidates.extend(accepted)
        rejected_candidates.extend(rejected)

    if settings.via_white_range_enabled:
        low = max(0, min(255, int(settings.via_white_range_min)))
        high = max(0, min(255, int(settings.via_white_range_max)))
        if low > high:
            low, high = high, low
        add_detection_results(
            *_via_detect_candidates(
                gray,
                low,
                high,
                settings,
                bright=True,
                include_range_method=True,
                include_algorithm_methods=False,
            )
        )
    if settings.via_black_range_enabled:
        low = max(0, min(255, int(settings.via_black_range_min)))
        high = max(0, min(255, int(settings.via_black_range_max)))
        if low > high:
            low, high = high, low
        add_detection_results(
            *_via_detect_candidates(
                gray,
                low,
                high,
                settings,
                bright=False,
                include_range_method=True,
                include_algorithm_methods=False,
            )
        )
    if algorithm_methods_enabled:
        low = max(0, min(255, int(settings.via_white_range_min))) if settings.via_white_range_enabled else 0
        high = max(0, min(255, int(settings.via_white_range_max))) if settings.via_white_range_enabled else 255
        if low > high:
            low, high = high, low
        add_detection_results(
            *_via_detect_candidates(
                gray,
                low,
                high,
                settings,
                bright=True,
                include_range_method=False,
                include_algorithm_methods=True,
            )
        )
        if settings.via_black_range_enabled:
            low = max(0, min(255, int(settings.via_black_range_min)))
            high = max(0, min(255, int(settings.via_black_range_max)))
            if low > high:
                low, high = high, low
            add_detection_results(
                *_via_detect_candidates(
                    gray,
                    low,
                    high,
                    settings,
                    bright=False,
                    include_range_method=False,
                    include_algorithm_methods=True,
                )
            )
    if not candidates and not rejected_candidates:
        return ensure_binary_mask(image), []

    merged_candidates, duplicate_candidates = _merge_via_candidates(candidates, settings)
    result = _render_via_candidates_mask(gray.shape, merged_candidates, settings)
    debug_candidates = _debug_candidates_from_via_candidates(merged_candidates, accepted=True)
    debug_candidates.extend(
        _debug_candidates_from_via_candidates(duplicate_candidates, accepted=False, reason="duplicate")
    )
    debug_candidates.extend(_debug_candidates_from_via_candidates(rejected_candidates, accepted=False))
    return result, debug_candidates


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
    markers = cv2.watershed(marker_image, markers)
    refined = np.where(markers > 1, 255, 0).astype(np.uint8)
    result = binary.copy()
    result[correction_band > 0] = refined[correction_band > 0]
    return ensure_binary_mask(result)


def _resolve_conductor_edge_method(settings: ContourExtractionSettings) -> str:
    preferred = settings.conductor_gradient_edge_method or settings.edge_method
    return normalize_edge_method(preferred)


def _resolve_via_edge_method(settings: ContourExtractionSettings) -> str:
    preferred = settings.via_gradient_edge_method or settings.edge_method
    return normalize_edge_method(preferred)


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


def _via_algorithm_methods_enabled(settings: ContourExtractionSettings) -> bool:
    return any(
        (
            settings.via_detector_gradient_enabled,
            settings.via_detector_spot_enabled,
            settings.via_detector_hough_enabled,
            settings.via_detector_components_enabled,
            settings.via_detector_contours_enabled,
            settings.via_detector_morphology_enabled,
            settings.via_detector_template_enabled,
            settings.via_detector_blob_enabled,
        )
    )


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


def _via_detect_candidates(
    gray: np.ndarray,
    low: int,
    high: int,
    settings: ContourExtractionSettings,
    *,
    bright: bool,
    include_range_method: bool,
    include_algorithm_methods: bool,
) -> tuple[list[_ViaCandidate], list[_ViaCandidate]]:
    expected_span = _expected_via_span(settings)
    column_normalized = _normalize_columns(gray)
    row_normalized = _normalize_rows(gray)
    clahe_gray = _clahe_gray(gray)
    clahe_columns = _clahe_gray(column_normalized)

    intensity_mask = np.zeros_like(gray, dtype=np.uint8)
    intensity_images = [gray]
    if (bright and high >= 180) or (not bright and low <= 75):
        intensity_images.append(column_normalized)
    for candidate_image in intensity_images:
        intensity_mask = cv2.bitwise_or(intensity_mask, _range_mask(candidate_image, low, high))

    response = _via_multiscale_response(
        [gray, column_normalized, row_normalized, clahe_gray, clahe_columns],
        expected_span,
        bright=bright,
    )
    response_mask = _response_threshold_mask(response, intensity_mask)
    intensity_gate = cv2.dilate(intensity_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    gated_response_mask = cv2.bitwise_and(response_mask, intensity_gate)
    morphology_mask: np.ndarray | None = None

    def get_morphology_mask() -> np.ndarray:
        nonlocal morphology_mask
        if morphology_mask is None:
            seed_mask = gated_response_mask
            if cv2.countNonZero(seed_mask) == 0:
                seed_mask = cv2.bitwise_and(response_mask, intensity_gate)
            morphology_mask = _via_morphology_mask(seed_mask, expected_span)
        return morphology_mask

    accepted: list[_ViaCandidate] = []
    rejected: list[_ViaCandidate] = []

    candidate_groups: list[tuple[str, list[_ViaCandidate]]] = []
    if include_range_method:
        candidate_groups.extend(
            [
                (
                    "range-components",
                    _via_candidates_from_components(intensity_mask, response, source="range-components"),
                ),
                ("range-contours", _via_candidates_from_contours(intensity_mask, response, source="range-contours")),
            ]
        )
    if include_algorithm_methods and settings.via_detector_gradient_enabled:
        candidate_groups.append(
            (
                "gradient",
                _via_candidates_from_gradient(
                    [gray, column_normalized, row_normalized, clahe_gray, clahe_columns],
                    intensity_mask,
                    expected_span,
                    settings,
                    bright=bright,
                    source="gradient",
                ),
            )
        )
    if include_algorithm_methods and settings.via_detector_spot_enabled:
        candidate_groups.append(
            (
                "spot",
                _via_candidates_from_spots(
                    [gray, column_normalized, clahe_gray, clahe_columns],
                    intensity_mask,
                    expected_span,
                    settings,
                    bright=bright,
                    source="spot",
                ),
            )
        )
    if include_algorithm_methods and settings.via_detector_components_enabled:
        method_morphology_mask = get_morphology_mask()
        candidate_groups.extend(
            [
                ("components", _via_candidates_from_components(method_morphology_mask, response, source="components")),
            ]
        )
    if include_algorithm_methods and settings.via_detector_contours_enabled:
        method_morphology_mask = get_morphology_mask()
        candidate_groups.extend(
            [
                ("contours", _via_candidates_from_contours(method_morphology_mask, response, source="contours")),
                (
                    "contours-response",
                    _via_candidates_from_contours(gated_response_mask, response, source="contours-response"),
                ),
            ]
        )
    if include_algorithm_methods and settings.via_detector_morphology_enabled:
        method_morphology_mask = get_morphology_mask()
        distance_peaks = _via_distance_peak_mask(method_morphology_mask, expected_span, settings)
        candidate_groups.append(
            ("morphology", _via_candidates_from_components(distance_peaks, response, source="morphology"))
        )
    if include_algorithm_methods and settings.via_detector_template_enabled:
        template_mask = _via_template_peak_mask(
            [gray, column_normalized, row_normalized, clahe_gray, clahe_columns],
            response,
            intensity_mask,
            expected_span,
            settings,
        )
        candidate_groups.append(
            ("template", _via_candidates_from_components(template_mask, response, source="template"))
        )
    if include_algorithm_methods and settings.via_detector_blob_enabled:
        method_morphology_mask = get_morphology_mask()
        candidate_groups.append(
            (
                "blob",
                _via_candidates_from_blobs(method_morphology_mask, response, expected_span, settings, source="blob"),
            )
        )
    if include_algorithm_methods and settings.via_detector_hough_enabled:
        candidate_groups.extend(
            [
                (
                    "hough",
                    _via_candidates_from_hough(response, intensity_mask, expected_span, settings, source="hough"),
                ),
                (
                    "hough-gray",
                    _via_candidates_from_hough(
                        clahe_gray if bright else cv2.bitwise_not(clahe_gray),
                        intensity_mask,
                        expected_span,
                        settings,
                        source="hough-gray",
                    ),
                ),
            ]
        )
    for _source, candidates in candidate_groups:
        for candidate in candidates:
            candidate = _apply_via_method_threshold(candidate, settings)
            normalized = _normalize_via_candidate(candidate, settings, expected_span)
            if normalized.reason:
                rejected.append(normalized)
            else:
                accepted.append(normalized)
    return accepted, rejected


def _via_morphology_mask(seed_mask: np.ndarray, expected_span: int) -> np.ndarray:
    small_size = _odd_kernel_size(max(3, expected_span * 0.35), minimum=3)
    close_size = _odd_kernel_size(max(3, expected_span * 0.55), minimum=3)
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small_size, small_size))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    opened = cv2.morphologyEx(ensure_binary_mask(seed_mask), cv2.MORPH_OPEN, small_kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return ensure_binary_mask(closed)


def _via_distance_peak_mask(mask: np.ndarray, expected_span: int, settings: ContourExtractionSettings) -> np.ndarray:
    binary = ensure_binary_mask(mask)
    if cv2.countNonZero(binary) == 0:
        return np.zeros_like(binary, dtype=np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if distance.size == 0 or float(distance.max()) <= 0.0:
        return np.zeros_like(binary, dtype=np.uint8)
    local_max = distance == cv2.dilate(distance, np.ones((3, 3), dtype=np.uint8))
    radius_floor = max(1.0, expected_span * float(settings.via_morphology_peak_scale))
    peaks = np.where(local_max & (distance >= radius_floor), 255, 0).astype(np.uint8)
    peak_size = _odd_kernel_size(max(3, expected_span * 0.35), minimum=3)
    peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (peak_size, peak_size))
    return cv2.dilate(peaks, peak_kernel, iterations=1)


def _via_template_peak_mask(
    images: list[np.ndarray],
    response: np.ndarray,
    intensity_mask: np.ndarray,
    expected_span: int,
    settings: ContourExtractionSettings,
) -> np.ndarray:
    gate = cv2.dilate(ensure_binary_mask(intensity_mask), np.ones((3, 3), dtype=np.uint8), iterations=1)
    if cv2.countNonZero(gate) == 0:
        gate = np.full_like(gate, 255, dtype=np.uint8)
    mask = np.zeros_like(ensure_uint8(response), dtype=np.uint8)
    saved_templates = _via_saved_template_arrays(settings)
    if saved_templates:
        for search_image in images:
            mask = cv2.bitwise_or(mask, _via_saved_template_peak_mask(search_image, gate, saved_templates, settings))
        return mask

    size = _odd_kernel_size(max(5, expected_span), minimum=5)
    template = np.zeros((size, size), dtype=np.uint8)
    radius = max(1, size // 2 - 1)
    cv2.circle(template, (size // 2, size // 2), radius, 255, thickness=-1, lineType=cv2.LINE_AA)
    search = cv2.bitwise_and(ensure_uint8(response), gate)
    if search.shape[0] < size or search.shape[1] < size or cv2.countNonZero(search) == 0:
        return np.zeros_like(search, dtype=np.uint8)
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    scores = np.nan_to_num(scores, copy=False)
    if scores.size == 0:
        return np.zeros_like(search, dtype=np.uint8)
    cutoff = max(float(settings.via_template_min_score), min(0.72, float(np.percentile(scores, 99.0))))
    max_scores = cv2.dilate(scores, np.ones((3, 3), dtype=np.uint8))
    locations = np.where((scores >= cutoff) & (scores >= max_scores - 1e-6))
    max_points = 600
    if len(locations[0]) > max_points:
        values = scores[locations]
        order = np.argsort(values)[-max_points:]
        ys = locations[0][order]
        xs = locations[1][order]
    else:
        ys = locations[0]
        xs = locations[1]
    for y_coord, x_coord in zip(ys, xs, strict=False):
        center = (int(x_coord) + size // 2, int(y_coord) + size // 2)
        cv2.circle(mask, center, max(1, expected_span // 3), 255, thickness=-1, lineType=cv2.LINE_AA)
    return mask


def _via_saved_template_arrays(settings: ContourExtractionSettings) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for payload in settings.via_template_images:
        try:
            template = ensure_uint8(np.asarray(payload))
        except Exception:
            continue
        if template.ndim == 3:
            template = _via_grayscale(template)
        if template.ndim != 2 or template.shape[0] < 2 or template.shape[1] < 2:
            continue
        arrays.append(template)
    return arrays


def _via_saved_template_peak_mask(
    image: np.ndarray,
    gate: np.ndarray,
    templates: list[np.ndarray],
    settings: ContourExtractionSettings,
) -> np.ndarray:
    search = ensure_uint8(image)
    if search.ndim != 2:
        search = _via_grayscale(search)
    result = np.zeros_like(search, dtype=np.uint8)
    search = cv2.bitwise_and(search, gate)
    if cv2.countNonZero(search) == 0:
        return result
    min_score = float(settings.via_template_min_score)
    for template in templates:
        if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
            continue
        prepared_template = _normalize_template_for_matching(template)
        prepared_search = _normalize_template_for_matching(search)
        scores = cv2.matchTemplate(prepared_search, prepared_template, cv2.TM_CCOEFF_NORMED)
        scores = np.nan_to_num(scores, copy=False)
        if scores.size == 0:
            continue
        cutoff = max(min_score, min(0.92, float(np.percentile(scores, 99.4))))
        local_max = scores >= (cv2.dilate(scores, np.ones((3, 3), dtype=np.uint8)) - 1e-6)
        ys, xs = np.where((scores >= cutoff) & local_max)
        if len(xs) > 600:
            values = scores[ys, xs]
            order = np.argsort(values)[-600:]
            xs = xs[order]
            ys = ys[order]
        radius = max(1, min(template.shape[:2]) // 3)
        for y_coord, x_coord in zip(ys, xs, strict=False):
            center = (int(x_coord) + template.shape[1] // 2, int(y_coord) + template.shape[0] // 2)
            if gate[center[1], center[0]] == 0:
                continue
            cv2.circle(result, center, radius, 255, thickness=-1, lineType=cv2.LINE_AA)
    return result


def _normalize_template_for_matching(image: np.ndarray) -> np.ndarray:
    data = ensure_uint8(image)
    if data.size == 0:
        return data
    minimum = int(data.min())
    maximum = int(data.max())
    if maximum <= minimum:
        return data
    return np.clip((data.astype(np.float32) - minimum) * (255.0 / float(maximum - minimum)), 0, 255).astype(np.uint8)


def _via_candidate_radii(settings: ContourExtractionSettings, expected_span: int) -> list[int]:
    diameters: list[float] = []
    for width, height in zip(settings.fixed_via_widths, settings.fixed_via_heights, strict=False):
        if int(width) > 0 and int(height) > 0:
            diameters.append((float(width) + float(height)) / 2.0)
    if not diameters:
        min_width, max_width, min_height, max_height = _candidate_size_limits(settings, expected_span)
        min_diameter = max(3.0, min(min_width, min_height, expected_span * 0.75))
        max_diameter = max(min_diameter, min(max_width, max_height, expected_span * 1.55))
        if max_diameter - min_diameter < 2.0:
            diameters.append(float(expected_span))
        else:
            step = max(1.0, expected_span * 0.18)
            value = min_diameter
            while value <= max_diameter + 0.5:
                diameters.append(value)
                value += step
    radii = {max(2, round(diameter / 2.0)) for diameter in diameters}
    if not radii:
        radii.add(max(2, round(expected_span / 2.0)))
    return sorted(radii)


def _via_candidates_from_gradient(
    images: list[np.ndarray],
    intensity_mask: np.ndarray,
    expected_span: int,
    settings: ContourExtractionSettings,
    *,
    bright: bool,
    source: str,
) -> list[_ViaCandidate]:
    gate = cv2.dilate(
        ensure_binary_mask(intensity_mask),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_odd_kernel_size(expected_span * 1.7), _odd_kernel_size(expected_span * 1.7))
        ),
        iterations=1,
    )
    min_strength = max(0.0, float(settings.via_gradient_min_strength))
    min_coverage = max(0.0, min(1.0, float(settings.via_gradient_min_coverage)))
    edge_method = _resolve_via_edge_method(settings)
    candidates: list[_ViaCandidate] = []
    for image_index, image in enumerate(images[:1]):
        gray = ensure_uint8(image)
        smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
        grad_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        gradient = build_gradient_elevation(gray, edge_method)
        if gradient.size == 0 or int(gradient.max()) <= 0:
            continue
        if cv2.countNonZero(gradient) == 0:
            continue
        for radius in _via_candidate_radii(settings, expected_span):
            candidates.extend(
                _via_candidates_from_gradient_radius(
                    gray,
                    gradient,
                    grad_x,
                    grad_y,
                    gate,
                    radius,
                    min_strength,
                    min_coverage,
                    bright=bright,
                    source=f"{source}{image_index}",
                )
            )
        if image_index == 0:
            candidates.extend(
                _via_candidates_from_gradient_spots(
                    gray,
                    gate,
                    expected_span,
                    settings,
                    min_strength,
                    min_coverage,
                    bright=bright,
                    source=f"{source}-spot",
                )
            )
    return candidates


def _via_candidates_from_gradient_spots(
    image: np.ndarray,
    gate: np.ndarray,
    expected_span: int,
    settings: ContourExtractionSettings,
    min_strength: float,
    min_coverage: float,
    *,
    bright: bool,
    source: str,
) -> list[_ViaCandidate]:
    response = cv2.bitwise_and(
        _via_spot_response(image, expected_span, settings, bright=bright), ensure_binary_mask(gate)
    )
    values = response[response > 0]
    if values.size == 0:
        return []
    cutoff = max(max(2.0, min_strength * 0.7), min(120.0, float(np.percentile(values, 96.5))))
    peak_kernel_size = _odd_kernel_size(max(3, expected_span * 1.25), minimum=3)
    local_max = response == cv2.dilate(response, np.ones((peak_kernel_size, peak_kernel_size), dtype=np.uint8))
    ys, xs = np.where((response >= cutoff) & local_max)
    if len(xs) == 0:
        return []
    if len(xs) > 1200:
        peak_values = response[ys, xs]
        order = np.argsort(peak_values)[-1200:]
        ys = ys[order]
        xs = xs[order]

    candidates: list[_ViaCandidate] = []
    radii = _via_candidate_radii(settings, expected_span)
    for y_coord, x_coord in zip(ys, xs, strict=False):
        best_candidate: _ViaCandidate | None = None
        for radius in radii:
            if not _candidate_overlaps_gate(gate, int(x_coord), int(y_coord), radius):
                continue
            center_contrast, center_coverage = _source_radial_metrics(
                image, int(x_coord), int(y_coord), radius, bright=bright
            )
            if center_contrast < max(2.0, min_strength * 0.55):
                continue
            if center_coverage < max(0.12, min_coverage * 0.65):
                continue
            if _source_line_extension_coverage(image, int(x_coord), int(y_coord), radius, bright=bright) > 0.18:
                continue
            response_contrast, response_roundness = _spot_candidate_metrics(
                response, int(x_coord), int(y_coord), max(2, radius)
            )
            if response_contrast < max(1.0, min_strength * 0.35):
                continue
            center_x, center_y = _spot_candidate_centroid(response, int(x_coord), int(y_coord), max(2, radius))
            roundness = max(float(response_roundness), center_coverage * 100.0)
            score = (
                520.0
                + float(response[y_coord, x_coord])
                + center_contrast * 2.0
                + center_coverage * 80.0
                + response_roundness
            )
            candidate = _ViaCandidate(
                center_x=float(center_x),
                center_y=float(center_y),
                width=float(radius * 2 + 1),
                height=float(radius * 2 + 1),
                score=score,
                source=source,
                roundness=max(0.0, min(100.0, roundness)),
            )
            if best_candidate is None or candidate.score > best_candidate.score:
                best_candidate = candidate
        if best_candidate is not None:
            candidates.append(best_candidate)
    return candidates


def _via_candidates_from_gradient_radius(
    gradient_source: np.ndarray,
    gradient: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    gate: np.ndarray,
    radius: int,
    min_strength: float,
    min_coverage: float,
    *,
    bright: bool,
    source: str,
) -> list[_ViaCandidate]:
    radius = max(2, int(radius))
    padding = max(2, round(radius * 0.35))
    size = radius * 2 + padding * 2 + 1
    if gradient.shape[0] < size or gradient.shape[1] < size:
        return []
    center = size // 2
    template = np.zeros((size, size), dtype=np.float32)
    thickness = max(1, round(radius * 0.22))
    cv2.circle(template, (center, center), radius, 1.0, thickness=thickness, lineType=cv2.LINE_AA)
    if float(template.sum()) <= 0.0:
        return []

    search = gradient.astype(np.float32) / 255.0
    scores = cv2.matchTemplate(search, template, cv2.TM_CCORR_NORMED)
    scores = np.nan_to_num(scores, copy=False)
    if scores.size == 0:
        return []
    cutoff = max(0.18, min(0.78, float(np.percentile(scores, 99.35))))
    local_max = scores >= (cv2.dilate(scores, np.ones((3, 3), dtype=np.uint8)) - 1e-6)
    ys, xs = np.where((scores >= cutoff) & local_max)
    if len(xs) > 800:
        values = scores[ys, xs]
        order = np.argsort(values)[-800:]
        xs = xs[order]
        ys = ys[order]

    candidates: list[_ViaCandidate] = []
    for top, left in zip(ys, xs, strict=False):
        center_x = int(left) + center
        center_y = int(top) + center
        if not _candidate_overlaps_gate(gate, center_x, center_y, radius):
            continue
        edge_strength, edge_coverage = _circle_gradient_metrics(
            gradient_source,
            gradient,
            gradient_x,
            gradient_y,
            center_x,
            center_y,
            radius,
            min_strength,
            bright=bright,
        )
        if edge_strength < min_strength or edge_coverage < min_coverage:
            continue
        roundness = max(0.0, min(100.0, edge_coverage * 100.0))
        score = 220.0 + float(scores[top, left]) * 120.0 + edge_strength + edge_coverage * 80.0
        candidates.append(
            _ViaCandidate(
                center_x=float(center_x),
                center_y=float(center_y),
                width=float(radius * 2 + 1),
                height=float(radius * 2 + 1),
                score=score,
                source=source,
                roundness=roundness,
            )
        )
    return candidates


def _circle_gradient_metrics(
    source_image: np.ndarray,
    gradient: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
    min_strength: float,
    *,
    bright: bool,
) -> tuple[float, float]:
    radius = max(2, int(radius))
    padding = max(2, round(radius * 0.35))
    left = max(0, int(center_x) - radius - padding)
    top = max(0, int(center_y) - radius - padding)
    right = min(gradient.shape[1], int(center_x) + radius + padding + 1)
    bottom = min(gradient.shape[0], int(center_y) + radius + padding + 1)
    if right <= left or bottom <= top:
        return 0.0, 0.0
    patch = gradient[top:bottom, left:right]
    source_patch = ensure_uint8(source_image)[top:bottom, left:right]
    patch_x = gradient_x[top:bottom, left:right].astype(np.float32)
    patch_y = gradient_y[top:bottom, left:right].astype(np.float32)
    ring = np.zeros_like(patch, dtype=np.uint8)
    local_center = (int(center_x) - left, int(center_y) - top)
    thickness = max(1, round(radius * 0.24))
    cv2.circle(ring, local_center, radius, 255, thickness=thickness, lineType=cv2.LINE_AA)
    ring_mask = ring > 0
    ring_values = patch[ring_mask]
    if ring_values.size == 0:
        return 0.0, 0.0
    strength = float(np.mean(ring_values))
    coverage_threshold = max(float(min_strength), float(np.percentile(patch, 72)))
    yy, xx = np.ogrid[top - center_y : bottom - center_y, left - center_x : right - center_x]
    grid_x = np.broadcast_to(xx.astype(np.float32), patch.shape)
    grid_y = np.broadcast_to(yy.astype(np.float32), patch.shape)
    radius_map = np.sqrt(grid_x * grid_x + grid_y * grid_y).astype(np.float32)
    gradient_norm = np.sqrt(patch_x * patch_x + patch_y * patch_y)
    directional = np.zeros_like(gradient_norm, dtype=np.float32)
    valid = ring_mask & (radius_map > 1e-3) & (gradient_norm > 1e-3)
    directional[valid] = np.abs(
        (patch_x[valid] * grid_x[valid] + patch_y[valid] * grid_y[valid]) / (gradient_norm[valid] * radius_map[valid])
    )
    strong_ring = ring_mask & (patch >= coverage_threshold)
    strong_directional = strong_ring & (directional >= 0.35)
    directional_coverage = float(np.count_nonzero(strong_directional)) / float(ring_values.size)
    angular_coverage = _angular_ring_coverage(strong_directional, grid_x, grid_y)
    center_contrast, center_fill_coverage = _radial_center_metrics(
        source_patch, radius_map, grid_x, grid_y, radius, bright=bright
    )
    if center_contrast < max(float(min_strength), strength * 0.32):
        return strength, 0.0
    coverage = min(
        float(np.count_nonzero(ring_values >= coverage_threshold)) / float(ring_values.size),
        directional_coverage * 1.35,
        angular_coverage,
        center_fill_coverage,
    )
    return strength, coverage


def _radial_center_metrics(
    patch: np.ndarray,
    radius_map: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    radius: int,
    *,
    bright: bool,
) -> tuple[float, float]:
    inner_mask = radius_map <= max(1.0, float(radius) * 0.45)
    outer_mask = (radius_map >= float(radius) * 1.15) & (radius_map <= float(radius) * 1.65)
    if not np.any(inner_mask) or not np.any(outer_mask):
        return 0.0, 0.0
    data = ensure_uint8(patch).astype(np.float32)
    inner_mean = float(np.mean(data[inner_mask]))
    outer_mean = float(np.mean(data[outer_mask]))
    signed_contrast = inner_mean - outer_mean if bright else outer_mean - inner_mean
    if signed_contrast <= 0.0:
        return 0.0, 0.0
    threshold = (inner_mean + outer_mean) * 0.5
    if bright:
        fill_mask = inner_mask & (data >= threshold)
    else:
        fill_mask = inner_mask & (data <= threshold)
    pixel_coverage = float(np.count_nonzero(fill_mask)) / float(max(1, np.count_nonzero(inner_mask)))
    angular_coverage = _angular_ring_coverage(fill_mask, grid_x, grid_y, sectors=16)
    shape_coverage = _mask_aspect_coverage(fill_mask)
    fill_coverage = min(pixel_coverage * 1.15, angular_coverage, shape_coverage)
    return signed_contrast, fill_coverage


def _source_radial_metrics(
    source_image: np.ndarray, center_x: int, center_y: int, radius: int, *, bright: bool
) -> tuple[float, float]:
    radius = max(2, int(radius))
    padding = max(2, round(radius * 0.65))
    left = max(0, int(center_x) - radius - padding)
    top = max(0, int(center_y) - radius - padding)
    right = min(source_image.shape[1], int(center_x) + radius + padding + 1)
    bottom = min(source_image.shape[0], int(center_y) + radius + padding + 1)
    if right <= left or bottom <= top:
        return 0.0, 0.0
    patch = ensure_uint8(source_image)[top:bottom, left:right]
    yy, xx = np.ogrid[top - center_y : bottom - center_y, left - center_x : right - center_x]
    grid_x = np.broadcast_to(xx.astype(np.float32), patch.shape)
    grid_y = np.broadcast_to(yy.astype(np.float32), patch.shape)
    radius_map = np.sqrt(grid_x * grid_x + grid_y * grid_y).astype(np.float32)
    return _radial_center_metrics(patch, radius_map, grid_x, grid_y, radius, bright=bright)


def _source_line_extension_coverage(
    source_image: np.ndarray, center_x: int, center_y: int, radius: int, *, bright: bool
) -> float:
    radius = max(2, int(radius))
    outer_radius = max(radius + 2, round(radius * 2.4))
    left = max(0, int(center_x) - outer_radius)
    top = max(0, int(center_y) - outer_radius)
    right = min(source_image.shape[1], int(center_x) + outer_radius + 1)
    bottom = min(source_image.shape[0], int(center_y) + outer_radius + 1)
    if right <= left or bottom <= top:
        return 0.0
    patch = ensure_uint8(source_image)[top:bottom, left:right].astype(np.float32)
    yy, xx = np.ogrid[top - center_y : bottom - center_y, left - center_x : right - center_x]
    radius_map = np.sqrt(xx.astype(np.float32) * xx.astype(np.float32) + yy.astype(np.float32) * yy.astype(np.float32))
    inner = radius_map <= max(1.0, float(radius) * 0.55)
    outer = (radius_map >= float(radius) * 1.2) & (radius_map <= float(outer_radius))
    if not np.any(inner) or not np.any(outer):
        return 0.0
    inner_mean = float(np.mean(patch[inner]))
    outer_mean = float(np.mean(patch[outer]))
    threshold = (inner_mean + outer_mean) * 0.5
    strip_width = max(1.0, float(radius) * 0.28)
    axial_strips = outer & ((np.abs(xx) <= strip_width) | (np.abs(yy) <= strip_width))
    if not np.any(axial_strips):
        return 0.0
    extended = patch[axial_strips] >= threshold if bright else patch[axial_strips] <= threshold
    return float(np.count_nonzero(extended)) / float(max(1, np.count_nonzero(axial_strips)))


def _mask_aspect_coverage(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    return float(min(width, height)) / float(max(1, max(width, height)))


def _angular_ring_coverage(mask: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, *, sectors: int = 24) -> float:
    if sectors <= 0 or not np.any(mask):
        return 0.0
    angles = np.arctan2(grid_y[mask], grid_x[mask])
    bins = np.floor(((angles + np.pi) / (2.0 * np.pi)) * sectors).astype(np.int32)
    bins = np.clip(bins, 0, sectors - 1)
    return float(np.unique(bins).size) / float(sectors)


def _candidate_overlaps_gate(gate: np.ndarray, center_x: int, center_y: int, radius: int) -> bool:
    if gate.size == 0 or cv2.countNonZero(gate) == 0:
        return True
    radius = max(1, int(radius))
    left = max(0, int(center_x) - radius)
    top = max(0, int(center_y) - radius)
    right = min(gate.shape[1], int(center_x) + radius + 1)
    bottom = min(gate.shape[0], int(center_y) + radius + 1)
    if right <= left or bottom <= top:
        return False
    patch = gate[top:bottom, left:right]
    disk = np.zeros_like(patch, dtype=np.uint8)
    cv2.circle(disk, (int(center_x) - left, int(center_y) - top), radius, 255, thickness=-1, lineType=cv2.LINE_AA)
    disk_area = max(1, cv2.countNonZero(disk))
    overlap = cv2.countNonZero(cv2.bitwise_and(patch, disk))
    return (overlap / disk_area) >= 0.03


def _via_candidates_from_spots(
    images: list[np.ndarray],
    intensity_mask: np.ndarray,
    expected_span: int,
    settings: ContourExtractionSettings,
    *,
    bright: bool,
    source: str,
) -> list[_ViaCandidate]:
    gate = cv2.dilate(
        ensure_binary_mask(intensity_mask),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_odd_kernel_size(expected_span * 1.5), _odd_kernel_size(expected_span * 1.5))
        ),
        iterations=1,
    )
    if cv2.countNonZero(gate) == 0:
        gate = np.full_like(gate, 255, dtype=np.uint8)
    candidates: list[_ViaCandidate] = []
    for image_index, image in enumerate(images):
        response = _via_spot_response(image, expected_span, settings, bright=bright)
        candidates.extend(
            _via_candidates_from_spot_response(
                response,
                gate,
                expected_span,
                settings,
                source=f"{source}{image_index}",
            )
        )
    return candidates


def _via_spot_response(
    image: np.ndarray,
    expected_span: int,
    settings: ContourExtractionSettings,
    *,
    bright: bool,
) -> np.ndarray:
    gray = ensure_uint8(image)
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
    small_sigma = max(0.7, expected_span * 0.18)
    large_sigma = max(small_sigma + 0.5, expected_span * 0.75)
    small_blur = cv2.GaussianBlur(smoothed, (0, 0), small_sigma)
    large_blur = cv2.GaussianBlur(smoothed, (0, 0), large_sigma)
    if bright:
        response = cv2.subtract(small_blur, large_blur)
        operation = cv2.MORPH_TOPHAT
    else:
        response = cv2.subtract(large_blur, small_blur)
        operation = cv2.MORPH_BLACKHAT
    kernel_size = _odd_kernel_size(max(5, expected_span * 2.1), minimum=5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    response = cv2.max(response, cv2.morphologyEx(smoothed, operation, kernel))
    response = cv2.max(response, _via_log_blob_response(smoothed, expected_span, bright=bright))

    line_length = _odd_kernel_size(max(9, expected_span * 4.0), minimum=9)
    line_width = _odd_kernel_size(max(3, expected_span * 0.45), minimum=3)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_length, line_width))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_width, line_length))
    horizontal_lines = cv2.morphologyEx(response, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(response, cv2.MORPH_OPEN, vertical_kernel)
    line_response = cv2.max(horizontal_lines, vertical_lines)
    line_suppression = max(0.0, min(1.0, float(settings.via_spot_line_suppression)))
    if line_suppression > 0.0:
        response = cv2.subtract(response, cv2.convertScaleAbs(line_response, alpha=line_suppression))
    return ensure_uint8(response)


def _via_log_blob_response(gray: np.ndarray, expected_span: int, *, bright: bool) -> np.ndarray:
    data = ensure_uint8(gray).astype(np.float32)
    result = np.zeros_like(data, dtype=np.float32)
    sigmas = (
        max(0.65, expected_span * 0.20),
        max(0.85, expected_span * 0.32),
        max(1.05, expected_span * 0.46),
    )
    for sigma in sigmas:
        blurred = cv2.GaussianBlur(data, (0, 0), sigma)
        laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        local = (-laplacian if bright else laplacian) * float(sigma * sigma)
        local = np.maximum(local, 0.0)
        values = local[local > 0.0]
        if values.size == 0:
            continue
        scale = float(np.percentile(values, 99.5))
        if scale <= 1e-6:
            continue
        result = np.maximum(result, np.clip(local * (255.0 / scale), 0.0, 255.0))
    return result.astype(np.uint8)


def _via_candidates_from_spot_response(
    response: np.ndarray,
    gate: np.ndarray,
    expected_span: int,
    settings: ContourExtractionSettings,
    *,
    source: str,
) -> list[_ViaCandidate]:
    gated = cv2.bitwise_and(ensure_uint8(response), ensure_binary_mask(gate))
    values = gated[gated > 0]
    if values.size == 0:
        return []
    cutoff = max(float(settings.via_spot_min_contrast), min(95.0, float(np.percentile(values, 98.7))))
    local_max = gated == cv2.dilate(gated, np.ones((3, 3), dtype=np.uint8))
    peaks = np.where((gated >= cutoff) & local_max)
    if len(peaks[0]) == 0:
        return []
    ys = peaks[0]
    xs = peaks[1]
    if len(xs) > 1500:
        peak_values = gated[ys, xs]
        order = np.argsort(peak_values)[-1500:]
        ys = ys[order]
        xs = xs[order]

    radius = max(2, round(expected_span * 0.55))
    candidates: list[_ViaCandidate] = []
    for y_coord, x_coord in zip(ys, xs, strict=False):
        contrast, roundness = _spot_candidate_metrics(gated, int(x_coord), int(y_coord), radius)
        if contrast < float(settings.via_spot_min_contrast):
            continue
        if roundness < float(settings.via_spot_min_roundness):
            continue
        center_x, center_y = _spot_candidate_centroid(gated, int(x_coord), int(y_coord), radius)
        score = float(gated[y_coord, x_coord]) + contrast * 1.6 + roundness
        candidates.append(
            _ViaCandidate(
                center_x=float(center_x),
                center_y=float(center_y),
                width=float(max(2, expected_span)),
                height=float(max(2, expected_span)),
                score=score,
                source=source,
                roundness=roundness,
            )
        )
    return candidates


def _spot_candidate_centroid(response: np.ndarray, center_x: int, center_y: int, radius: int) -> tuple[float, float]:
    radius = max(2, int(radius))
    left = max(0, center_x - radius)
    top = max(0, center_y - radius)
    right = min(response.shape[1], center_x + radius + 1)
    bottom = min(response.shape[0], center_y + radius + 1)
    if right <= left or bottom <= top:
        return float(center_x), float(center_y)
    patch = response[top:bottom, left:right].astype(np.float32)
    yy, xx = np.ogrid[top - center_y : bottom - center_y, left - center_x : right - center_x]
    disk = (xx * xx + yy * yy) <= radius * radius
    if not np.any(disk):
        return float(center_x), float(center_y)
    baseline = float(np.percentile(patch[disk], 45))
    weights = np.maximum(patch - baseline, 0.0)
    weights[~disk] = 0.0
    total_weight = float(weights.sum())
    if total_weight <= 1e-6:
        return float(center_x), float(center_y)
    local_x = np.arange(left, right, dtype=np.float32)
    local_y = np.arange(top, bottom, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(local_x, local_y)
    return float((weights * grid_x).sum() / total_weight), float((weights * grid_y).sum() / total_weight)


def _spot_candidate_metrics(response: np.ndarray, center_x: int, center_y: int, radius: int) -> tuple[float, float]:
    radius = max(2, int(radius))
    outer_radius = max(radius + 2, round(radius * 1.9))
    left = max(0, center_x - outer_radius)
    top = max(0, center_y - outer_radius)
    right = min(response.shape[1], center_x + outer_radius + 1)
    bottom = min(response.shape[0], center_y + outer_radius + 1)
    if right <= left or bottom <= top:
        return 0.0, 0.0
    patch = response[top:bottom, left:right].astype(np.float32)
    yy, xx = np.ogrid[top - center_y : bottom - center_y, left - center_x : right - center_x]
    distance_sq = xx * xx + yy * yy
    inner = distance_sq <= radius * radius
    outer = (distance_sq > radius * radius) & (distance_sq <= outer_radius * outer_radius)
    inner_values = patch[inner]
    outer_values = patch[outer]
    if inner_values.size == 0:
        return 0.0, 0.0
    contrast = float(np.mean(inner_values) - (np.mean(outer_values) if outer_values.size else 0.0))
    weights = np.maximum(patch - float(np.percentile(patch, 55)), 0.0)
    weights[~inner] = 0.0
    total_weight = float(weights.sum())
    if total_weight <= 1e-6:
        return max(0.0, contrast), 0.0
    local_x = np.arange(left, right, dtype=np.float32) - float(center_x)
    local_y = np.arange(top, bottom, dtype=np.float32) - float(center_y)
    grid_x, grid_y = np.meshgrid(local_x, local_y)
    cov_xx = float((weights * grid_x * grid_x).sum() / total_weight)
    cov_yy = float((weights * grid_y * grid_y).sum() / total_weight)
    cov_xy = float((weights * grid_x * grid_y).sum() / total_weight)
    trace = cov_xx + cov_yy
    determinant = cov_xx * cov_yy - cov_xy * cov_xy
    delta = max(0.0, trace * trace / 4.0 - determinant)
    lambda_max = trace / 2.0 + float(np.sqrt(delta))
    lambda_min = trace / 2.0 - float(np.sqrt(delta))
    roundness = 100.0 * max(0.0, lambda_min) / max(1e-6, lambda_max)
    return max(0.0, contrast), max(0.0, min(100.0, roundness))


def _via_candidates_from_components(mask: np.ndarray, response: np.ndarray, *, source: str) -> list[_ViaCandidate]:
    binary = ensure_binary_mask(mask)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates: list[_ViaCandidate] = []
    for index in range(1, count):
        x_coord, y_coord, width, height, area = stats[index]
        if area <= 0:
            continue
        score = _candidate_response_score(response, int(x_coord), int(y_coord), int(width), int(height))
        candidates.append(
            _ViaCandidate(
                center_x=float(centroids[index][0]),
                center_y=float(centroids[index][1]),
                width=float(width),
                height=float(height),
                score=score + min(50.0, float(area)),
                source=source,
            )
        )
    return candidates


def _via_candidates_from_contours(mask: np.ndarray, response: np.ndarray, *, source: str) -> list[_ViaCandidate]:
    contours, _hierarchy = cv2.findContours(ensure_binary_mask(mask).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[_ViaCandidate] = []
    for contour in contours:
        if contour is None or len(contour) < 3:
            continue
        x_coord, y_coord, width, height = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if abs(moments["m00"]) > 1e-6:
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
        else:
            center_x = x_coord + width / 2.0
            center_y = y_coord + height / 2.0
        roundness = _candidate_roundness_from_contour(contour, width, height)
        score = _candidate_response_score(response, int(x_coord), int(y_coord), int(width), int(height)) + roundness
        candidates.append(
            _ViaCandidate(
                center_x=float(center_x),
                center_y=float(center_y),
                width=float(width),
                height=float(height),
                score=score,
                source=source,
                roundness=roundness,
            )
        )
    return candidates


def _via_candidates_from_hough(
    image: np.ndarray,
    intensity_mask: np.ndarray,
    expected_span: int,
    settings: ContourExtractionSettings,
    *,
    source: str,
) -> list[_ViaCandidate]:
    prepared = ensure_uint8(image)
    if prepared.ndim != 2:
        prepared = _via_grayscale(prepared)
    prepared = cv2.GaussianBlur(prepared, (3, 3), 0)
    min_radius = max(2, round(expected_span * 0.28))
    max_radius = max(min_radius + 1, round(expected_span * 0.85))
    min_distance = max(3, round(expected_span * 0.85))
    circles = cv2.HoughCircles(
        prepared,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=float(min_distance),
        param1=float(settings.via_hough_edge_threshold),
        param2=float(settings.via_hough_accumulator_threshold),
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []
    gate = cv2.dilate(ensure_binary_mask(intensity_mask), np.ones((3, 3), dtype=np.uint8), iterations=1)
    candidates: list[_ViaCandidate] = []
    for x_coord, y_coord, radius in np.round(circles[0]).astype(np.int32):
        if x_coord < 0 or y_coord < 0 or x_coord >= gate.shape[1] or y_coord >= gate.shape[0]:
            continue
        if not _candidate_overlaps_gate(gate, int(x_coord), int(y_coord), int(radius)):
            continue
        width = float(max(2, radius * 2 + 1))
        height = width
        candidates.append(
            _ViaCandidate(
                center_x=float(x_coord),
                center_y=float(y_coord),
                width=width,
                height=height,
                score=160.0 + float(radius),
                source=source,
                roundness=100.0,
            )
        )
    return candidates


def _via_candidates_from_blobs(
    mask: np.ndarray,
    response: np.ndarray,
    expected_span: int,
    settings: ContourExtractionSettings,
    *,
    source: str,
) -> list[_ViaCandidate]:
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = max(3.0, (expected_span * 0.25) ** 2)
    params.maxArea = max(params.minArea + 1.0, (expected_span * 1.6) ** 2)
    params.filterByCircularity = True
    params.minCircularity = float(settings.via_blob_min_circularity)
    params.filterByConvexity = True
    params.minConvexity = 0.45
    params.filterByInertia = True
    params.minInertiaRatio = 0.25
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(ensure_binary_mask(mask))
    candidates: list[_ViaCandidate] = []
    for keypoint in keypoints:
        diameter = max(2.0, float(keypoint.size))
        x_coord = round(keypoint.pt[0] - diameter / 2.0)
        y_coord = round(keypoint.pt[1] - diameter / 2.0)
        score = 140.0 + _candidate_response_score(response, x_coord, y_coord, round(diameter), round(diameter))
        candidates.append(
            _ViaCandidate(
                center_x=float(keypoint.pt[0]),
                center_y=float(keypoint.pt[1]),
                width=diameter,
                height=diameter,
                score=score,
                source=source,
                roundness=100.0,
            )
        )
    return candidates


def _apply_via_method_threshold(candidate: _ViaCandidate, settings: ContourExtractionSettings) -> _ViaCandidate:
    reason = ""
    if candidate.source.startswith("components") and candidate.score < float(settings.via_component_min_score):
        reason = "component_score"
    elif candidate.source.startswith("contours") and candidate.score < float(settings.via_contour_min_score):
        reason = "contour_score"
    if not reason:
        return candidate
    return _ViaCandidate(
        center_x=candidate.center_x,
        center_y=candidate.center_y,
        width=candidate.width,
        height=candidate.height,
        score=candidate.score,
        source=candidate.source,
        roundness=candidate.roundness,
        reason=reason,
    )


def _candidate_response_score(response: np.ndarray, x_coord: int, y_coord: int, width: int, height: int) -> float:
    left = max(0, int(x_coord))
    top = max(0, int(y_coord))
    right = min(response.shape[1], left + max(1, int(width)))
    bottom = min(response.shape[0], top + max(1, int(height)))
    if right <= left or bottom <= top:
        return 0.0
    return float(np.mean(response[top:bottom, left:right]))


def _candidate_roundness_from_contour(contour: np.ndarray, width: int, height: int) -> float:
    area = abs(float(cv2.contourArea(contour)))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 0.0 or perimeter <= 0.0:
        return 0.0
    circularity = 100.0 * (4.0 * np.pi * area / max(1e-6, perimeter * perimeter))
    aspect_roundness = 100.0 * min(width, height) / max(1, max(width, height))
    return max(0.0, min(100.0, circularity, aspect_roundness))


def _normalize_via_candidate(
    candidate: _ViaCandidate, settings: ContourExtractionSettings, expected_span: int
) -> _ViaCandidate:
    width = max(1.0, float(candidate.width))
    height = max(1.0, float(candidate.height))
    reason = candidate.reason or _via_candidate_rejection_reason(
        width, height, candidate.roundness, settings, expected_span
    )
    return _ViaCandidate(
        center_x=float(candidate.center_x),
        center_y=float(candidate.center_y),
        width=width,
        height=height,
        score=float(candidate.score),
        source=candidate.source,
        roundness=float(candidate.roundness),
        reason=reason,
    )


def _via_candidate_rejection_reason(
    width: float,
    height: float,
    roundness: float,
    settings: ContourExtractionSettings,
    expected_span: int,
) -> str:
    if width <= 0.0 or height <= 0.0:
        return "empty_geometry"
    min_width, max_width, min_height, max_height = _candidate_size_limits(settings, expected_span)
    if width < min_width:
        return "min_via_width"
    if width > max_width:
        return "max_via_width"
    if height < min_height:
        return "min_via_height"
    if height > max_height:
        return "max_via_height"
    aspect = width / max(1.0, height)
    min_aspect = settings.min_aspect_ratio if settings.min_aspect_ratio > 0.0 else 0.25
    max_aspect = settings.max_aspect_ratio if settings.max_aspect_ratio is not None else 4.0
    if aspect < min_aspect:
        return "min_aspect_ratio"
    if aspect > max_aspect:
        return "max_aspect_ratio"
    if roundness < settings.via_min_roundness:
        return "roundness"
    return ""


def _candidate_size_limits(
    settings: ContourExtractionSettings, expected_span: int
) -> tuple[float, float, float, float]:
    if settings.via_size_mode == "fixed" and settings.fixed_via_widths and settings.fixed_via_heights:
        widths = [float(value) for value in settings.fixed_via_widths if int(value) > 0]
        heights = [float(value) for value in settings.fixed_via_heights if int(value) > 0]
        min_width = max(1.0, min(widths) * 0.45)
        max_width = max(widths) * 1.65
        min_height = max(1.0, min(heights) * 0.45)
        max_height = max(heights) * 1.65
        return min_width, max_width, min_height, max_height
    min_width = float(settings.min_via_width if settings.min_via_width > 0 else max(2, expected_span * 0.35))
    min_height = float(settings.min_via_height if settings.min_via_height > 0 else max(2, expected_span * 0.35))
    max_width = float(settings.max_via_width if settings.max_via_width is not None else max(12, expected_span * 3.5))
    max_height = float(settings.max_via_height if settings.max_via_height is not None else max(12, expected_span * 3.5))
    return min_width, max_width, min_height, max_height


def _merge_via_candidates(
    candidates: list[_ViaCandidate],
    settings: ContourExtractionSettings,
) -> tuple[list[_ViaCandidate], list[_ViaCandidate]]:
    expected_span = _expected_via_span(settings)
    merge_distance = max(2.0, expected_span * 1.15)
    accepted: list[_ViaCandidate] = []
    duplicates: list[_ViaCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(_candidate_centers_close(candidate, existing, merge_distance) for existing in accepted):
            duplicates.append(candidate)
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda item: (item.center_y, item.center_x)), duplicates


def _candidate_centers_close(first: _ViaCandidate, second: _ViaCandidate, distance: float) -> bool:
    dx = first.center_x - second.center_x
    dy = first.center_y - second.center_y
    return (dx * dx + dy * dy) <= distance * distance


def _render_via_candidates_mask(
    shape: tuple[int, ...],
    candidates: list[_ViaCandidate],
    settings: ContourExtractionSettings,
) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    for candidate in candidates:
        box_width, box_height = _rendered_candidate_size(candidate, settings)
        center = (round(candidate.center_x), round(candidate.center_y))
        axes = (max(1, round(box_width / 2.0)), max(1, round(box_height / 2.0)))
        cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, 255, thickness=-1, lineType=cv2.LINE_8)
    return ensure_binary_mask(mask)


def _rendered_candidate_size(candidate: _ViaCandidate, settings: ContourExtractionSettings) -> tuple[float, float]:
    pairs = list(zip(settings.fixed_via_widths, settings.fixed_via_heights, strict=False))
    if settings.via_size_mode == "fixed" and pairs:
        best_width, best_height = min(
            pairs,
            key=lambda pair: (
                abs(float(pair[0]) - candidate.width) / max(1.0, float(pair[0]))
                + abs(float(pair[1]) - candidate.height) / max(1.0, float(pair[1]))
            ),
        )
        return float(best_width), float(best_height)
    return max(2.0, float(candidate.width)), max(2.0, float(candidate.height))


def _debug_candidates_from_via_candidates(
    candidates: list[_ViaCandidate],
    *,
    accepted: bool,
    reason: str | None = None,
) -> list[ContourDebugCandidate]:
    debug: list[ContourDebugCandidate] = []
    for index, candidate in enumerate(candidates):
        bbox = _candidate_bbox(candidate)
        debug.append(
            ContourDebugCandidate(
                contour_index=index,
                bbox=bbox,
                area=float(candidate.width * candidate.height),
                perimeter=float(2.0 * (candidate.width + candidate.height)),
                roundness=float(candidate.roundness),
                accepted=bool(accepted),
                reason=reason
                or candidate.reason
                or (f"accepted:{candidate.source}" if accepted else f"rejected:{candidate.source}"),
                source=str(candidate.source),
                score=float(candidate.score),
            )
        )
    return debug


def _candidate_bbox(candidate: _ViaCandidate) -> tuple[int, int, int, int]:
    left = round(candidate.center_x - candidate.width / 2.0)
    top = round(candidate.center_y - candidate.height / 2.0)
    return left, top, max(1, round(candidate.width)), max(1, round(candidate.height))


def _via_local_range_mask(
    gray: np.ndarray, low: int, high: int, settings: ContourExtractionSettings, *, bright: bool
) -> np.ndarray:
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

    seed_mask = cv2.bitwise_and(
        response_mask, cv2.dilate(intensity_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    )
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
    pipeline: PreprocessingPipeline | None = None,
    image_loader: Callable[[str], Any] = load_image_color,
    save_bundle: Callable[..., dict[str, str]] = save_result_bundle,
    include_images_in_result: bool = True,
) -> BatchImageResult:
    source = source_image if source_image is not None else image_loader(image_path)
    if preprocessed_image is not None:
        preprocessed = preprocessed_image
    else:
        active_pipeline = pipeline or PreprocessingPipeline.from_dict(pipeline_config)
        preprocessed = active_pipeline.apply(source)
    if contour_settings.object_type == "via" or contour_settings.output_mode == "box":
        mask, debug_candidates = build_via_vectorization_mask(preprocessed, contour_settings)
    else:
        mask = build_conductor_vectorization_mask(source, preprocessed, contour_settings)
        debug_candidates = []
    polygons = extract_polygons(mask, contour_settings)
    if not contour_settings.debug_enabled:
        debug_candidates = []
    debug_gradient_maps: dict[str, np.ndarray] = {}
    if contour_settings.debug_gradient_map_enabled or (
        contour_settings.debug_enabled and contour_settings.debug_gradient_map_enabled
    ):
        try:
            debug_gradient_maps = build_detection_debug_maps(source, preprocessed, contour_settings)
        except Exception:  # pragma: no cover - defensive: debug never breaks processing
            debug_gradient_maps = {}
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
        debug_gradient_maps=debug_gradient_maps if include_images_in_result else {},
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
        normalized[:, column_index] = np.clip((column - minimum) * (255.0 / (maximum - minimum)), 0, 255).astype(
            np.uint8
        )
    return normalized
