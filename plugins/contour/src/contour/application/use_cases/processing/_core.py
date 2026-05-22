from __future__ import annotations

import cProfile
import json
import hashlib
import io
import pstats
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

from ....contour_extractor import extract_polygons
from ....domain import PolygonData
from ....edge_detection import (
    build_gradient_elevation,
    normalize_edge_method,
    phase_congruency,
    ridge_response,
    scharr_magnitude,
    structured_edges,
)
from ....pipeline import PreprocessingPipeline
from ....serializers import save_result_bundle
from ....utils import ensure_binary_mask, ensure_uint8, load_image_color
from ...preview_cancellation import raise_if_preview_cancelled
from ....infrastructure.profiling import (
    processing_profiling_enabled,
    processing_top_lines,
    try_disable_profiler,
    try_enable_profiler,
)
from ...processing import (
    ALGORITHM_BACKEND_LEGACY,
    RECOGNITION_MODE_CONDUCTORS,
    RECOGNITION_MODE_DISABLED,
    RECOGNITION_MODE_VIA,
    VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG,
    VIA_SEARCH_MODE_HEURISTIC,
    VIA_SEARCH_MODE_TEMPLATE,
    BatchImageResult,
    ContourDebugCandidate,
    ContourExtractionSettings,
    DisplaySettings,
    SaveOptions,
    normalize_algorithm_backend,
    normalize_recognition_mode,
    normalize_via_search_mode,
)


@dataclass(frozen=True, slots=True)
class PreviewProcessingRequest:
    image_path: str
    pipeline_config: dict[str, Any]
    contour_settings: ContourExtractionSettings
    source_image: Any | None = None
    preprocessed_image: Any | None = None
    passthrough_polygons: tuple[PolygonData, ...] | None = None


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
    contrast: float = 0.0
    edge_strength: float = 0.0
    reason: str = ""


_VIA_DETECTION_CACHE: dict[tuple[str, str, str], Any] = {}
_VIA_DETECTION_CACHE_MAX_ITEMS = 8


def _make_image_signature(gray: np.ndarray) -> str:
    digest = hashlib.sha1(gray.tobytes()).hexdigest()
    return f"{gray.shape[0]}x{gray.shape[1]}:{digest}"


def _bounded_cache_set(key: tuple[str, str, str], value: Any) -> None:
    if key in _VIA_DETECTION_CACHE:
        _VIA_DETECTION_CACHE[key] = value
        return
    if len(_VIA_DETECTION_CACHE) >= _VIA_DETECTION_CACHE_MAX_ITEMS:
        oldest = next(iter(_VIA_DETECTION_CACHE))
        _VIA_DETECTION_CACHE.pop(oldest, None)
    _VIA_DETECTION_CACHE[key] = value


# ---------------------------------------------------------------------------
# Public signature / preview helpers
# ---------------------------------------------------------------------------


def build_preview_request_signature(request: PreviewProcessingRequest) -> tuple[str, str, str, int]:
    return (
        request.image_path,
        json.dumps(request.pipeline_config, ensure_ascii=False, sort_keys=True),
        json.dumps(request.contour_settings.to_dict(), ensure_ascii=False, sort_keys=True),
        len(request.passthrough_polygons or ()),
    )


def build_prepared_image_signature(request: PreparedImageRequest) -> tuple[str, str]:
    return (
        request.image_path,
        json.dumps(request.pipeline_config, ensure_ascii=False, sort_keys=True),
    )


def prepare_image_for_preview(source_image: Any, pipeline_config: dict[str, Any]) -> Any:
    raise_if_preview_cancelled()
    return PreprocessingPipeline.from_dict(pipeline_config).apply(source_image)


# ---------------------------------------------------------------------------
# Public mask builders
# ---------------------------------------------------------------------------


def apply_via_vectorization_mask(image: Any, settings: ContourExtractionSettings) -> Any:
    mask, _debug_candidates = build_via_vectorization_mask(image, settings)
    return mask


def build_conductor_vectorization_mask(
    source_image: Any,
    preprocessed_image: Any,
    settings: ContourExtractionSettings,
) -> np.ndarray:
    base_mask = ensure_binary_mask(preprocessed_image)
    legacy = str(getattr(settings, "algorithm_backend", "legacy")).lower() == "legacy"
    conductor_polygon = (
        settings.object_type == "conductor" and str(getattr(settings, "output_mode", "polygon")) != "box"
    )
    # Legacy conductors: binary mask from the pipeline only, then findContours — no gradient refinement.
    if legacy and conductor_polygon:
        return base_mask
    if settings.object_type == "via" or settings.output_mode == "box" or not settings.conductor_gradient_enabled:
        return base_mask
    return _refine_conductor_mask_by_gradient(source_image, base_mask, settings)


def build_via_vectorization_mask(
    image: Any, settings: ContourExtractionSettings
) -> tuple[Any, list[ContourDebugCandidate]]:
    """Detect via candidates and render them into a binary mask.

    Dispatches only to the selected modern mode: saved-template matching or
    heuristic local analysis. Old blob/top-hat/DoG detector stacks are not part
    of this runtime path.
    """

    if settings.object_type != "via" and settings.output_mode != "box":
        return ensure_binary_mask(image), []

    gray = _via_grayscale(image)
    if gray.size == 0:
        return ensure_binary_mask(image), []

    return _build_modern_via_vectorization_mask(gray, settings)


def _build_modern_via_vectorization_mask(
    gray: np.ndarray, settings: ContourExtractionSettings
) -> tuple[np.ndarray, list[ContourDebugCandidate]]:
    from ....vision.via.bright_tophat_dog import (
        BrightViaDetectorConfig,
        prepare_bright_via_candidates,
        score_bright_via_candidates,
    )
    from ....vision.via_detection.heuristic_detector import detect_vias_heuristic
    from ....vision.via_detection.result import DetectionResult, ViaDetection
    from ....vision.via_detection.settings_bridge import heuristic_config_from_settings, template_config_from_settings
    from ....vision.via_detection.template_detector import (
        detect_vias_template_raw,
        score_vias_template_raw,
    )

    mode = normalize_via_search_mode(settings.via_search_mode)
    image_sig = _make_image_signature(gray)
    if mode == VIA_SEARCH_MODE_TEMPLATE:
        tcfg = template_config_from_settings(settings)
        if not tcfg.templates:
            empty = np.zeros(gray.shape[:2], dtype=np.uint8)
            debug_candidates = [
                ContourDebugCandidate(
                    contour_index=0,
                    bbox=(0, 0, 0, 0),
                    accepted=False,
                    reason="Для режима поиска по шаблону добавьте хотя бы один шаблон",
                    source=VIA_SEARCH_MODE_TEMPLATE,
                    score=0.0,
                )
            ]
            return empty, debug_candidates
        heavy_key = (
            VIA_SEARCH_MODE_TEMPLATE,
            image_sig,
            json.dumps(
                {
                    "via_search_mode": mode,
                    "via_template_images": len(tcfg.templates),
                    "via_template_scale_min": float(tcfg.scale_min),
                    "via_template_scale_max": float(tcfg.scale_max),
                    "via_template_scale_step": float(tcfg.scale_step),
                },
                sort_keys=True,
            ),
        )
        raw_payload = _VIA_DETECTION_CACHE.get(heavy_key)
        if raw_payload is None:
            raw_payload = detect_vias_template_raw(gray, tcfg)
            _bounded_cache_set(heavy_key, raw_payload)
        raw_matches, shape = raw_payload
        result = score_vias_template_raw(raw_matches, shape, tcfg)
    elif mode == VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG:
        bright_config = BrightViaDetectorConfig.from_legacy_settings(settings)
        heavy_key = (
            VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG,
            image_sig,
            json.dumps(
                {
                    "via_search_mode": mode,
                    "bright_via_diameter_min": bright_config.diameter_min,
                    "bright_via_diameter_max": bright_config.diameter_max,
                    "bright_via_clahe_clip_limit": bright_config.clahe_clip_limit,
                    "bright_via_clahe_tile_grid_size": bright_config.clahe_tile_grid_size,
                    "bright_via_median_blur_kernel": bright_config.median_blur_kernel,
                    "bright_via_tophat_kernel_size": bright_config.tophat_kernel_size,
                    "bright_via_dog_sigma_small": bright_config.dog_sigma_small,
                    "bright_via_dog_sigma_large": bright_config.dog_sigma_large,
                    "bright_via_threshold_percentile": bright_config.threshold_percentile,
                    "bright_via_mask_combine_mode": bright_config.mask_combine_mode,
                    "bright_via_use_metal_mask": bright_config.use_metal_mask,
                    "bright_via_metal_constraint_mode": bright_config.metal_constraint_mode,
                },
                sort_keys=True,
            ),
        )
        prepared = _VIA_DETECTION_CACHE.get(heavy_key)
        if prepared is None:
            prepared = prepare_bright_via_candidates(gray, bright_config)
            _bounded_cache_set(heavy_key, prepared)
        bright = score_bright_via_candidates(prepared, bright_config)
        accepted = [
            ViaDetection(
                x=float(det.center[0]),
                y=float(det.center[1]),
                bbox=det.bbox,
                score=float(det.final_score),
                diameter_estimate=float((det.bbox[2] + det.bbox[3]) * 0.5),
                contrast=float(det.brightness_score),
                prominence=float(det.tophat_response + det.dog_response) * 0.5,
                compactness=float(det.circularity),
                aspect=float(det.aspect),
                polarity_hypothesis="bright",
                reject_reason=det.hard_reason or None,
            )
            for det in bright.detections
        ]
        rejected = [
            ViaDetection(
                x=float(det.center[0]),
                y=float(det.center[1]),
                bbox=det.bbox,
                score=float(det.final_score),
                diameter_estimate=float((det.bbox[2] + det.bbox[3]) * 0.5),
                contrast=float(det.brightness_score),
                prominence=float(det.tophat_response + det.dog_response) * 0.5,
                compactness=float(det.circularity),
                aspect=float(det.aspect),
                polarity_hypothesis="bright",
                reject_reason=det.hard_reason or det.status,
            )
            for det in bright.candidates
            if det not in bright.detections
        ]
        result = DetectionResult(
            method=VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG,
            accepted=accepted,
            rejected=rejected,
            debug_images=dict(bright.debug_images),
            parameters_snapshot={"config": repr(bright_config)},
        )
    else:
        result = detect_vias_heuristic(gray, heuristic_config_from_settings(settings))
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    for det in result.accepted:
        center = (round(det.x), round(det.y))
        r = det.diameter_estimate * 0.5 if getattr(det, "diameter_estimate", 0) else 0.0
        if r <= 0.0:
            _x, _y, bw, bh = det.bbox
            ax = max(1, int(bw * 0.5))
            ay = max(1, int(bh * 0.5))
        else:
            ax = ay = max(1, int(round(r)))
        cv2.ellipse(mask, center, (ax, ay), 0.0, 0.0, 360.0, 255, thickness=-1, lineType=cv2.LINE_8)
    debug_candidates: list[ContourDebugCandidate] = []
    idx = 0
    for det in result.accepted:
        debug_candidates.append(
            ContourDebugCandidate(
                contour_index=idx,
                bbox=det.bbox,
                area=float(det.bbox[2] * det.bbox[3]),
                perimeter=float(2.0 * (det.bbox[2] + det.bbox[3])),
                roundness=float(det.compactness * 100.0),
                accepted=True,
                reason=f"accepted:{mode}",
                source=mode,
                score=float(det.score),
            )
        )
        idx += 1
    for det in result.below_threshold:
        debug_candidates.append(
            ContourDebugCandidate(
                contour_index=idx,
                bbox=det.bbox,
                area=0.0,
                accepted=False,
                reason="below_threshold",
                source=mode,
                score=float(det.score),
            )
        )
        idx += 1
    for det in result.rejected:
        st = str(det.reject_reason or "hard_reject")
        debug_candidates.append(
            ContourDebugCandidate(
                contour_index=idx,
                bbox=det.bbox,
                area=0.0,
                accepted=False,
                reason=f"rejected:{st}",
                source=mode,
                score=float(getattr(det, "score", 0.0)),
            )
        )
        idx += 1
    return ensure_binary_mask(mask), debug_candidates


# ---------------------------------------------------------------------------
# Debug map builder (used by the gradient overlay / debug panel)
# ---------------------------------------------------------------------------


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
    Modern via debug maps are populated by the selected template or heuristic
    detector; old blob/top-hat response maps are intentionally not produced.
    """

    maps: dict[str, np.ndarray] = {}
    if source_image is None:
        return maps
    raise_if_preview_cancelled()

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

    vmode = normalize_via_search_mode(settings.via_search_mode)
    if (settings.object_type == "via" or settings.output_mode == "box") and vmode in (
        VIA_SEARCH_MODE_HEURISTIC,
        VIA_SEARCH_MODE_TEMPLATE,
        VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG,
    ):
        from ....vision.via.bright_tophat_dog import BrightViaDetectorConfig, detect_bright_vias
        from ....vision.via_detection.heuristic_detector import detect_vias_heuristic
        from ....vision.via_detection.settings_bridge import heuristic_config_from_settings, template_config_from_settings
        from ....vision.via_detection.template_detector import detect_vias_template

        try:
            if vmode == VIA_SEARCH_MODE_TEMPLATE:
                r = detect_vias_template(source_gray, template_config_from_settings(settings))
                dbg = dict(r.debug_images)
            elif vmode == VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG:
                r = detect_bright_vias(source_gray, BrightViaDetectorConfig.from_legacy_settings(settings))
                dbg = dict(r.debug_images)
            else:
                r = detect_vias_heuristic(source_gray, heuristic_config_from_settings(settings))
                dbg = dict(r.debug_images)
            for _guard in ("source_gray", "gradient_elevation", "gradient_color"):
                dbg.pop(_guard, None)
            maps.update(dbg)
        except Exception:  # pragma: no cover - defensive debug path
            pass

    maps["scharr"] = scharr_magnitude(source_gray)
    try:
        maps["phase_congruency"] = phase_congruency(source_gray)
    except Exception:  # pragma: no cover - numerical fallback
        maps["phase_congruency"] = np.zeros_like(source_gray, dtype=np.uint8)
    maps["structured"] = structured_edges(source_gray)
    maps["ridge"] = ridge_response(source_gray)

    if settings.object_type == "via" or settings.output_mode == "box":
        mask, _candidates = build_via_vectorization_mask(preprocessed_image, settings)
        maps["mask"] = ensure_binary_mask(mask)
    else:
        mask = build_conductor_vectorization_mask(source_image, preprocessed_image, settings)
        maps["mask"] = ensure_binary_mask(mask)
        if include_color_maps:
            maps["conductor_gradient_elevation"] = _conductor_gradient_elevation(source_gray, settings)
    return maps


# ---------------------------------------------------------------------------
# Conductor refinement helpers (unchanged)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Edge-method resolvers + small image utilities
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Via detection pipeline
# ---------------------------------------------------------------------------


def _detect_via_candidates(
    gray: np.ndarray, settings: ContourExtractionSettings
) -> tuple[list[_ViaCandidate], list[_ViaCandidate]]:
    expected_span = _expected_via_span(settings)
    radii = _via_candidate_radii(settings, expected_span)
    if not radii:
        return [], []

    enhanced = _clahe_gray(gray) if min(gray.shape[:2]) >= 8 else ensure_uint8(gray)
    polarity_scans = _via_polarity_scans(settings)
    line_suppression = max(0.0, min(1.0, float(settings.via_spot_line_suppression)))
    search_mode = normalize_via_search_mode(settings.via_search_mode)
    run_blob = False
    run_template = search_mode == VIA_SEARCH_MODE_TEMPLATE
    edge_method = _resolve_via_edge_method(settings)
    gradient = build_gradient_elevation(enhanced, edge_method)
    if gradient.size == 0:
        gradient = np.zeros_like(gray, dtype=np.uint8)

    raw_candidates: list[_ViaCandidate] = []
    if run_blob:
        for low, high, bright in polarity_scans:
            intensity_mask = _intensity_mask([gray, enhanced], low, high)
            gate = _intensity_gate(intensity_mask, expected_span)

            blob = _multiscale_blob_response(enhanced, expected_span, bright=bright, line_suppression=line_suppression)
            ring = _ring_template_response(gradient, radii)
            combined = cv2.max(blob, ring)
            gated_combined = cv2.bitwise_and(combined, gate) if cv2.countNonZero(gate) > 0 else combined

            isolation = _build_isolation_context(intensity_mask, gray, bright=bright)

            min_distance = max(2, round(expected_span * 0.85))
            peaks = _extract_response_peaks(gated_combined, min_distance=min_distance)
            for cy, cx in peaks:
                candidate = _verify_peak(
                    gray=gray,
                    gradient=gradient,
                    blob_response=blob,
                    ring_response=ring,
                    cx=cx,
                    cy=cy,
                    radii=radii,
                    bright=bright,
                    isolation=isolation,
                )
                if candidate is not None:
                    raw_candidates.append(candidate)

    template_arrays = _via_saved_template_arrays(settings)
    if run_template and template_arrays:
        for low, high, bright in polarity_scans:
            intensity_mask = _intensity_mask([gray, enhanced], low, high)
            gate = _intensity_gate(intensity_mask, expected_span)
            tmpl_candidates = _template_match_candidates(
                gray=gray,
                gradient=gradient,
                templates=template_arrays,
                gate=gate,
                radii=radii,
                expected_span=expected_span,
                settings=settings,
                bright=bright,
            )
            raw_candidates.extend(tmpl_candidates)

    accepted: list[_ViaCandidate] = []
    rejected: list[_ViaCandidate] = []
    for candidate in raw_candidates:
        normalized = _apply_via_filters(candidate, settings, expected_span)
        if normalized.reason:
            rejected.append(normalized)
        else:
            accepted.append(normalized)
    return accepted, rejected


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


def _log_blob_response(gray: np.ndarray, expected_span: int, *, bright: bool) -> np.ndarray:
    data = ensure_uint8(gray).astype(np.float32)
    result = np.zeros_like(data, dtype=np.float32)
    sigmas = (
        max(0.65, expected_span * 0.20),
        max(0.85, expected_span * 0.34),
        max(1.10, expected_span * 0.50),
    )
    for sigma in sigmas:
        blurred = cv2.GaussianBlur(data, (0, 0), sigma)
        laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        local = (-laplacian if bright else laplacian) * float(sigma * sigma)
        local = np.maximum(local, 0.0)
        values = local[local > 0.0]
        if values.size == 0:
            continue
        # A confident bright via produces LoG(x)*sigma^2 on the order of the
        # intensity contrast. Do not normalize below ~40 intensity units,
        # otherwise weak noise blobs get amplified to near-saturation.
        scale = max(40.0, float(np.percentile(values, 99.5)))
        result = np.maximum(result, np.clip(local * (255.0 / scale), 0.0, 255.0))
    return result.astype(np.uint8)


def _ring_template_response(gradient: np.ndarray, radii: list[int]) -> np.ndarray:
    """Maximum response of circle-template matched against gradient elevation."""

    if gradient.size == 0 or not radii:
        return np.zeros_like(gradient, dtype=np.uint8)
    search = ensure_uint8(gradient).astype(np.float32) / 255.0
    response = np.zeros(gradient.shape, dtype=np.float32)
    for radius in radii:
        radius = max(2, int(radius))
        size = radius * 2 + 5
        if gradient.shape[0] < size or gradient.shape[1] < size:
            continue
        center = size // 2
        template = np.zeros((size, size), dtype=np.float32)
        thickness = max(1, round(radius * 0.22))
        cv2.circle(template, (center, center), radius, 1.0, thickness=thickness, lineType=cv2.LINE_AA)
        if float(template.sum()) <= 0.0:
            continue
        scores = cv2.matchTemplate(search, template, cv2.TM_CCORR_NORMED)
        scores = np.nan_to_num(scores, copy=False)
        if scores.size == 0:
            continue
        placed = np.zeros_like(response, dtype=np.float32)
        ph, pw = scores.shape
        placed[center : center + ph, center : center + pw] = scores
        response = np.maximum(response, placed)
    return np.clip(response * 255.0, 0.0, 255.0).astype(np.uint8)


def _extract_response_peaks(response: np.ndarray, *, min_distance: int) -> list[tuple[int, int]]:
    """Return list of ``(y, x)`` local maxima with NMS, sorted by descending value."""

    if response.size == 0:
        return []
    data = ensure_uint8(response)
    nonzero_values = data[data > 0]
    if nonzero_values.size == 0:
        return []
    # Require peaks to stand out against the local background. The old cutoff
    # (50 % of the 80th percentile) was dominated by noise on low-signal
    # images; raising both the floor and the percentile keeps the peak list
    # short and focused on genuine candidates.
    cutoff = max(30, int(np.percentile(nonzero_values, 90) * 0.65))
    kernel_size = max(3, int(min_distance) * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(data, kernel)
    local_max = (data >= dilated) & (data >= cutoff)
    ys, xs = np.where(local_max)
    if len(xs) == 0:
        return []
    values = data[ys, xs]
    order = np.argsort(values)[::-1]
    used = np.zeros(data.shape, dtype=bool)
    selected: list[tuple[int, int]] = []
    radius = max(1, int(min_distance))
    max_peaks = 2000
    for idx in order:
        y = int(ys[idx])
        x = int(xs[idx])
        if used[y, x]:
            continue
        selected.append((y, x))
        if len(selected) >= max_peaks:
            break
        y0 = max(0, y - radius)
        y1 = min(data.shape[0], y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(data.shape[1], x + radius + 1)
        used[y0:y1, x0:x1] = True
    return selected


@dataclass(frozen=True, slots=True)
class _IsolationContext:
    """Labels / stats of the foreground mask used to reject trace-like shapes.

    ``active`` is ``False`` when the user-provided intensity range covers most
    of the image (so the connected-component gate would wrongly reject
    everything) or when there is nothing to anchor on.
    """

    active: bool
    labels: np.ndarray
    stats: np.ndarray


_EMPTY_ISOLATION: _IsolationContext = _IsolationContext(
    active=False,
    labels=np.zeros((0, 0), dtype=np.int32),
    stats=np.zeros((0, 5), dtype=np.int32),
)


def _build_isolation_context(intensity_mask: np.ndarray, gray: np.ndarray, *, bright: bool) -> _IsolationContext:
    """Compute connected-component labels for the bright/dark anchor mask."""

    if intensity_mask.size == 0:
        return _EMPTY_ISOLATION
    mask_area = float(cv2.countNonZero(intensity_mask))
    total_area = float(intensity_mask.size)
    if mask_area <= 0.0:
        return _EMPTY_ISOLATION
    ratio = mask_area / total_area
    if ratio > 0.55:
        # The user-supplied range covers most of the image; the component gate
        # would be meaningless. Fall back to an Otsu-like threshold derived
        # from the target polarity. If even that is too permissive we give up
        # on the isolation check for this scan.
        source = ensure_uint8(gray)
        try:
            threshold_value, otsu = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except cv2.error:
            return _EMPTY_ISOLATION
        del threshold_value
        mask = otsu if bright else cv2.bitwise_not(otsu)
        mask = cv2.bitwise_and(mask, intensity_mask)
        mask_area = float(cv2.countNonZero(mask))
        if mask_area <= 0.0 or mask_area / total_area > 0.55:
            return _EMPTY_ISOLATION
    else:
        mask = intensity_mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return _EMPTY_ISOLATION
    return _IsolationContext(active=True, labels=labels, stats=stats)


def _component_extent(isolation: _IsolationContext, cx: int, cy: int) -> tuple[int, int, int] | None:
    """Return (area, width, height) of the component at (cx, cy), or ``None``."""

    if not isolation.active:
        return None
    labels = isolation.labels
    if cy < 0 or cx < 0 or cy >= labels.shape[0] or cx >= labels.shape[1]:
        return None
    label = int(labels[cy, cx])
    if label == 0:
        return None
    stats = isolation.stats
    if label >= stats.shape[0]:
        return None
    return (
        int(stats[label, cv2.CC_STAT_AREA]),
        int(stats[label, cv2.CC_STAT_WIDTH]),
        int(stats[label, cv2.CC_STAT_HEIGHT]),
    )


def _refine_center(gray: np.ndarray, cx: int, cy: int, radius: int, *, bright: bool) -> tuple[float, float]:
    """Return an intensity-weighted centroid in the neighbourhood of ``(cx, cy)``.

    The blob/ring response peaks do not always coincide with the geometric
    centre of a via - they can be pulled towards a neighbouring trace or the
    brighter half of an asymmetric contact. We therefore re-centre each
    candidate on the centroid of the brightest (or darkest for dark vias)
    pixels inside a local window. Movement is clamped so that noisy patches
    cannot teleport the candidate across the image.
    """

    r = max(2, int(radius))
    half = max(r, round(r * 1.1))
    height, width = gray.shape[:2]
    x0 = max(0, int(cx) - half)
    x1 = min(width, int(cx) + half + 1)
    y0 = max(0, int(cy) - half)
    y1 = min(height, int(cy) + half + 1)
    if x1 <= x0 or y1 <= y0:
        return float(cx), float(cy)
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    lo = float(patch.min())
    hi = float(patch.max())
    span = hi - lo
    if span < 5.0:
        return float(cx), float(cy)
    threshold = lo + span * 0.5
    weights = patch - threshold if bright else threshold - patch
    np.maximum(weights, 0.0, out=weights)
    total = float(weights.sum())
    if total <= 1e-6:
        return float(cx), float(cy)
    ys = np.arange(patch.shape[0], dtype=np.float32).reshape(-1, 1)
    xs = np.arange(patch.shape[1], dtype=np.float32).reshape(1, -1)
    mean_x = float((weights * xs).sum() / total)
    mean_y = float((weights * ys).sum() / total)
    new_cx = float(x0) + mean_x
    new_cy = float(y0) + mean_y
    dx = new_cx - float(cx)
    dy = new_cy - float(cy)
    dist = float(np.hypot(dx, dy))
    max_move = float(r) * 0.9
    if dist > max_move and dist > 1e-6:
        scale = max_move / dist
        new_cx = float(cx) + dx * scale
        new_cy = float(cy) + dy * scale
    return new_cx, new_cy


def _verify_peak(
    *,
    gray: np.ndarray,
    gradient: np.ndarray,
    blob_response: np.ndarray,
    ring_response: np.ndarray,
    cx: int,
    cy: int,
    radii: list[int],
    bright: bool,
    isolation: _IsolationContext,
) -> _ViaCandidate | None:
    height, width = gray.shape[:2]
    if cx < 0 or cy < 0 or cx >= width or cy >= height:
        return None

    extent = _component_extent(isolation, cx, cy)
    if isolation.active and extent is None:
        # The peak sits outside every bright component - it cannot be a via
        # under the current polarity/range.
        return None

    best_score = -1.0
    best_candidate: _ViaCandidate | None = None
    for radius in radii:
        radius = max(2, int(radius))
        contrast = _radial_contrast(gray, cx, cy, radius, bright=bright)
        edge_strength, angular_coverage = _edge_ring_metrics(gradient, cx, cy, radius)

        # 80 units of absolute intensity difference is a confident via.
        contrast_score = max(0.0, min(1.0, contrast / 80.0))
        edge_score = max(0.0, min(1.0, edge_strength / 80.0))
        coverage_score = max(0.0, min(1.0, angular_coverage))

        isolation_score = _isolation_score(extent, radius)

        # Score from absolute circle-fit evidence only; peak response is used
        # for localization, not for scoring (otherwise any bright speckle on
        # texture wins because the response is percentile-normalized).
        score = contrast_score * 0.35 + edge_score * 0.25 + coverage_score * 0.20 + isolation_score * 0.20
        if score > best_score:
            best_score = score
            roundness = float(coverage_score * 100.0)
            best_candidate = _ViaCandidate(
                center_x=float(cx),
                center_y=float(cy),
                width=float(radius * 2 + 1),
                height=float(radius * 2 + 1),
                score=score,
                source="blob",
                roundness=roundness,
                contrast=float(contrast),
                edge_strength=float(edge_strength),
            )
    if best_candidate is not None:
        winning_radius = max(2, int((best_candidate.width - 1) // 2))
        refined_cx, refined_cy = _refine_center(gray, cx, cy, winning_radius, bright=bright)
        best_candidate = replace(best_candidate, center_x=refined_cx, center_y=refined_cy)
    return best_candidate


def _isolation_score(extent: tuple[int, int, int] | None, radius: int) -> float:
    """Map the connected-component extent around the candidate to ``[0, 1]``.

    A via sits on an isolated bright (or dark) island. If the island that
    contains the peak is much larger than the expected via, the peak is most
    likely a trace extension or the corner of a trace and should be penalised.
    """

    if extent is None:
        return 1.0
    _area, comp_width, comp_height = extent
    comp_max = float(max(comp_width, comp_height))
    if comp_max <= 0.0:
        return 1.0
    # Give a full score while the island fits within ~2.5 via diameters; drop
    # to zero once the island grows past ~5 diameters (clearly a trace).
    diameter = float(radius * 2 + 1)
    lo = diameter * 2.5
    hi = diameter * 5.0
    if comp_max <= lo:
        return 1.0
    if comp_max >= hi:
        return 0.0
    return float(1.0 - (comp_max - lo) / max(1e-6, hi - lo))


def _radial_contrast(gray: np.ndarray, cx: int, cy: int, radius: int, *, bright: bool) -> float:
    radius = max(2, int(radius))
    padding = max(2, round(radius * 0.7))
    left = max(0, int(cx) - radius - padding)
    top = max(0, int(cy) - radius - padding)
    right = min(gray.shape[1], int(cx) + radius + padding + 1)
    bottom = min(gray.shape[0], int(cy) + radius + padding + 1)
    if right <= left or bottom <= top:
        return 0.0
    patch = ensure_uint8(gray)[top:bottom, left:right].astype(np.float32)
    yy, xx = np.ogrid[top - cy : bottom - cy, left - cx : right - cx]
    distance_sq = xx * xx + yy * yy
    inner_mask = distance_sq <= max(1.0, radius * 0.55) ** 2
    outer_lo = (radius * 1.15) ** 2
    outer_hi = (radius * 1.75) ** 2
    outer_mask = (distance_sq >= outer_lo) & (distance_sq <= outer_hi)
    if not np.any(inner_mask) or not np.any(outer_mask):
        return 0.0
    inner_mean = float(np.mean(patch[inner_mask]))
    outer_mean = float(np.mean(patch[outer_mask]))
    return inner_mean - outer_mean if bright else outer_mean - inner_mean


def _edge_ring_metrics(gradient: np.ndarray, cx: int, cy: int, radius: int) -> tuple[float, float]:
    """Return (mean ring gradient, angular coverage in [0..1])."""

    radius = max(2, int(radius))
    padding = max(2, round(radius * 0.35))
    left = max(0, int(cx) - radius - padding)
    top = max(0, int(cy) - radius - padding)
    right = min(gradient.shape[1], int(cx) + radius + padding + 1)
    bottom = min(gradient.shape[0], int(cy) + radius + padding + 1)
    if right <= left or bottom <= top or gradient.size == 0:
        return 0.0, 0.0
    patch = gradient[top:bottom, left:right]
    if patch.size == 0:
        return 0.0, 0.0
    yy, xx = np.ogrid[top - cy : bottom - cy, left - cx : right - cx]
    grid_x = np.broadcast_to(xx.astype(np.float32), patch.shape)
    grid_y = np.broadcast_to(yy.astype(np.float32), patch.shape)
    distance = np.sqrt(grid_x * grid_x + grid_y * grid_y)
    inner_radius = max(1.0, float(radius) - max(1.0, radius * 0.30))
    outer_radius = float(radius) + max(1.0, radius * 0.30)
    ring_mask = (distance >= inner_radius) & (distance <= outer_radius)
    if not np.any(ring_mask):
        return 0.0, 0.0
    ring_values = patch[ring_mask].astype(np.float32)
    mean_strength = float(ring_values.mean()) if ring_values.size else 0.0
    threshold = max(20.0, mean_strength * 1.10)
    strong_mask = ring_mask & (patch.astype(np.float32) >= threshold)
    coverage = _angular_coverage(strong_mask, grid_x, grid_y, sectors=16)
    return mean_strength, coverage


def _angular_coverage(mask: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, *, sectors: int) -> float:
    if sectors <= 0 or not np.any(mask):
        return 0.0
    angles = np.arctan2(grid_y[mask], grid_x[mask])
    bins = np.floor(((angles + np.pi) / (2.0 * np.pi)) * sectors).astype(np.int32)
    bins = np.clip(bins, 0, sectors - 1)
    return float(np.unique(bins).size) / float(sectors)


# ---------------------------------------------------------------------------
# Optional template matching pass
# ---------------------------------------------------------------------------


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


def _normalize_template_for_matching(image: np.ndarray) -> np.ndarray:
    data = ensure_uint8(image)
    if data.size == 0:
        return data
    minimum = int(data.min())
    maximum = int(data.max())
    if maximum <= minimum:
        return data
    return np.clip((data.astype(np.float32) - minimum) * (255.0 / float(maximum - minimum)), 0, 255).astype(np.uint8)


def _template_match_candidates(
    *,
    gray: np.ndarray,
    gradient: np.ndarray,
    templates: list[np.ndarray],
    gate: np.ndarray,
    radii: list[int],
    expected_span: int,
    settings: ContourExtractionSettings,
    bright: bool,
) -> list[_ViaCandidate]:
    if not templates or gray.size == 0:
        return []
    search_image = _normalize_template_for_matching(gray)
    if cv2.countNonZero(gate) > 0:
        search_image = cv2.bitwise_and(search_image, gate)
    if cv2.countNonZero(search_image) == 0:
        return []
    min_score = max(0.0, min(1.0, float(settings.via_template_min_score)))
    candidates: list[_ViaCandidate] = []
    for template in templates:
        if search_image.shape[0] < template.shape[0] or search_image.shape[1] < template.shape[1]:
            continue
        prepared_template = _normalize_template_for_matching(template)
        scores = cv2.matchTemplate(search_image, prepared_template, cv2.TM_CCOEFF_NORMED)
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
        radius = max(2, min(template.shape[:2]) // 3)
        for y_coord, x_coord in zip(ys, xs, strict=False):
            cx = int(x_coord) + template.shape[1] // 2
            cy = int(y_coord) + template.shape[0] // 2
            if cx >= gray.shape[1] or cy >= gray.shape[0]:
                continue
            if cv2.countNonZero(gate) > 0 and gate[cy, cx] == 0:
                continue
            refined_cx, refined_cy = _refine_center(gray, cx, cy, radius, bright=bright)
            refined_cx_int = round(refined_cx)
            refined_cy_int = round(refined_cy)
            contrast = _radial_contrast(gray, refined_cx_int, refined_cy_int, radius, bright=bright)
            edge_strength, coverage = _edge_ring_metrics(gradient, refined_cx_int, refined_cy_int, radius)
            template_score = float(scores[y_coord, x_coord])
            fit_score = (
                max(0.0, min(1.0, contrast / 80.0)) * 0.40
                + max(0.0, min(1.0, edge_strength / 80.0)) * 0.30
                + max(0.0, min(1.0, coverage)) * 0.30
            )
            score = max(fit_score, template_score * 0.6 + fit_score * 0.4)
            candidates.append(
                _ViaCandidate(
                    center_x=refined_cx,
                    center_y=refined_cy,
                    width=float(template.shape[1]),
                    height=float(template.shape[0]),
                    score=score,
                    source="template",
                    roundness=max(60.0, coverage * 100.0),
                    contrast=float(contrast),
                    edge_strength=float(edge_strength),
                )
            )
    return candidates


# ---------------------------------------------------------------------------
# Filtering, NMS and rendering
# ---------------------------------------------------------------------------


def _apply_via_filters(
    candidate: _ViaCandidate, settings: ContourExtractionSettings, expected_span: int
) -> _ViaCandidate:
    width = max(1.0, float(candidate.width))
    height = max(1.0, float(candidate.height))
    reason = _via_candidate_rejection_reason(width, height, candidate.roundness, settings, expected_span)
    if not reason:
        min_contrast = max(0.0, float(settings.via_min_contrast))
        min_edge_coverage = max(0.0, min(1.0, float(settings.via_min_edge_coverage)))
        coverage = max(0.0, min(1.0, candidate.roundness / 100.0))
        if min_contrast > 0.0 and candidate.contrast < min_contrast:
            reason = "low_contrast"
        elif min_edge_coverage > 0.0 and coverage < min_edge_coverage:
            reason = "low_edge_coverage"
        elif candidate.score < float(settings.via_min_score):
            reason = "low_score"
    return _ViaCandidate(
        center_x=float(candidate.center_x),
        center_y=float(candidate.center_y),
        width=width,
        height=height,
        score=float(candidate.score),
        source=candidate.source,
        roundness=float(candidate.roundness),
        contrast=float(candidate.contrast),
        edge_strength=float(candidate.edge_strength),
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


def _iou_nms(
    candidates: list[_ViaCandidate], *, iou_threshold: float
) -> tuple[list[_ViaCandidate], list[_ViaCandidate]]:
    if not candidates:
        return [], []
    accepted: list[_ViaCandidate] = []
    duplicates: list[_ViaCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        is_duplicate = False
        for existing in accepted:
            if _candidate_iou(candidate, existing) >= iou_threshold:
                is_duplicate = True
                break
        if is_duplicate:
            duplicates.append(candidate)
        else:
            accepted.append(candidate)
    accepted.sort(key=lambda item: (item.center_y, item.center_x))
    return accepted, duplicates


def _candidate_iou(first: _ViaCandidate, second: _ViaCandidate) -> float:
    a_left = first.center_x - first.width / 2.0
    a_top = first.center_y - first.height / 2.0
    a_right = a_left + first.width
    a_bottom = a_top + first.height
    b_left = second.center_x - second.width / 2.0
    b_top = second.center_y - second.height / 2.0
    b_right = b_left + second.width
    b_bottom = b_top + second.height
    intersect_left = max(a_left, b_left)
    intersect_top = max(a_top, b_top)
    intersect_right = min(a_right, b_right)
    intersect_bottom = min(a_bottom, b_bottom)
    if intersect_right <= intersect_left or intersect_bottom <= intersect_top:
        return 0.0
    intersection = (intersect_right - intersect_left) * (intersect_bottom - intersect_top)
    union = first.width * first.height + second.width * second.height - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


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


def _should_run_sem_dual_branch(settings: ContourExtractionSettings) -> bool:
    """Return ``True`` when legacy conductor flow should also merge template vias.

    The SEM vision backend uses a separate entry point; dual-branch merging only
    applies on the legacy path so template vias can be combined with conductor masks.
    """

    if str(getattr(settings, "algorithm_backend", "legacy")).lower() == "sem":
        return False
    if settings.object_type == "via" or settings.output_mode == "box":
        return False
    if settings.extraction_profile != "conductors":
        return False
    if normalize_via_search_mode(settings.via_search_mode) != VIA_SEARCH_MODE_TEMPLATE:
        return False
    return bool(settings.via_template_images)


def _build_via_settings_for_dual_branch(settings: ContourExtractionSettings) -> ContourExtractionSettings:
    return replace(
        settings,
        extraction_profile="vias",
        object_type="via",
        output_mode="box",
    )


def _component_map(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    binary = ensure_binary_mask(mask)
    if cv2.countNonZero(binary) == 0:
        return None
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    if count <= 1:
        return None
    return labels, stats


def _filter_vias_by_conductor_linearity(vias: list[PolygonData], conductor_mask: np.ndarray) -> list[PolygonData]:
    """Reject likely false vias sitting on long, straight conductor segments."""

    component_payload = _component_map(conductor_mask)
    if component_payload is None:
        return vias
    labels, stats = component_payload
    accepted: list[PolygonData] = []
    for polygon in vias:
        x_coord, y_coord, width, height = polygon.bbox
        center_x = round(x_coord + width / 2.0)
        center_y = round(y_coord + height / 2.0)
        if center_x < 0 or center_y < 0 or center_y >= labels.shape[0] or center_x >= labels.shape[1]:
            accepted.append(polygon)
            continue
        label = int(labels[center_y, center_x])
        if label <= 0 or label >= stats.shape[0]:
            accepted.append(polygon)
            continue
        comp_width = max(1, int(stats[label, cv2.CC_STAT_WIDTH]))
        comp_height = max(1, int(stats[label, cv2.CC_STAT_HEIGHT]))
        comp_area = max(1, int(stats[label, cv2.CC_STAT_AREA]))
        elongation = float(max(comp_width, comp_height)) / float(min(comp_width, comp_height))
        via_area = max(1.0, float(width * height))
        comp_to_via_area = float(comp_area) / via_area
        # If the candidate center lies on a very elongated component and that
        # component is much larger than via size, this is usually a trace tip /
        # edge artifact rather than a real contact.
        if elongation > 7.0 and comp_to_via_area > 16.0:
            continue
        accepted.append(polygon)
    return accepted


def _merge_dual_branch_polygons(
    conductor_polygons: list[PolygonData], via_polygons: list[PolygonData]
) -> list[PolygonData]:
    merged: list[PolygonData] = []
    for polygon in conductor_polygons:
        merged.append(polygon.clone())
    next_id = max((polygon.id for polygon in merged), default=0) + 1
    for polygon in via_polygons:
        clone = polygon.clone()
        clone.id = next_id
        clone.parent_id = None
        merged.append(clone)
        next_id += 1
    return merged


def _build_metalization_mask(gray: np.ndarray, settings: ContourExtractionSettings) -> np.ndarray:
    if gray.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    from ....vision.metal_recovery.detector import build_metal_extraction_mask
    from ....vision.metal_recovery import metal_recovery_config_from_settings

    cfg = metal_recovery_config_from_settings(settings)
    mask, _dbg = build_metal_extraction_mask(gray, cfg)
    return mask


def _metal_preview_batch_result(
    *,
    image_path: str,
    pipeline_config: dict[str, Any],
    contour_settings: ContourExtractionSettings,
    source: Any,
    preprocessed: Any,
    include_images_in_result: bool,
    output_directory: str | None,
    save_options: SaveOptions | None,
    display_settings: DisplaySettings | None,
    save_bundle: Callable[..., dict[str, str]],
) -> BatchImageResult:
    gray = _via_grayscale(preprocessed)
    mask = _build_metalization_mask(gray, contour_settings)
    min_area = max(0.0, float(getattr(contour_settings, "metal_min_object_area", 30.0)))
    metal_settings = replace(
        contour_settings,
        min_area=min_area,
        object_type="conductor",
        recognition_mode=RECOGNITION_MODE_VIA,
    )
    show_contours = bool(getattr(contour_settings, "metal_display_show_contours", True))
    polygons: list[PolygonData] = extract_polygons(mask, metal_settings) if show_contours else []
    debug_gradient_maps: dict[str, np.ndarray] = {}
    if bool(getattr(contour_settings, "metal_display_show_mask", True)) and include_images_in_result:
        debug_gradient_maps["metal_mask"] = mask
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
    return BatchImageResult(
        image_path=image_path,
        source_image=source if include_images_in_result else None,
        preprocessed_image=preprocessed if include_images_in_result else None,
        pipeline_config=dict(pipeline_config),
        mask_image=mask if include_images_in_result else None,
        polygons=polygons,
        debug_candidates=[],
        debug_gradient_maps=debug_gradient_maps if include_images_in_result else {},
        saved_files=saved_files,
    )


# ---------------------------------------------------------------------------
# Structural metal recovery (legacy backend + conductors recognition)
# ---------------------------------------------------------------------------


def _use_structural_metal_recovery(settings: ContourExtractionSettings) -> bool:
    if not bool(getattr(settings, "metal_structural_pipeline", False)):
        return False
    if normalize_algorithm_backend(getattr(settings, "algorithm_backend", "")) != ALGORITHM_BACKEND_LEGACY:
        return False
    if settings.object_type != "conductor" or str(getattr(settings, "output_mode", "polygon")) == "box":
        return False
    return normalize_recognition_mode(getattr(settings, "recognition_mode", "")) == RECOGNITION_MODE_CONDUCTORS


def _run_structural_metal_recovery(
    preprocessed: Any,
    contour_settings: ContourExtractionSettings,
) -> tuple[list[PolygonData], np.ndarray, dict[str, np.ndarray], dict[str, list[PolygonData]]]:
    from ....vision.metal_recovery import detect_metalization, metal_recovery_config_from_settings

    gray = ensure_uint8(_via_grayscale(preprocessed))
    cfg = metal_recovery_config_from_settings(contour_settings)
    mr = detect_metalization(gray, cfg)
    mask = mr.debug_images.get("metal_filtered_mask")
    if mask is None:
        mask = mr.debug_images["metal_binary_mask"]
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = ensure_binary_mask(mask)

    debug_maps: dict[str, np.ndarray] = dict(mr.debug_images)
    if bool(getattr(contour_settings, "metal_display_show_mask", True)):
        debug_maps["metal_mask"] = mr.debug_images["metal_binary_mask"]

    show_c = bool(getattr(contour_settings, "metal_display_show_conductors", True))
    polygons = list(mr.accepted) if show_c else []

    overlays: dict[str, list[PolygonData]] = {"rejected": [], "suspicious": [], "border": []}
    for r in mr.rejected:
        p = r.polygon.clone()
        p.category = "metal_rejected"
        p.reject_reason = str(getattr(r, "reject_reason", "") or "")
        overlays["rejected"].append(p)
    for r in mr.suspicious:
        p = r.polygon.clone()
        p.category = "metal_suspicious"
        p.reject_reason = str(getattr(r, "reject_reason", "") or "")
        overlays["suspicious"].append(p)
    for r in mr.border:
        p = r.polygon.clone()
        p.category = "metal_border"
        br = str(getattr(r, "reject_reason", "") or "").strip()
        if not br:
            bx, by, bw, bh = p.bbox
            br = (
                "Касание границы кадра: контур касается края изображения; "
                f"полигон в прямоугольнике x={int(bx)}, y={int(by)}, "
                f"ширина={int(bw)} px, высота={int(bh)} px (основной проводник также в списке принятых)"
            )
        p.reject_reason = br
        overlays["border"].append(p)
    for layer_key, polys in (mr.wide_gradient_overlays or {}).items():
        if not polys:
            continue
        overlays.setdefault(layer_key, []).extend(p.clone() for p in polys)

    return polygons, mask, debug_maps, overlays


# ---------------------------------------------------------------------------
# Top-level batch entry point
# ---------------------------------------------------------------------------


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
    passthrough_polygons: list[PolygonData] | None = None,
) -> BatchImageResult:
    if processing_profiling_enabled():
        profiler = cProfile.Profile()
        profiler_enabled = try_enable_profiler(profiler)
        try:
            return _process_image_path_impl(
                image_path=image_path,
                pipeline_config=pipeline_config,
                contour_settings=contour_settings,
                output_directory=output_directory,
                save_options=save_options,
                display_settings=display_settings,
                source_image=source_image,
                preprocessed_image=preprocessed_image,
                pipeline=pipeline,
                image_loader=image_loader,
                save_bundle=save_bundle,
                include_images_in_result=include_images_in_result,
                passthrough_polygons=passthrough_polygons,
            )
        finally:
            if not profiler_enabled:
                print(f"[contour profiling] image={image_path} skipped=yes reason=cprofile_already_active")
            else:
                try_disable_profiler(profiler)
                top_lines = processing_top_lines()
                stream = io.StringIO()
                stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
                stats.print_stats(top_lines)
                print(f"[contour profiling] image={image_path} top={top_lines}")
                print(stream.getvalue())

    return _process_image_path_impl(
        image_path=image_path,
        pipeline_config=pipeline_config,
        contour_settings=contour_settings,
        output_directory=output_directory,
        save_options=save_options,
        display_settings=display_settings,
        source_image=source_image,
        preprocessed_image=preprocessed_image,
        pipeline=pipeline,
        image_loader=image_loader,
        save_bundle=save_bundle,
        include_images_in_result=include_images_in_result,
        passthrough_polygons=passthrough_polygons,
    )


def _process_image_path_impl(
    *,
    image_path: str,
    pipeline_config: dict[str, Any],
    contour_settings: ContourExtractionSettings,
    output_directory: str | None = None,
    save_options: SaveOptions | None = None,
    display_settings: DisplaySettings | None = None,
    source_image: Any | None = None,
    preprocessed_image: Any | None = None,
    pipeline: PreprocessingPipeline | None = None,
    image_loader: Callable[[str], Any] = load_image_color,
    save_bundle: Callable[..., dict[str, str]] = save_result_bundle,
    include_images_in_result: bool = True,
    passthrough_polygons: list[PolygonData] | None = None,
) -> BatchImageResult:
    raise_if_preview_cancelled()
    source = source_image if source_image is not None else image_loader(image_path)
    raise_if_preview_cancelled()
    if preprocessed_image is not None:
        preprocessed = preprocessed_image
    else:
        active_pipeline = pipeline or PreprocessingPipeline.from_dict(pipeline_config)
        preprocessed = active_pipeline.apply(source)
    raise_if_preview_cancelled()

    rec = normalize_recognition_mode(getattr(contour_settings, "recognition_mode", "via"))
    if rec == RECOGNITION_MODE_DISABLED:
        saved_files: dict[str, str] = {}
        if output_directory:
            saved_files = save_bundle(
                output_directory=output_directory,
                image_path=image_path,
                polygons=list(passthrough_polygons) if passthrough_polygons else [],
                source_image=source,
                display_settings=display_settings or DisplaySettings(),
                save_options=save_options or SaveOptions(),
                metadata={
                    "contour_settings": contour_settings.to_dict(),
                    "pipeline": pipeline_config,
                },
            )
        polys = list(passthrough_polygons) if passthrough_polygons else []
        return BatchImageResult(
            image_path=image_path,
            source_image=source if include_images_in_result else None,
            preprocessed_image=preprocessed if include_images_in_result else None,
            pipeline_config=dict(pipeline_config),
            mask_image=ensure_uint8(preprocessed) if include_images_in_result else None,
            polygons=polys,
            debug_candidates=[],
            debug_gradient_maps={},
            saved_files=saved_files,
        )
    if str(getattr(contour_settings, "algorithm_backend", "legacy")).lower() == "sem":
        return _process_image_path_sem_backend(
            image_path=image_path,
            pipeline_config=pipeline_config,
            contour_settings=contour_settings,
            output_directory=output_directory,
            save_options=save_options,
            display_settings=display_settings,
            source=source,
            preprocessed=preprocessed,
            save_bundle=save_bundle,
            include_images_in_result=include_images_in_result,
        )

    metal_overlays: dict[str, list[PolygonData]] = {}
    metal_debug_extra: dict[str, np.ndarray] = {}
    if contour_settings.object_type == "via" or contour_settings.output_mode == "box":
        mask, debug_candidates = build_via_vectorization_mask(preprocessed, contour_settings)
        raise_if_preview_cancelled()
        polygons = extract_polygons(mask, contour_settings)
        raise_if_preview_cancelled()
    elif _use_structural_metal_recovery(contour_settings):
        raise_if_preview_cancelled()
        polygons, mask, metal_debug_extra, metal_overlays = _run_structural_metal_recovery(
            preprocessed, contour_settings
        )
        raise_if_preview_cancelled()
        debug_candidates = []
    else:
        mask = build_conductor_vectorization_mask(source, preprocessed, contour_settings)
        raise_if_preview_cancelled()
        debug_candidates = []
        polygons = extract_polygons(mask, contour_settings)
        if _should_run_sem_dual_branch(contour_settings):
            raise_if_preview_cancelled()
            via_settings = _build_via_settings_for_dual_branch(contour_settings)
            via_mask, via_debug_candidates = build_via_vectorization_mask(preprocessed, via_settings)
            via_polygons = extract_polygons(via_mask, via_settings)
            via_polygons = _filter_vias_by_conductor_linearity(via_polygons, mask)
            polygons = _merge_dual_branch_polygons(polygons, via_polygons)
            debug_candidates.extend(via_debug_candidates)
    if not contour_settings.debug_enabled:
        debug_candidates = []
    base_metal_maps: dict[str, np.ndarray] = (
        dict(metal_debug_extra) if metal_debug_extra and include_images_in_result else {}
    )
    debug_gradient_maps: dict[str, np.ndarray] = dict(base_metal_maps)
    if contour_settings.debug_gradient_map_enabled or (
        contour_settings.debug_enabled and contour_settings.debug_gradient_map_enabled
    ):
        raise_if_preview_cancelled()
        try:
            extra_maps = build_detection_debug_maps(source, preprocessed, contour_settings)
            debug_gradient_maps = {**base_metal_maps, **extra_maps}
        except Exception:  # pragma: no cover - defensive: debug never breaks processing
            debug_gradient_maps = dict(base_metal_maps)
    if metal_debug_extra and "mask" not in debug_gradient_maps:
        debug_gradient_maps["mask"] = ensure_binary_mask(mask)
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
        metal_overlay_polygons=metal_overlays,
        saved_files=saved_files,
    )


def _process_image_path_sem_backend(
    *,
    image_path: str,
    pipeline_config: dict[str, Any],
    contour_settings: ContourExtractionSettings,
    output_directory: str | None,
    save_options: SaveOptions | None,
    display_settings: DisplaySettings | None,
    source: Any,
    preprocessed: Any,
    save_bundle: Callable[..., dict[str, str]],
    include_images_in_result: bool,
) -> BatchImageResult:
    from ....vision.integration import (
        contour_output_to_polygons,
        output_kind_from_text,
        run_contour_filled_mask,
        run_via_detection,
        via_output_to_polygons,
    )

    output_kind = output_kind_from_text(contour_settings.output_mode)
    debug_candidates: list[ContourDebugCandidate] = []
    debug_gradient_maps: dict[str, np.ndarray] = {}
    vision_json: dict[str, Any]

    raise_if_preview_cancelled()
    if contour_settings.object_type == "via":
        via_output = run_via_detection(
            source,
            image_path=image_path,
            output_kind=output_kind,
            legacy_settings=contour_settings,
        )
        polygons = via_output_to_polygons(via_output)
        mask = _render_polygon_mask_from_polygons(source, polygons)
        vision_json = via_output.to_json_dict()
        if contour_settings.debug_enabled:
            debug_candidates = _debug_candidates_from_via_hits(via_output.hits)
            debug_candidates.extend(_debug_candidates_from_via_debug(via_output.debug))
        if contour_settings.debug_gradient_map_enabled:
            debug_gradient_maps = dict(getattr(via_output, "debug", {}) or {})
    else:
        raise_if_preview_cancelled()
        contour_output = run_contour_filled_mask(
            source,
            image_path=image_path,
            output_kind=output_kind,
            noise_level=str(getattr(contour_settings, "sem_noise_level", "medium") or "medium"),
            legacy_settings=contour_settings,
        )
        polygons = contour_output_to_polygons(contour_output)
        mask = contour_output.filled_mask
        vision_json = contour_output.to_json_dict()
        preprocessed = contour_output.debug.get("preprocessed", preprocessed)
        if contour_settings.debug_gradient_map_enabled:
            debug_gradient_maps = {
                "source_gray": _via_grayscale(source),
                "preprocessed": ensure_uint8(preprocessed),
                "mask": ensure_binary_mask(mask),
            }

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
                "vision_backend": vision_json,
            },
        )

    return BatchImageResult(
        image_path=image_path,
        source_image=source if include_images_in_result else None,
        preprocessed_image=preprocessed if include_images_in_result else None,
        pipeline_config=dict(pipeline_config),
        mask_image=mask if include_images_in_result else None,
        polygons=polygons,
        debug_candidates=debug_candidates if contour_settings.debug_enabled else [],
        debug_gradient_maps=debug_gradient_maps if include_images_in_result else {},
        saved_files=saved_files,
    )


def _render_polygon_mask_from_polygons(source: Any, polygons: list[PolygonData]) -> np.ndarray:
    shape = np.asarray(source).shape[:2]
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        if len(polygon.points) < 3:
            continue
        pts = np.array([[round(x), round(y)] for x, y in polygon.points], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return ensure_binary_mask(mask)


def _debug_candidates_from_via_hits(hits: list[Any]) -> list[ContourDebugCandidate]:
    debug: list[ContourDebugCandidate] = []
    for index, hit in enumerate(hits):
        x_coord, y_coord, width, height = hit.to_axis_aligned_box()
        debug.append(
            ContourDebugCandidate(
                contour_index=index,
                bbox=(x_coord, y_coord, width, height),
                area=float(width * height),
                perimeter=float(2 * (width + height)),
                roundness=float(getattr(hit, "annulus_coverage", 0.0) * 100.0),
                accepted=True,
                reason=f"accepted:{getattr(hit, 'strategy', 'sem_primary')}",
                source=str(getattr(hit, "strategy", "sem_primary")),
                score=float(getattr(hit, "score", 0.0)),
            )
        )
    return debug


def _debug_candidates_from_via_debug(debug_payload: Any) -> list[ContourDebugCandidate]:
    payload = dict(debug_payload or {}) if isinstance(debug_payload, dict) else {}
    out: list[ContourDebugCandidate] = []
    index = 10_000
    for group_name, accepted in (("below_threshold", False), ("rejected", False)):
        items = payload.get(group_name, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            bbox_raw = item.get("bbox", [0, 0, 0, 0])
            if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) < 4:
                bbox = (0, 0, 0, 0)
            else:
                bbox = tuple(int(round(float(v))) for v in bbox_raw[:4])
            reason = str(item.get("reject_reason") or group_name)
            out.append(
                ContourDebugCandidate(
                    contour_index=index,
                    bbox=bbox,  # type: ignore[arg-type]
                    area=float(max(0, bbox[2]) * max(0, bbox[3])),
                    perimeter=float(2 * (max(0, bbox[2]) + max(0, bbox[3]))),
                    roundness=float(item.get("compactness", 0.0) or 0.0) * 100.0,
                    accepted=accepted,
                    reason=f"{group_name}:{reason}",
                    source=str(item.get("polarity_hypothesis") or payload.get("strategy") or "via"),
                    score=float(item.get("score", 0.0) or 0.0),
                )
            )
            index += 1
    return out
