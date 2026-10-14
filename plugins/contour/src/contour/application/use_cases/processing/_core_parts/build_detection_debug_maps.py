"""_core: build detection debug maps."""
from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from contour.edge_detection import (
    build_gradient_elevation,
    phase_congruency,
    ridge_response,
    scharr_magnitude,
    structured_edges,
)
from contour.utils import ensure_binary_mask
from contour.application.preview_cancellation import raise_if_preview_cancelled
from contour.application.processing import (
    VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG,
    VIA_SEARCH_MODE_HEURISTIC,
    VIA_SEARCH_MODE_TEMPLATE,
    ContourExtractionSettings,
    normalize_via_search_mode,
)

from contour.application.use_cases.processing._core_parts.apply_via_filters import _via_candidate_radii

from contour.application.use_cases.processing._core_parts.build_via_vectorization_mask import (
    _recognition_image,
    build_conductor_vectorization_mask,
    build_via_vectorization_mask,
)

from contour.application.use_cases.processing._core_parts.extract_response_peaks import _ring_template_response

from contour.application.use_cases.processing._core_parts.refine_conductor_mask_by import (
    _clahe_gray,
    _conductor_gradient_elevation,
    _expected_via_span,
    _multiscale_blob_response,
    _resolve_conductor_edge_method,
    _resolve_via_edge_method,
    _via_grayscale,
)

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
    Via debug maps include the common spot and ring responses used by the
    unified legacy/modern candidate view.
    """

    maps: dict[str, np.ndarray] = {}
    if source_image is None and preprocessed_image is None:
        return maps
    recognition_image = _recognition_image(source_image, preprocessed_image)
    if recognition_image is None:
        return maps
    raise_if_preview_cancelled()

    if source_image is not None:
        maps["source_gray"] = _via_grayscale(source_image)
    recognition_gray = _via_grayscale(recognition_image)

    if settings.object_type == "via" or settings.output_mode == "box":
        edge_method = _resolve_via_edge_method(settings)
    else:
        edge_method = _resolve_conductor_edge_method(settings)

    elevation = build_gradient_elevation(recognition_gray, edge_method)
    maps["gradient_elevation"] = elevation
    if include_color_maps and elevation.size:
        maps["gradient_color"] = cv2.applyColorMap(elevation, cv2.COLORMAP_TURBO)

    vmode = normalize_via_search_mode(settings.via_search_mode)
    if (settings.object_type == "via" or settings.output_mode == "box") and vmode in (
        VIA_SEARCH_MODE_HEURISTIC,
        VIA_SEARCH_MODE_TEMPLATE,
        VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG,
    ):
        from .....vision.via.bright_tophat_dog import BrightViaDetectorConfig, detect_bright_vias
        from .....vision.via_detection.heuristic_detector import detect_vias_heuristic
        from .....vision.via_detection.settings_bridge import (
            heuristic_config_from_settings,
            template_config_from_settings,
        )
        from .....vision.via_detection.template_detector import detect_vias_template

        try:
            if vmode == VIA_SEARCH_MODE_TEMPLATE:
                r = detect_vias_template(recognition_gray, template_config_from_settings(settings))
                dbg = dict(r.debug_images)
            elif vmode == VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG:
                r = detect_bright_vias(recognition_gray, BrightViaDetectorConfig.from_legacy_settings(settings))
                dbg = dict(r.debug_images)
            else:
                r = detect_vias_heuristic(recognition_gray, heuristic_config_from_settings(settings))
                dbg = dict(r.debug_images)
            for _guard in ("source_gray", "gradient_elevation", "gradient_color"):
                dbg.pop(_guard, None)
            maps.update(dbg)
        except Exception:  # pragma: no cover - defensive debug path
            pass

        expected_span = _expected_via_span(settings)
        enhanced = _clahe_gray(recognition_gray) if min(recognition_gray.shape[:2]) >= 8 else recognition_gray
        maps["spot_response"] = _multiscale_blob_response(
            enhanced,
            expected_span,
            bright=True,
            line_suppression=max(0.0, min(1.0, float(settings.via_spot_line_suppression))),
        )
        maps["ring_response"] = _ring_template_response(elevation, _via_candidate_radii(settings, expected_span))

    maps["scharr"] = scharr_magnitude(recognition_gray)
    try:
        maps["phase_congruency"] = phase_congruency(recognition_gray)
    except Exception:  # pragma: no cover - numerical fallback
        maps["phase_congruency"] = np.zeros_like(recognition_gray, dtype=np.uint8)
    maps["structured"] = structured_edges(recognition_gray)
    maps["ridge"] = ridge_response(recognition_gray)

    if settings.object_type == "via" or settings.output_mode == "box":
        mask, _candidates = build_via_vectorization_mask(recognition_image, settings)
        maps["mask"] = ensure_binary_mask(mask)
    else:
        mask = build_conductor_vectorization_mask(source_image, preprocessed_image, settings)
        maps["mask"] = ensure_binary_mask(mask)
        if include_color_maps:
            maps["conductor_gradient_elevation"] = _conductor_gradient_elevation(recognition_gray, settings)
    return maps
