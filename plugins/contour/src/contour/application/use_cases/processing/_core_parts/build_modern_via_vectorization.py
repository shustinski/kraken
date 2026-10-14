"""_core: build modern via vectorization."""
from __future__ import annotations
import json
import cv2
import numpy as np
from contour.utils import ensure_binary_mask
from contour.application.processing import (
    VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG,
    VIA_SEARCH_MODE_HYBRID,
    VIA_SEARCH_MODE_TEMPLATE,
    ContourDebugCandidate,
    ContourExtractionSettings,
    normalize_via_search_mode,
)

from contour.application.use_cases.processing._core_parts.module_state import _VIA_DETECTION_CACHE

def _build_modern_via_vectorization_mask(
    gray: np.ndarray, settings: ContourExtractionSettings
) -> tuple[np.ndarray, list[ContourDebugCandidate]]:
    from contour.application.use_cases.processing._core_parts.build_via_vectorization_mask import _bounded_cache_set, _make_image_signature
    from .....vision.via.bright_tophat_dog import (
        BrightViaDetectorConfig,
        prepare_bright_via_candidates,
        score_bright_via_candidates,
    )
    from .....vision.via_detection.heuristic_detector import detect_vias_heuristic
    from .....vision.via_detection.result import DetectionResult, ViaDetection
    from .....vision.via_detection.settings_bridge import heuristic_config_from_settings, template_config_from_settings
    from .....vision.via_detection.template_detector import (
        detect_vias_template_raw,
        score_vias_template_raw,
    )

    mode = normalize_via_search_mode(settings.via_search_mode)
    image_sig = _make_image_signature(gray)
    if mode in (VIA_SEARCH_MODE_TEMPLATE, VIA_SEARCH_MODE_HYBRID):
        tcfg = template_config_from_settings(settings)
        if not tcfg.templates and mode == VIA_SEARCH_MODE_TEMPLATE:
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
        if tcfg.templates:
            heavy_key = (
                VIA_SEARCH_MODE_TEMPLATE,
                image_sig,
                json.dumps(
                    {
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
            template_result = score_vias_template_raw(raw_matches, shape, tcfg)
        else:
            template_result = DetectionResult(method=VIA_SEARCH_MODE_TEMPLATE, accepted=[])
        if mode == VIA_SEARCH_MODE_HYBRID:
            heuristic_result = detect_vias_heuristic(gray, heuristic_config_from_settings(settings))
            combined = sorted(
                [*template_result.accepted, *heuristic_result.accepted],
                key=lambda detection: float(detection.score),
                reverse=True,
            )
            accepted: list[ViaDetection] = []
            for detection in combined:
                distance = max(detection.bbox[2], detection.bbox[3]) + 2
                if any(
                    (detection.x - other.x) ** 2 + (detection.y - other.y) ** 2
                    <= max(distance, max(other.bbox[2], other.bbox[3]) + 2) ** 2
                    for other in accepted
                ):
                    continue
                accepted.append(detection)
            result = DetectionResult(
                method=VIA_SEARCH_MODE_HYBRID,
                accepted=accepted,
                rejected=[*template_result.rejected, *heuristic_result.rejected],
                below_threshold=[*template_result.below_threshold, *heuristic_result.below_threshold],
            )
        else:
            result = template_result
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
            ax = ay = max(1, round(r))
        cv2.ellipse(mask, center, (ax, ay), 0.0, 0.0, 360.0, 255, thickness=-1, lineType=cv2.LINE_8)
    debug_candidates = []
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
                template_index=getattr(det, "template_index", None),
                metrics=dict(getattr(det, "features", {}) or {}),
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
