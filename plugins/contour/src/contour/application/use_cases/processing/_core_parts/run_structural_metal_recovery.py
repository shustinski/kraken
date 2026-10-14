"""_core: run structural metal recovery."""
from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from contour.domain import PolygonData
from contour.utils import ensure_binary_mask, ensure_uint8
from contour.application.processing import (
    ALGORITHM_BACKEND_LEGACY,
    RECOGNITION_MODE_CONDUCTORS,
    ContourExtractionSettings,
    normalize_algorithm_backend,
    normalize_recognition_mode,
)

from contour.application.use_cases.processing._core_parts.refine_conductor_mask_by import _via_grayscale

def _use_structural_metal_recovery(settings: ContourExtractionSettings) -> bool:
    if not bool(getattr(settings, "metal_structural_pipeline", False)):
        return False
    if normalize_algorithm_backend(getattr(settings, "algorithm_backend", "")) != ALGORITHM_BACKEND_LEGACY:
        return False
    if settings.object_type != "conductor" or str(getattr(settings, "output_mode", "polygon")) == "box":
        return False
    return normalize_recognition_mode(getattr(settings, "recognition_mode", "")) == RECOGNITION_MODE_CONDUCTORS

def _run_structural_metal_recovery(
    source: Any,
    preprocessed: Any,
    contour_settings: ContourExtractionSettings,
) -> tuple[list[PolygonData], np.ndarray, dict[str, np.ndarray], dict[str, list[PolygonData]]]:
    from .....vision.metal_recovery import detect_metalization, metal_recovery_config_from_settings

    gray = ensure_uint8(_via_grayscale(preprocessed))
    source_gray = ensure_uint8(_via_grayscale(source))
    cfg = metal_recovery_config_from_settings(contour_settings)
    mr = detect_metalization(gray, cfg, source_image=source_gray)
    mask = mr.debug_images.get("metal_filtered_mask")
    if mask is None:
        mask = mr.debug_images["metal_binary_mask"]
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = ensure_binary_mask(mask)

    debug_maps: dict[str, np.ndarray] = dict(mr.debug_images)
    debug_maps["metal_preprocessed"] = gray
    if bool(getattr(contour_settings, "metal_display_show_mask", False)):
        debug_maps["metal_mask"] = mr.debug_images["metal_binary_mask"]

    polygons = list(mr.accepted)

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
    return polygons, mask, debug_maps, overlays
