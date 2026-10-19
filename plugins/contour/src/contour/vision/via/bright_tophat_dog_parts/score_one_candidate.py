"""bright_tophat_dog: score one candidate."""
from __future__ import annotations
from typing import TYPE_CHECKING
from math import pi
import cv2
import numpy as np

from contour.vision.via.bright_tophat_dog_parts.build_isolation_map import (
    _disk_mean_f32,
    _expanded_bbox,
    _isolation_score,
    _sample_distance,
)

from contour.vision.via.bright_tophat_dog_parts.composite_final import _composite_final_0_100

from contour.vision.via.bright_tophat_dog_parts.draw_overlay import _paint_score_disk

from contour.vision.via.bright_tophat_dog_parts.hard import _hard

from contour.vision.via.bright_tophat_dog_parts.local_maxima_points import (
    _brightness_gate_reason,
    _passes_brightness_gate,
)

from contour.vision.via.bright_tophat_dog_parts.refine_center_bright_core import (
    _contour_geometry_center,
    _fixed_size_bbox,
    _refine_center_on_bright_core,
)

from contour.vision.via.bright_tophat_dog_parts.via_matched_filter_kernel import (
    _bright_center_score_f32,
    _center_mean_score_f32,
    _edge_likeness_score_f32,
    _line_likeness_score_f32,
    _radial_symmetry_score_f32,
    mask_fraction,
)

def _score_one_candidate(
    *,
    raw: _RawStage1,
    gray: np.ndarray,
    gray_f32: np.ndarray,
    processed: np.ndarray,
    processed_f32: np.ndarray,
    radial_f32: np.ndarray,
    tophat: np.ndarray,
    tophat_f32: np.ndarray,
    dog: np.ndarray,
    dog_f32: np.ndarray,
    metal_mask: np.ndarray,
    edge_distance: np.ndarray,
    cfg: BrightViaDetectorConfig,
    metal_mode: str,
    nominal_diameter: float,
    nominal_radius: float,
    min_area: float,
    max_area: float,
    isolation: tuple[np.ndarray, np.ndarray] | None,
    debug: _CandidateDebugMaps,
) -> BrightViaDetection | None:
    from contour.vision.via.bright_tophat_dog_parts.bright_via_detection import BrightViaDetection
    x, y, w, h = raw.bbox
    if raw.contour is not None:
        center = _contour_geometry_center(raw)
    else:
        center = _refine_center_on_bright_core(gray_f32, raw.center, nominal_diameter)
    area = raw.area
    aspect = float(w) / float(max(1, h)) if h > 0 else 999.0
    if raw.contour is not None:
        perimeter = float(cv2.arcLength(raw.contour, True))
        circularity = float(4.0 * pi * area / (perimeter * perimeter + 1e-6))
    else:
        circularity = 1.0
    if area < min_area * 0.99 or area > max_area * 1.01:
        return _hard(
            center,
            (x, y, w, h),
            area,
            circularity,
            aspect,
            "size",
        )
    if aspect < float(cfg.min_aspect) * 0.999 or aspect > float(cfg.max_aspect) * 1.001:
        return _hard(center, (x, y, w, h), area, circularity, aspect, "aspect")

    eff_d = max(nominal_diameter, float(np.sqrt(max(area, 1.0) * 4.0 / pi)))
    eff_r = max(1.0, eff_d * 0.5)
    center_mean = _center_mean_score_f32(gray_f32, center, eff_d)
    annular = _bright_center_score_f32(processed_f32, center, eff_d)
    local_contrast = annular
    if annular < float(cfg.min_annular_contrast) * 0.999:
        return _hard(
            center,
            (x, y, w, h),
            area,
            circularity,
            aspect,
            f"low_annular (raw={annular:.1f} min={cfg.min_annular_contrast})",
        )
    max_annular = float(cfg.max_annular_contrast)
    if max_annular > 0.0 and annular > max_annular * 1.001:
        return _hard(
            center,
            (x, y, w, h),
            area,
            circularity,
            aspect,
            f"high_annular (raw={annular:.1f} max={max_annular})",
        )
    if not _passes_brightness_gate(center_mean, cfg):
        return _hard(
            center,
            (x, y, w, h),
            area,
            circularity,
            aspect,
            _brightness_gate_reason(center_mean, cfg),
        )

    radial = _radial_symmetry_score_f32(radial_f32, center[0], center[1], eff_r)
    edge = _edge_likeness_score_f32(processed_f32, center[0], center[1], max(nominal_radius, eff_r))
    line = _line_likeness_score_f32(processed_f32, center[0], center[1], max(nominal_radius, eff_r))
    d_edge = _sample_distance(edge_distance, center)

    if (
        cfg.hard_reject_on_asymmetry
        and float(cfg.max_radial_asymmetry) > 0
        and radial > float(cfg.max_radial_asymmetry)
    ):
        return _hard(center, (x, y, w, h), area, circularity, aspect, f"asymmetry>{cfg.max_radial_asymmetry}")
    if cfg.hard_reject_on_edge and float(cfg.max_edge_likeness) > 0 and edge > float(cfg.max_edge_likeness):
        return _hard(center, (x, y, w, h), area, circularity, aspect, f"edge>{cfg.max_edge_likeness}")
    if cfg.hard_reject_on_line and float(cfg.max_line_likeness) > 0 and line > float(cfg.max_line_likeness):
        return _hard(center, (x, y, w, h), area, circularity, aspect, f"line>{cfg.max_line_likeness}")

    iso_score = _isolation_score(
        isolation,
        round(center[0]),
        round(center[1]),
        round(max(nominal_radius, eff_r)),
    )
    if iso_score < float(cfg.min_isolation_score) - 1e-4:
        return _hard(
            center,
            (x, y, w, h),
            area,
            circularity,
            aspect,
            f"low_isolation (iso={iso_score:.2f})",
        )

    metal = 1.0
    if metal_mode != "disabled":
        metal = mask_fraction(metal_mask, _expanded_bbox(center, nominal_diameter * 1.8, processed.shape))
    if metal_mode == "strict" and metal < float(cfg.metal_fraction_min) - 1e-4:
        return _hard(
            center,
            (x, y, w, h),
            area,
            circularity,
            aspect,
            f"metal<{cfg.metal_fraction_min}",
        )

    if debug is not None:
        _paint_score_disk(debug.radial_symmetry, center, max(2.0, eff_r), radial, 60.0)
        _paint_score_disk(debug.edge_likeness, center, max(2.0, eff_r), edge, 80.0)
        _paint_score_disk(debug.line_likeness, center, max(2.0, eff_r), line, 100.0)

    top = _disk_mean_f32(tophat_f32, center, max(1.0, nominal_diameter * 0.35))
    dogm = _disk_mean_f32(dog_f32, center, max(1.0, nominal_diameter * 0.35))
    local_peak = 0.5 * (float(top) + float(dogm))
    fscore = _composite_final_0_100(
        center_mean=center_mean,
        local_contrast=local_contrast,
        local_peak=local_peak,
        area=area,
        min_area=min_area,
        max_area=max_area,
        circularity=circularity,
        min_circularity=float(cfg.min_circularity),
        radial_asymmetry=radial,
        edge_likeness=edge,
        line_likeness=line,
        max_asym=cfg.max_radial_asymmetry,
        max_edge=cfg.max_edge_likeness,
        max_line=cfg.max_line_likeness,
        metal_fraction=metal,
        metal_mode=metal_mode,
        distance_to_edge=d_edge,
        annular_contrast=annular,
        isolation_score=iso_score,
    )
    st = "accepted" if fscore >= float(cfg.min_final_score) else "soft_reject"
    if abs(float(cfg.diameter_max) - float(cfg.diameter_min)) < 1e-6:
        out_diameter = float(cfg.diameter_min)
    else:
        out_diameter = min(max(eff_d, float(cfg.diameter_min)), float(cfg.diameter_max))
    out_bbox = _fixed_size_bbox(center, out_diameter, processed.shape)
    return BrightViaDetection(
        center=(float(center[0]), float(center[1])),
        bbox=out_bbox,
        area=float(area),
        circularity=float(circularity),
        aspect=float(aspect if aspect < 1000 else 1.0),
        brightness_score=float(center_mean),
        local_peak_score=float(local_peak),
        tophat_response=float(top),
        dog_response=float(dogm),
        metal_fraction=float(metal),
        final_score=float(fscore),
        radial_asymmetry=float(radial),
        edge_likeness=float(edge),
        line_likeness=float(line),
        distance_to_edge=float(d_edge),
        annular_contrast=float(annular),
        isolation_score=float(iso_score),
        status=st,
        hard_reason="",
    )

if TYPE_CHECKING:
    from contour.vision.via.bright_tophat_dog_parts.bright_via_detection import BrightViaDetection, _CandidateDebugMaps, _RawStage1
    from contour.vision.via.bright_tophat_dog_parts.bright_via_detector_config import BrightViaDetectorConfig
