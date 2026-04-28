"""Map :class:`ContourExtractionSettings` to detector configs (stable defaults / persistence)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import HeuristicViaDetectorConfig, TemplateViaDetectorConfig, ViaPolarity, parse_diameter_list

if TYPE_CHECKING:
    from ...application.processing import ContourExtractionSettings


def _norm_polarity(v: Any) -> str:
    t = str(v or "auto").strip().lower()
    if t in {"auto", "авто"}:
        return str(ViaPolarity.AUTO)
    if t in {"bright", "свет", "светлые", "light"}:
        return str(ViaPolarity.BRIGHT)
    if t in {"dark", "тём", "тёмные", "dark_fg"}:
        return str(ViaPolarity.DARK)
    if t in {"ring_light_ring", "светлое_кольцо", "ring_light", "light_ring"}:
        return str(ViaPolarity.RING_LIGHT_RING)
    if t in {"ring_dark_ring", "тёмное_кольцо", "ring_dark", "dark_ring"}:
        return str(ViaPolarity.RING_DARK_RING)
    return str(ViaPolarity.AUTO)


def heuristic_config_from_settings(settings: ContourExtractionSettings) -> HeuristicViaDetectorConfig:
    from ...application.processing import VIA_SIZE_MODE_FIXED, normalize_via_size_mode, normalize_via_search_sensitivity

    mode = normalize_via_size_mode(getattr(settings, "via_size_mode", "range") or "range")
    dmin = max(1, int(getattr(settings, "bright_via_diameter_min", 6) or 6))
    dmax = max(dmin, int(getattr(settings, "bright_via_diameter_max", 8) or 8))
    text = str(getattr(settings, "via_fixed_diameters_text", "") or "").strip() or f"{dmin}, {dmax}"
    fixed = parse_diameter_list(text) or [dmin, dmax]

    return HeuristicViaDetectorConfig(
        diameter_mode="fixed" if mode == VIA_SIZE_MODE_FIXED else "range",
        diameter_min=dmin,
        diameter_max=dmax,
        fixed_diameters=fixed,
        polarity=_norm_polarity(getattr(settings, "via_heuristic_polarity", "auto")),
        sensitivity=normalize_via_search_sensitivity(getattr(settings, "via_search_sensitivity", "medium")),
        nms_distance=max(0, int(getattr(settings, "bright_via_nms_distance", 5) or 0)),
        min_final_score=float(getattr(settings, "bright_via_min_final_score", 40.0) or 0.0),
        min_distance_between_peaks=0,
        min_peak_grey=float(getattr(settings, "heuristic_min_abs_peak", 0.0) or 0.0),
        background_sigma=float(getattr(settings, "heuristic_background_sigma", 25.0) or 25.0),
        analysis_window_scale=float(getattr(settings, "heuristic_analysis_window_scale", 3.0) or 3.0),
        min_analyze_size=24,
        use_bilateral=bool(getattr(settings, "heuristic_use_bilateral", False)),
        min_center_contrast=float(getattr(settings, "heuristic_min_center_contrast", 6.0) or 0.0),
        min_peak_prominence=float(getattr(settings, "heuristic_min_peak_prominence", 4.0) or 0.0),
        min_compactness=float(getattr(settings, "heuristic_min_compactness", 0.12) or 0.0),
        max_elongation=float(getattr(settings, "heuristic_max_elongation", 3.2) or 3.2),
        line_penalty_scale=float(getattr(settings, "heuristic_line_penalty_scale", 1.0) or 1.0),
        border_penalty_scale=float(getattr(settings, "heuristic_border_penalty_scale", 1.0) or 1.0),
        local_binarize_percentile=float(getattr(settings, "heuristic_local_binarize_percentile", 88.0) or 88.0),
        size_tolerance_ratio=float(getattr(settings, "heuristic_size_tolerance_range", 0.30) or 0.30),
        size_tolerance_ratio_fixed=float(getattr(settings, "heuristic_size_tolerance_fixed", 0.18) or 0.18),
        max_center_drift_ratio=float(getattr(settings, "heuristic_max_center_drift_ratio", 0.55) or 0.55),
    )


def template_config_from_settings(settings: ContourExtractionSettings) -> TemplateViaDetectorConfig:
    raw = list(getattr(settings, "via_template_images", None) or [])
    templates: list[Any] = []
    for im in raw:
        try:
            import numpy as _np

            t = _np.array(im, dtype=_np.uint8)
            if t.size:
                templates.append(t)
        except Exception:
            continue
    return TemplateViaDetectorConfig(
        templates=templates,
        min_correlation=float(getattr(settings, "via_template_min_score", 0.35) or 0.35),
        nms_distance=max(0, int(getattr(settings, "via_template_nms_distance", 4) or 0)),
        scale_min=float(getattr(settings, "via_template_scale_min", 0.9) or 0.9),
        scale_max=float(getattr(settings, "via_template_scale_max", 1.1) or 1.1),
        scale_step=float(getattr(settings, "via_template_scale_step", 0.1) or 0.1),
        use_ccoeff_normed=True,
    )
