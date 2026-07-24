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


def _heuristic_polarity_from_brightness_settings(settings: ContourExtractionSettings) -> str:
    """Keep an explicit polarity authoritative; ranges only constrain candidates."""
    explicit = _norm_polarity(getattr(settings, "via_heuristic_polarity", "auto"))
    if explicit != str(ViaPolarity.AUTO):
        return explicit
    white = bool(getattr(settings, "via_white_range_enabled", True))
    black = bool(getattr(settings, "via_black_range_enabled", False))
    if white and not black:
        return str(ViaPolarity.BRIGHT)
    if black and not white:
        return str(ViaPolarity.DARK)
    if white and black:
        return str(ViaPolarity.AUTO)
    return str(ViaPolarity.BRIGHT)


def fixed_via_diameters_from_settings(settings: ContourExtractionSettings) -> list[int]:
    """Return the single configured output diameter, independent of search sizes."""

    explicit = int(getattr(settings, "via_output_diameter", 0) or 0)
    if explicit > 0:
        return [explicit]
    from ...application.processing import ALGORITHM_BACKEND_SEM, normalize_algorithm_backend

    if normalize_algorithm_backend(getattr(settings, "algorithm_backend", "")) != ALGORITHM_BACKEND_SEM:
        legacy_widths = [int(value) for value in (getattr(settings, "fixed_via_widths", None) or []) if int(value) > 0]
        legacy_heights = [
            int(value) for value in (getattr(settings, "fixed_via_heights", None) or []) if int(value) > 0
        ]
        migrated = sorted(
            {
                max(1, round((width + height) * 0.5))
                for width, height in zip(legacy_widths, legacy_heights, strict=False)
            }
        )
        if migrated:
            return migrated
    dmin = max(1, int(getattr(settings, "bright_via_diameter_min", 6) or 6))
    dmax = max(dmin, int(getattr(settings, "bright_via_diameter_max", dmin) or dmin))
    # The current UI exposes one fixed diameter and stores it as equal bounds.
    # Do not let the legacy comma-separated field override that selection.
    if dmin == dmax:
        return [dmin]
    text = str(getattr(settings, "via_fixed_diameters_text", "") or "").strip()
    return parse_diameter_list(text) or sorted({dmin, dmax})


def heuristic_config_from_settings(settings: ContourExtractionSettings) -> HeuristicViaDetectorConfig:
    from ...application.processing import (
        ALGORITHM_BACKEND_SEM,
        normalize_algorithm_backend,
    )

    dmin = max(1, int(getattr(settings, "bright_via_diameter_min", 6) or 6))
    dmax = max(dmin, int(getattr(settings, "bright_via_diameter_max", 8) or 8))
    migrated_legacy_settings = normalize_algorithm_backend(
        getattr(settings, "algorithm_backend", "")
    ) != ALGORITHM_BACKEND_SEM
    if migrated_legacy_settings:
        legacy_widths = [
            int(value)
            for value in (getattr(settings, "fixed_via_widths", None) or [])
            if int(value) > 0
        ]
        legacy_heights = [
            int(value)
            for value in (getattr(settings, "fixed_via_heights", None) or [])
            if int(value) > 0
        ]
        legacy_diameters = [
            max(1, round((width + height) * 0.5))
            for width, height in zip(legacy_widths, legacy_heights, strict=False)
        ]
        if legacy_diameters:
            dmin = min(legacy_diameters)
            dmax = max(legacy_diameters)

    return HeuristicViaDetectorConfig(
        diameter_mode="range",
        diameter_min=dmin,
        diameter_max=dmax,
        fixed_diameters=[],
        polarity=_heuristic_polarity_from_brightness_settings(settings),
        nms_distance=max(0, int(getattr(settings, "bright_via_nms_distance", 5) or 0)),
        min_final_score=float(getattr(settings, "bright_via_min_final_score", 38.0) or 0.0),
        min_distance_between_peaks=0,
        min_peak_grey=float(getattr(settings, "heuristic_min_abs_peak", 0.0) or 0.0),
        background_sigma=float(getattr(settings, "heuristic_background_sigma", 25.0) or 25.0),
        analysis_window_scale=float(getattr(settings, "heuristic_analysis_window_scale", 3.0) or 3.0),
        min_analyze_size=24,
        use_bilateral=bool(getattr(settings, "heuristic_use_bilateral", False)),
        min_center_brightness=float(
            getattr(settings, "heuristic_min_center_brightness", 0.0) or 0.0
        ),
        min_center_contrast=float(getattr(settings, "heuristic_min_center_contrast", 50.0) or 0.0),
        min_peak_prominence=float(getattr(settings, "heuristic_min_peak_prominence", 50.0) or 0.0),
        min_compactness=float(getattr(settings, "heuristic_min_compactness", 0.9) or 0.0),
        min_circularity=float(getattr(settings, "heuristic_min_circularity", 0.40)),
        max_elongation=float(getattr(settings, "heuristic_max_elongation", 2.5) or 2.5),
        line_penalty_scale=float(getattr(settings, "heuristic_line_penalty_scale", 3.0) or 3.0),
        border_penalty_scale=float(getattr(settings, "heuristic_border_penalty_scale", 1.0) or 1.0),
        local_binarize_percentile=float(getattr(settings, "heuristic_local_binarize_percentile", 88.0) or 88.0),
        size_tolerance_ratio=float(getattr(settings, "heuristic_size_tolerance_range", 0.36) or 0.36),
        size_tolerance_ratio_fixed=float(getattr(settings, "heuristic_size_tolerance_fixed", 0.26) or 0.26),
        max_center_drift_ratio=float(getattr(settings, "heuristic_max_center_drift_ratio", 0.72) or 0.72),
        max_line_coherence=float(getattr(settings, "heuristic_max_line_coherence", 0.82)),
        min_edge_sharpness=float(getattr(settings, "heuristic_min_edge_sharpness", 0.20)),
        contrast_score_min=float(getattr(settings, "heuristic_contrast_score_min", 3.0)),
        contrast_score_max=float(getattr(settings, "heuristic_contrast_score_max", 20.0)),
        prominence_score_min=float(getattr(settings, "heuristic_prominence_score_min", 2.0)),
        prominence_score_max=float(getattr(settings, "heuristic_prominence_score_max", 25.0)),
        edge_snr_score_min=float(getattr(settings, "heuristic_edge_snr_score_min", 0.70)),
        edge_snr_score_max=float(getattr(settings, "heuristic_edge_snr_score_max", 2.80)),
        edge_quality_floor=float(getattr(settings, "heuristic_edge_quality_floor", 0.55)),
        border_balance_scale=float(getattr(settings, "heuristic_border_balance_scale", 2.0)),
        seed_percentile=float(getattr(settings, "heuristic_seed_percentile", 90.0)),
        use_intensity_range_seeds=migrated_legacy_settings,
        w_contrast=float(getattr(settings, "heuristic_w_contrast", 25.0)),
        w_prominence=float(getattr(settings, "heuristic_w_prominence", 20.0)),
        w_size=float(getattr(settings, "heuristic_w_size", 20.0)),
        w_compact=float(getattr(settings, "heuristic_w_compact", 15.0)),
        w_round=float(getattr(settings, "heuristic_w_round", 10.0)),
        w_balance=float(getattr(settings, "heuristic_w_balance", 10.0)),
        w_line=float(getattr(settings, "heuristic_w_line", 20.0)),
        w_border=float(getattr(settings, "heuristic_w_border", 20.0)),
        bright_range_enabled=bool(getattr(settings, "via_white_range_enabled", True)),
        bright_range_min=float(getattr(settings, "via_white_range_min", 140) or 0),
        bright_range_max=float(getattr(settings, "via_white_range_max", 255) or 255),
        dark_range_enabled=bool(getattr(settings, "via_black_range_enabled", False)),
        dark_range_min=float(getattr(settings, "via_black_range_min", 0) or 0),
        dark_range_max=float(getattr(settings, "via_black_range_max", 30) or 255),
    )


def template_config_from_settings(settings: ContourExtractionSettings) -> TemplateViaDetectorConfig:
    raw = list(getattr(settings, "via_template_images", None) or [])
    raw_scores = list(getattr(settings, "via_template_min_scores", None) or [])
    raw_diameters = list(getattr(settings, "via_template_diameters", None) or [])
    templates: list[Any] = []
    min_correlations: list[float] = []
    output_diameters: list[int] = []
    suppression_distance = 0
    fallback_score = max(0.0, min(1.0, float(getattr(settings, "via_template_min_score", 0.35))))
    for source_index, im in enumerate(raw):
        try:
            import numpy as _np

            t = _np.array(im, dtype=_np.uint8)
            if t.size:
                templates.append(t)
                suppression_distance = max(suppression_distance, int(max(t.shape[:2])) + 2)
                score = raw_scores[source_index] if source_index < len(raw_scores) else fallback_score
                min_correlations.append(max(0.0, min(1.0, float(score))))
                fallback_diameter = max(1, round((int(t.shape[0]) + int(t.shape[1])) * 0.5))
                diameter = raw_diameters[source_index] if source_index < len(raw_diameters) else fallback_diameter
                output_diameters.append(max(1, int(diameter)))
        except Exception:
            continue
    return TemplateViaDetectorConfig(
        templates=templates,
        min_correlation=fallback_score,
        min_correlations=min_correlations,
        output_diameters=output_diameters,
        # This value is intentionally automatic and absent from the UI.
        nms_distance=suppression_distance,
        scale_min=float(getattr(settings, "via_template_scale_min", 0.9) or 0.9),
        scale_max=float(getattr(settings, "via_template_scale_max", 1.1) or 1.1),
        scale_step=float(getattr(settings, "via_template_scale_step", 0.1) or 0.1),
        use_ccoeff_normed=True,
    )
