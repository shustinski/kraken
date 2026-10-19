"""processing: dict 2."""
from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any

from contour.application.processing_parts.module_state import (
    ALGORITHM_BACKEND_LEGACY,
    RECOGNITION_MODE_CONDUCTORS,
    VIA_CHANNEL_MODE_GRAYSCALE,
    VIA_SEARCH_MODE_HEURISTIC,
    VIA_SIZE_MODE_FIXED,
)

from contour.application.processing_parts.pipeline_step_config import (
    _normalize_bright_via_mask_mode,
    _normalize_bright_via_metal_constraint_mode,
    _odd_positive,
    _parse_template_images,
    normalize_algorithm_backend,
    normalize_recognition_mode,
    normalize_via_channel_mode,
    normalize_via_search_mode,
    normalize_via_size_mode,
    parse_integer_value_list,
)


class Dict2Mixin:
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ContourExtractionSettings:
        from ...vision.metal_recovery.segmentation import (
            migrate_legacy_metal_settings,
            normalize_metal_adaptive_method,
            resolve_metal_segmentation_strategy,
        )
        from ...vision.metal_recovery.strategy_registry import MetalStrategyConfigs

        legacy_method_only = "metal_segmentation_method" in payload and "metal_segmentation_strategy" not in payload
        payload = migrate_legacy_metal_settings(dict(payload))
        if legacy_method_only:
            payload["metal_segmentation_strategy"] = "legacy_otsu"
        max_area = payload.get("max_area")
        max_perimeter = payload.get("max_perimeter")
        max_bbox_width = payload.get("max_bbox_width")
        max_bbox_height = payload.get("max_bbox_height")
        max_aspect_ratio = payload.get("max_aspect_ratio")
        max_via_width = payload.get("max_via_width")
        max_via_height = payload.get("max_via_height")
        max_hierarchy_depth = payload.get("max_hierarchy_depth")
        min_inner_hole_area = payload.get("min_inner_hole_area")
        max_hole_area_ratio = payload.get("max_hole_area_ratio")
        metal_max_trace_width = payload.get("metal_max_trace_width_px")
        metal_max_area = payload.get("metal_max_area")
        metal_max_perimeter = payload.get("metal_max_perimeter")
        metal_preprocess_denoise = str(payload.get("metal_preprocess_denoise", "low") or "low")
        if metal_preprocess_denoise not in {"low", "medium", "high"}:
            metal_preprocess_denoise = "low"
        white_range_enabled = payload.get("via_white_range_enabled", payload.get("via_white_threshold_enabled", True))
        white_range_min = payload.get("via_white_range_min", payload.get("via_white_threshold", 200))
        white_range_max = payload.get("via_white_range_max", 255)
        black_range_enabled = payload.get("via_black_range_enabled", payload.get("via_black_threshold_enabled", False))
        black_range_min = payload.get("via_black_range_min", 0)
        black_range_max = payload.get("via_black_range_max", payload.get("via_black_threshold", 30))
        if payload.get("via_threshold_range_enabled", False) and "via_white_range_min" not in payload:
            white_range_enabled = True
            white_range_min = payload.get("via_threshold_range_min", white_range_min)
            white_range_max = payload.get("via_threshold_range_max", white_range_max)
        fixed_via_widths = parse_integer_value_list(payload.get("fixed_via_widths"))
        fixed_via_heights = parse_integer_value_list(payload.get("fixed_via_heights"))
        legacy_diameters = [
            max(1, round((width + height) * 0.5))
            for width, height in zip(fixed_via_widths, fixed_via_heights, strict=False)
            if width > 0 and height > 0
        ]
        candidate_diameter_min = int(
            payload.get(
                "bright_via_diameter_min",
                min(legacy_diameters) if legacy_diameters else 6,
            )
        )
        candidate_diameter_max = int(
            payload.get(
                "bright_via_diameter_max",
                max(legacy_diameters) if legacy_diameters else 8,
            )
        )
        output_diameter = int(
            payload.get(
                "via_output_diameter",
                round(
                    sum(legacy_diameters) / len(legacy_diameters)
                    if legacy_diameters
                    else (candidate_diameter_min + candidate_diameter_max) * 0.5
                ),
            )
        )
        return cls(
            extraction_profile=str(payload.get("extraction_profile", "conductors")),
            algorithm_backend=normalize_algorithm_backend(payload.get("algorithm_backend", ALGORITHM_BACKEND_LEGACY)),
            sem_noise_level=str(payload.get("sem_noise_level", "medium") or "medium"),
            sem_polarity=str(payload.get("sem_polarity", "auto") or "auto"),
            sem_preserve_hierarchy=bool(payload.get("sem_preserve_hierarchy", True)),
            object_type=str(payload.get("object_type", "conductor")),
            output_mode=str(payload.get("output_mode", "polygon")),
            retrieval_mode=str(payload.get("retrieval_mode", "RETR_EXTERNAL")),
            approximation_mode=str(payload.get("approximation_mode", "CHAIN_APPROX_SIMPLE")),
            epsilon=float(payload.get("epsilon", 2.0)),
            epsilon_relative=bool(payload.get("epsilon_relative", False)),
            preserve_corners=bool(payload.get("preserve_corners", False)),
            min_polygon_angle=max(0.0, min(180.0, float(payload.get("min_polygon_angle", 0.0)))),
            min_area=float(payload.get("min_area", 0.0)),
            max_area=None if max_area in (None, "", 0, 0.0) else float(max_area),
            min_perimeter=float(payload.get("min_perimeter", 10.0)),
            min_points=max(3, int(payload.get("min_points", 3))),
            max_perimeter=None if max_perimeter in (None, "", 0, 0.0) else float(max_perimeter),
            min_bbox_width=max(0, int(payload.get("min_bbox_width", 0))),
            max_bbox_width=None if max_bbox_width in (None, "", 0, 0.0) else max(1, int(max_bbox_width)),
            min_bbox_height=max(0, int(payload.get("min_bbox_height", 0))),
            max_bbox_height=None if max_bbox_height in (None, "", 0, 0.0) else max(1, int(max_bbox_height)),
            min_aspect_ratio=max(0.0, float(payload.get("min_aspect_ratio", 0.0))),
            max_aspect_ratio=None if max_aspect_ratio in (None, "", 0, 0.0) else float(max_aspect_ratio),
            exclude_border_touching=bool(payload.get("exclude_border_touching", False)),
            min_solidity=max(0.0, float(payload.get("min_solidity", 0.0))),
            min_extent=max(0.0, float(payload.get("min_extent", 0.0))),
            min_polygon_width_px=max(0.0, float(payload.get("min_polygon_width_px", 0.0))),
            min_via_width=max(0, int(payload.get("min_via_width", 0))),
            max_via_width=None if max_via_width in (None, "", 0, 0.0) else max(1, int(max_via_width)),
            min_via_height=max(0, int(payload.get("min_via_height", 0))),
            max_via_height=None if max_via_height in (None, "", 0, 0.0) else max(1, int(max_via_height)),
            via_size_mode=normalize_via_size_mode(payload.get("via_size_mode", VIA_SIZE_MODE_FIXED)),
            via_search_mode=normalize_via_search_mode(payload.get("via_search_mode", VIA_SEARCH_MODE_HEURISTIC)),
            fixed_via_widths=fixed_via_widths,
            fixed_via_heights=fixed_via_heights,
            via_channel_mode=normalize_via_channel_mode(payload.get("via_channel_mode", VIA_CHANNEL_MODE_GRAYSCALE)),
            via_white_range_enabled=bool(white_range_enabled),
            via_white_range_min=max(0, min(255, int(white_range_min))),
            via_white_range_max=max(0, min(255, int(white_range_max))),
            via_black_range_enabled=bool(black_range_enabled),
            via_black_range_min=max(0, min(255, int(black_range_min))),
            via_black_range_max=max(0, min(255, int(black_range_max))),
            via_min_roundness=max(0.0, float(payload.get("via_min_roundness", 40.0))),
            via_min_score=max(0.0, min(1.0, float(payload.get("via_min_score", 0.35)))),
            via_min_contrast=max(0.0, min(255.0, float(payload.get("via_min_contrast", 14.0)))),
            via_min_edge_coverage=max(0.0, min(1.0, float(payload.get("via_min_edge_coverage", 0.45)))),
            via_template_min_score=max(0.0, min(1.0, float(payload.get("via_template_min_score", 0.35)))),
            via_template_images=_parse_template_images(payload.get("via_template_images", [])),
            via_template_min_scores=[
                max(0.0, min(1.0, float(value))) for value in (payload.get("via_template_min_scores", []) or [])
            ],
            via_template_diameters=[max(1, int(value)) for value in (payload.get("via_template_diameters", []) or [])],
            via_template_nms_distance=max(0, int(payload.get("via_template_nms_distance", 4))),
            via_template_scale_min=max(0.1, float(payload.get("via_template_scale_min", 0.9))),
            via_template_scale_max=max(0.1, float(payload.get("via_template_scale_max", 1.1))),
            via_template_scale_step=max(0.01, float(payload.get("via_template_scale_step", 0.1))),
            via_heuristic_polarity=str(payload.get("via_heuristic_polarity", "auto") or "auto"),
            via_fixed_diameters_text=str(payload.get("via_fixed_diameters_text", "6, 8, 10") or "6, 8, 10"),
            heuristic_background_sigma=max(0.1, float(payload.get("heuristic_background_sigma", 25.0))),
            heuristic_analysis_window_scale=max(1.0, float(payload.get("heuristic_analysis_window_scale", 3.0))),
            heuristic_min_center_brightness=max(
                0.0, min(255.0, float(payload.get("heuristic_min_center_brightness", 0.0)))
            ),
            heuristic_min_center_contrast=max(0.0, float(payload.get("heuristic_min_center_contrast", 50.0))),
            heuristic_min_peak_prominence=max(0.0, float(payload.get("heuristic_min_peak_prominence", 50.0))),
            heuristic_min_compactness=max(0.0, float(payload.get("heuristic_min_compactness", 0.9))),
            heuristic_min_circularity=max(
                0.0, min(1.0, float(payload.get("heuristic_min_circularity", 0.40)))
            ),
            heuristic_max_elongation=max(1.0, float(payload.get("heuristic_max_elongation", 2.5))),
            heuristic_line_penalty_scale=max(0.0, float(payload.get("heuristic_line_penalty_scale", 3.0))),
            heuristic_border_penalty_scale=max(0.0, float(payload.get("heuristic_border_penalty_scale", 1.0))),
            heuristic_local_binarize_percentile=max(
                1.0, min(99.0, float(payload.get("heuristic_local_binarize_percentile", 88.0)))
            ),
            heuristic_min_abs_peak=max(0.0, float(payload.get("heuristic_min_abs_peak", 0.0))),
            heuristic_use_bilateral=bool(payload.get("heuristic_use_bilateral", False)),
            heuristic_size_tolerance_range=max(
                0.05, min(0.95, float(payload.get("heuristic_size_tolerance_range", 0.36)))
            ),
            heuristic_size_tolerance_fixed=max(
                0.05, min(0.95, float(payload.get("heuristic_size_tolerance_fixed", 0.26)))
            ),
            heuristic_max_center_drift_ratio=max(
                0.1, min(1.5, float(payload.get("heuristic_max_center_drift_ratio", 0.72)))
            ),
            heuristic_max_line_coherence=max(
                0.0, min(1.0, float(payload.get("heuristic_max_line_coherence", 0.82)))
            ),
            heuristic_min_edge_sharpness=max(0.0, float(payload.get("heuristic_min_edge_sharpness", 0.20))),
            heuristic_contrast_score_min=max(0.0, float(payload.get("heuristic_contrast_score_min", 3.0))),
            heuristic_contrast_score_max=max(0.0, float(payload.get("heuristic_contrast_score_max", 20.0))),
            heuristic_prominence_score_min=max(
                0.0, float(payload.get("heuristic_prominence_score_min", 2.0))
            ),
            heuristic_prominence_score_max=max(
                0.0, float(payload.get("heuristic_prominence_score_max", 25.0))
            ),
            heuristic_edge_snr_score_min=max(0.0, float(payload.get("heuristic_edge_snr_score_min", 0.70))),
            heuristic_edge_snr_score_max=max(0.0, float(payload.get("heuristic_edge_snr_score_max", 2.80))),
            heuristic_edge_quality_floor=max(
                0.0, min(1.0, float(payload.get("heuristic_edge_quality_floor", 0.55)))
            ),
            heuristic_border_balance_scale=max(
                0.0, float(payload.get("heuristic_border_balance_scale", 2.0))
            ),
            heuristic_seed_percentile=max(
                0.0,
                min(
                    100.0,
                    float(
                        payload.get(
                            "heuristic_seed_percentile",
                            payload.get("heuristic_seed_percentile_medium", 90.0),
                        )
                    ),
                ),
            ),
            heuristic_w_contrast=max(0.0, float(payload.get("heuristic_w_contrast", 25.0))),
            heuristic_w_prominence=max(0.0, float(payload.get("heuristic_w_prominence", 20.0))),
            heuristic_w_size=max(0.0, float(payload.get("heuristic_w_size", 20.0))),
            heuristic_w_compact=max(0.0, float(payload.get("heuristic_w_compact", 15.0))),
            heuristic_w_round=max(0.0, float(payload.get("heuristic_w_round", 10.0))),
            heuristic_w_balance=max(0.0, float(payload.get("heuristic_w_balance", 10.0))),
            heuristic_w_line=max(0.0, float(payload.get("heuristic_w_line", 20.0))),
            heuristic_w_border=max(0.0, float(payload.get("heuristic_w_border", 20.0))),
            via_spot_line_suppression=max(0.0, min(1.0, float(payload.get("via_spot_line_suppression", 0.65)))),
            bright_via_diameter_min=max(1, candidate_diameter_min),
            bright_via_diameter_max=max(1, candidate_diameter_max),
            via_output_diameter=max(1, output_diameter),
            bright_via_clahe_clip_limit=max(0.01, float(payload.get("bright_via_clahe_clip_limit", 2.0))),
            bright_via_clahe_tile_grid_size=max(1, int(payload.get("bright_via_clahe_tile_grid_size", 8))),
            bright_via_median_blur_kernel=_odd_positive(payload.get("bright_via_median_blur_kernel", 3), minimum=1),
            bright_via_tophat_kernel_size=_odd_positive(payload.get("bright_via_tophat_kernel_size", 11), minimum=3),
            bright_via_dog_sigma_small=max(0.01, float(payload.get("bright_via_dog_sigma_small", 0.8))),
            bright_via_dog_sigma_large=max(0.02, float(payload.get("bright_via_dog_sigma_large", 2.0))),
            bright_via_threshold_percentile=max(
                90.0, min(99.9, float(payload.get("bright_via_threshold_percentile", 99.0)))
            ),
            bright_via_mask_combine_mode=_normalize_bright_via_mask_mode(
                payload.get("bright_via_mask_combine_mode", "OR")
            ),
            bright_via_min_area_factor=max(0.01, float(payload.get("bright_via_min_area_factor", 0.45))),
            bright_via_max_area_factor=max(0.01, float(payload.get("bright_via_max_area_factor", 1.8))),
            bright_via_min_circularity=max(0.0, float(payload.get("bright_via_min_circularity", 0.30))),
            bright_via_min_aspect=max(0.01, float(payload.get("bright_via_min_aspect", 0.45))),
            bright_via_max_aspect=max(0.01, float(payload.get("bright_via_max_aspect", 2.2))),
            bright_via_bright_center_min_score=max(
                0.0, float(payload.get("bright_via_bright_center_min_score", 140.0))
            ),
            bright_via_metal_constraint_mode=_normalize_bright_via_metal_constraint_mode(
                payload.get(
                    "bright_via_metal_constraint_mode",
                    "soft" if bool(payload.get("bright_via_use_metal_mask", True)) else "disabled",
                )
            ),
            bright_via_use_metal_mask=bool(payload.get("bright_via_use_metal_mask", True)),
            bright_via_metal_fraction_min=max(0.0, min(1.0, float(payload.get("bright_via_metal_fraction_min", 0.3)))),
            bright_via_max_radial_asymmetry=max(0.0, float(payload.get("bright_via_max_radial_asymmetry", 32.0))),
            bright_via_max_edge_likeness=max(0.0, float(payload.get("bright_via_max_edge_likeness", 35.0))),
            bright_via_max_line_likeness=max(0.0, float(payload.get("bright_via_max_line_likeness", 65.0))),
            bright_via_nms_distance=max(0, int(payload.get("bright_via_nms_distance", 5))),
            bright_via_min_final_score=max(0.0, min(100.0, float(payload.get("bright_via_min_final_score", 38.0)))),
            bright_via_show_rejected=bool(payload.get("bright_via_show_rejected", True)),
            bright_via_hard_reject_on_asymmetry=bool(payload.get("bright_via_hard_reject_on_asymmetry", True)),
            bright_via_hard_reject_on_edge=bool(payload.get("bright_via_hard_reject_on_edge", False)),
            bright_via_hard_reject_on_line=bool(payload.get("bright_via_hard_reject_on_line", False)),
            bright_via_min_isolation_score=float(payload.get("bright_via_min_isolation_score", 0.38)),
            bright_via_min_annular_contrast=float(payload.get("bright_via_min_annular_contrast", 6.0)),
            bright_via_max_annular_contrast=float(payload.get("bright_via_max_annular_contrast", 0.0)),
            bright_via_tiled_threshold_size=max(32, int(payload.get("bright_via_tiled_threshold_size", 256))),
            bright_via_scale_kernels_from_diameter=bool(payload.get("bright_via_scale_kernels_from_diameter", True)),
            debug_enabled=bool(payload.get("debug_enabled", True)),
            debug_gradient_map_enabled=bool(payload.get("debug_gradient_map_enabled", False)),
            min_hierarchy_depth=max(0, int(payload.get("min_hierarchy_depth", 0))),
            max_hierarchy_depth=None if max_hierarchy_depth in (None, "", 0, 0.0) else max(0, int(max_hierarchy_depth)),
            min_inner_hole_area=max(
                0.0,
                float(100.0 if min_inner_hole_area in (None, "") else min_inner_hole_area),
            ),
            max_hole_area_ratio=None
            if max_hole_area_ratio in (None, "", 0, 0.0)
            else max(0.0, float(max_hole_area_ratio)),
            conductor_gradient_enabled=bool(payload.get("conductor_gradient_enabled", False)),
            conductor_gradient_min_strength=max(
                0.0, min(255.0, float(payload.get("conductor_gradient_min_strength", 18.0)))
            ),
            conductor_gradient_band_radius=max(0, min(25, int(payload.get("conductor_gradient_band_radius", 3)))),
            edge_method=str(payload.get("edge_method", "sobel") or "sobel"),
            via_gradient_edge_method=str(payload.get("via_gradient_edge_method", "") or ""),
            conductor_gradient_edge_method=str(payload.get("conductor_gradient_edge_method", "") or ""),
            recognition_mode=normalize_recognition_mode(payload.get("recognition_mode", RECOGNITION_MODE_CONDUCTORS)),
            via_display_show_detected=bool(payload.get("via_display_show_detected", True)),
            via_display_show_candidates=bool(payload.get("via_display_show_candidates", True)),
            metal_structural_pipeline=bool(payload.get("metal_structural_pipeline", False)),
            metal_preset=str(payload.get("metal_preset", "standard") or "standard"),
            metal_noise_suppression=max(0, min(100, int(payload.get("metal_noise_suppression", 20)))),
            metal_min_contrast=max(
                1.0,
                min(
                    255.0,
                    float(
                        payload.get(
                            "metal_min_contrast",
                            max(1.0, float(payload.get("metal_contrast_bias", 50.0))),
                        )
                    ),
                ),
            ),
            metal_contrast_bias=max(-50.0, min(50.0, float(payload.get("metal_contrast_bias", 0.0)))),
            metal_min_hole_source_contrast=max(
                0.0,
                min(255.0, float(payload.get("metal_min_hole_source_contrast", 8.0))),
            ),
            metal_min_hole_source_contrast_fraction=max(
                0.0,
                min(1.0, float(payload.get("metal_min_hole_source_contrast_fraction", 0.35))),
            ),
            metal_segmentation_strategy=resolve_metal_segmentation_strategy(
                payload.get("metal_segmentation_strategy", "legacy_otsu"),
                use_wide_conductor_gradient=bool(payload.get("metal_use_wide_conductor_gradient", False)),
            ),
            metal_auto_contrast_step=max(
                0.0,
                min(255.0, float(payload.get("metal_auto_contrast_step", 10.0))),
            ),
            metal_auto_source_contrast_step=max(
                0.0,
                min(255.0, float(payload.get("metal_auto_source_contrast_step", 4.0))),
            ),
            metal_auto_directional_gap_bridge_px=max(
                0,
                min(64, int(payload.get("metal_auto_directional_gap_bridge_px", 3))),
            ),
            metal_auto_directional_gap_min_source_intensity=max(
                0.0,
                min(
                    255.0,
                    float(payload.get("metal_auto_directional_gap_min_source_intensity", 45.0)),
                ),
            ),
            metal_gap_bridge_px=max(
                0, int(payload.get("metal_gap_bridge_px", payload.get("metal_morph_close_radius", 1)) or 0)
            ),
            metal_speckle_removal_px=max(
                0, int(payload.get("metal_speckle_removal_px", payload.get("metal_morph_open_radius", 1)) or 0)
            ),
            metal_contour_smooth_px=max(0.0, float(payload.get("metal_contour_smooth_px", 0.0) or 0.0)),
            metal_contour_close_px=max(0, int(payload.get("metal_contour_close_px", 0) or 0)),
            metal_min_object_area=max(0.0, float(payload.get("metal_min_object_area", 30.0))),
            metal_min_trace_width_px=max(0.5, float(payload.get("metal_min_trace_width_px", 8.0) or 8.0)),
            metal_conductor_size_offset_px=max(
                -20.0, min(20.0, float(payload.get("metal_conductor_size_offset_px", 0.0) or 0.0))
            ),
            metal_max_trace_width_px=None
            if metal_max_trace_width in (None, "", 0, 0.0)
            else max(0.5, float(metal_max_trace_width)),
            metal_min_trace_length_px=max(1.0, float(payload.get("metal_min_trace_length_px", 8.0) or 8.0)),
            metal_allowed_angles=str(payload.get("metal_allowed_angles", "free") or "free"),
            metal_angle_tolerance_deg=max(0.5, float(payload.get("metal_angle_tolerance_deg", 7.0) or 7.0)),
            metal_min_straightness=max(0.05, min(1.0, float(payload.get("metal_min_straightness", 0.2) or 0.2))),
            metal_allow_t_junction=bool(payload.get("metal_allow_t_junction", True)),
            metal_border_handling=str(payload.get("metal_border_handling", "mark") or "mark"),
            metal_check_contour_validity=bool(payload.get("metal_check_contour_validity", True)),
            metal_hierarchy_mode=str(payload.get("metal_hierarchy_mode", "full") or "full"),
            metal_min_area=max(
                0.0,
                float(
                    payload.get(
                        "metal_min_area",
                        payload.get("metal_min_object_area", 60.0),
                    )
                ),
            ),
            metal_max_area=None if metal_max_area in (None, "", 0, 0.0) else float(metal_max_area),
            metal_min_perimeter=max(0.0, float(payload.get("metal_min_perimeter", 32.0) or 32.0)),
            metal_max_perimeter=None if metal_max_perimeter in (None, "", 0, 0.0) else float(metal_max_perimeter),
            metal_approximation_enabled=bool(payload.get("metal_approximation_enabled", True)),
            metal_morph_close_radius=max(1, int(payload.get("metal_morph_close_radius", 1) or 1)),
            metal_morph_open_radius=max(0, int(payload.get("metal_morph_open_radius", 1) or 0)),
            metal_display_show_conductors=bool(payload.get("metal_display_show_conductors", True)),
            metal_display_show_mask=bool(payload.get("metal_display_show_mask", False)),
            metal_display_show_contours=bool(payload.get("metal_display_show_contours", True)),
            metal_display_show_rejected=bool(payload.get("metal_display_show_rejected", True)),
            metal_display_show_suspicious=bool(payload.get("metal_display_show_suspicious", True)),
            metal_display_show_border_highlight=bool(payload.get("metal_display_show_border_highlight", False)),
            metal_debug_visual=str(payload.get("metal_debug_visual", "overlay") or "overlay"),
            metal_overlay_opacity=max(0.05, min(1.0, float(payload.get("metal_overlay_opacity", 0.45) or 0.45))),
            metal_use_wide_conductor_gradient=bool(payload.get("metal_use_wide_conductor_gradient", False)),
            metal_watershed_smoothing_sigma=max(
                0.1, min(8.0, float(payload.get("metal_watershed_smoothing_sigma", 1.2) or 1.2))
            ),
            metal_watershed_core_margin=max(
                0.0, min(40.0, float(payload.get("metal_watershed_core_margin", 23.0) or 0.0))
            ),
            metal_watershed_groove_margin=max(
                0.0, min(40.0, float(payload.get("metal_watershed_groove_margin", 19.0) or 0.0))
            ),
            metal_watershed_rim_probe_px=max(1, min(32, int(payload.get("metal_watershed_rim_probe_px", 2) or 1))),
            metal_watershed_seed_speckle_px=max(
                0, min(8, int(payload.get("metal_watershed_seed_speckle_px", 1) or 0))
            ),
            metal_watershed_valley_span_px=max(
                0, min(16, int(payload.get("metal_watershed_valley_span_px", 5) or 0))
            ),
            metal_watershed_valley_depth=max(
                0.0, min(160.0, float(payload.get("metal_watershed_valley_depth", 65.0) or 0.0))
            ),
            metal_random_walker_beta=max(
                1.0, min(400.0, float(payload.get("metal_random_walker_beta", 90.0) or 90.0))
            ),
            metal_random_walker_iterations=max(
                8, min(400, int(payload.get("metal_random_walker_iterations", 160) or 160))
            ),
            metal_graph_cut_iterations=max(
                1, min(16, int(payload.get("metal_graph_cut_iterations", 5) or 5))
            ),
            metal_reconstruction_erode_px=max(
                0, min(16, int(payload.get("metal_reconstruction_erode_px", 0) or 0))
            ),
            metal_boundary_relief=max(
                1.0, min(80.0, float(payload.get("metal_boundary_relief", 16.0) or 16.0))
            ),
            metal_boundary_background_sigma=max(
                2.0,
                min(60.0, float(payload.get("metal_boundary_background_sigma", 12.0) or 12.0)),
            ),
            metal_structural_variant=str(payload.get("metal_structural_variant", "s2") or "s2"),
            metal_strategy_parameters=MetalStrategyConfigs.from_mapping(
                payload.get("metal_strategy_parameters")
                if isinstance(payload.get("metal_strategy_parameters"), dict)
                else None
            ).to_dict(),
            metal_adaptive_block_size=max(
                0, min(255, int(payload.get("metal_adaptive_block_size", 0) or 0))
            ),
            metal_adaptive_c=max(
                -64.0, min(64.0, float(payload.get("metal_adaptive_c", 0.0) or 0.0))
            ),
            metal_adaptive_method=normalize_metal_adaptive_method(
                payload.get("metal_adaptive_method", "gaussian")
            ),
            metal_preprocess_subtract_background=bool(
                payload.get("metal_preprocess_subtract_background", True)
            ),
            metal_preprocess_background_sigma_fraction=max(
                0.005,
                min(
                    0.25,
                    float(payload.get("metal_preprocess_background_sigma_fraction", 0.05) or 0.05),
                ),
            ),
            metal_min_object_source_contrast=max(
                0.0,
                min(255.0, float(payload.get("metal_min_object_source_contrast", 12.0))),
            ),
            metal_min_object_rim_contrast=max(
                0.0,
                min(255.0, float(payload.get("metal_min_object_rim_contrast", 36.0))),
            ),
            metal_min_object_rim_area_fraction=max(
                0.000001,
                min(1.0, float(payload.get("metal_min_object_rim_area_fraction", 0.001))),
            ),
            metal_preprocess_clahe_clip=max(
                0.1,
                min(20.0, float(payload.get("metal_preprocess_clahe_clip", 2.0) or 2.0)),
            ),
            metal_preprocess_clahe_grid=max(
                2,
                min(64, int(payload.get("metal_preprocess_clahe_grid", 8) or 8)),
            ),
            metal_preprocess_denoise=metal_preprocess_denoise,
            metal_edge_close_cap_px=max(5, min(21, int(payload.get("metal_edge_close_cap_px", 9) or 9) | 1)),
            metal_edge_watershed_split=bool(payload.get("metal_edge_watershed_split", True)),
            metal_edge_watershed_dist_peak_frac=max(
                0.22,
                min(0.55, float(payload.get("metal_edge_watershed_dist_peak_frac", 0.38) or 0.38)),
            ),
            metal_edge_watershed_max_pixels=int(payload.get("metal_edge_watershed_max_pixels", 3_000_000) or 3_000_000),
        )

if TYPE_CHECKING:
    from contour.application.processing import ContourExtractionSettings
