"""extraction_settings_mixin: current contour settings."""
from __future__ import annotations
from contour.widget_parts._imports import (
    VIA_SIZE_MODE_RANGE,
    ContourExtractionSettings,
    _normalize_bright_via_metal_constraint_mode,
    normalize_algorithm_backend,
    normalize_recognition_mode,
    normalize_via_search_mode,
)


class CurrentContourSettingsMixin:
    def _current_contour_settings(self) -> ContourExtractionSettings:
        if hasattr(self, "_ensure_via_template_metadata"):
            self._ensure_via_template_metadata()
        max_area = self.max_area_spin.value()
        max_perimeter = self.max_perimeter_spin.value()
        max_bbox_width = self.max_bbox_width_spin.value()
        max_bbox_height = self.max_bbox_height_spin.value()
        max_aspect_ratio = self.max_aspect_ratio_spin.value()
        max_via_width = self.max_via_width_spin.value()
        max_via_height = self.max_via_height_spin.value()
        raw_rec = (
            str(self.recognition_mode_combo.currentData() or "conductors")
            if hasattr(self, "recognition_mode_combo")
            else "conductors"
        )
        rec_mode = normalize_recognition_mode(raw_rec)
        via_size_mode = VIA_SIZE_MODE_RANGE
        via_search_mode_effective = normalize_via_search_mode(self.via_search_mode_combo.currentData())
        if rec_mode == "via":
            extraction_profile = "vias"
            object_type = "via"
            output_mode = "box"
            algorithm_backend = normalize_algorithm_backend("sem")
            via_search_mode_effective = normalize_via_search_mode(self.via_search_mode_combo.currentData())
        else:
            extraction_profile = "conductors"
            object_type = "conductor"
            output_mode = "polygon"
            algorithm_backend = normalize_algorithm_backend("legacy")
            via_search_mode_effective = normalize_via_search_mode(self.via_search_mode_combo.currentData())
        fixed_via_pairs = self._fixed_via_pairs()
        fixed_via_widths = [width for width, _height in fixed_via_pairs]
        fixed_via_heights = [height for _width, height in fixed_via_pairs]
        max_hierarchy_depth = self.max_hierarchy_depth_spin.value()
        max_hole_area_ratio = self.max_hole_area_ratio_spin.value()
        bright_via_diameter_min, bright_via_diameter_max = self._bright_via_diameter_bounds_from_widgets()
        return ContourExtractionSettings(
            algorithm_backend=algorithm_backend,
            sem_noise_level="medium",
            extraction_profile=extraction_profile,
            object_type=object_type,
            output_mode=output_mode,
            retrieval_mode=str(self.retrieval_mode_combo.currentData() or self.retrieval_mode_combo.currentText()),
            approximation_mode=str(
                self.approximation_mode_combo.currentData() or self.approximation_mode_combo.currentText()
            ),
            epsilon=self.metal_epsilon_spin.value()
            if hasattr(self, "metal_epsilon_spin") and rec_mode != "via"
            else self.epsilon_spin.value(),
            epsilon_relative=self.epsilon_relative_checkbox.isChecked(),
            min_polygon_angle=self.metal_min_angle_spin.value()
            if hasattr(self, "metal_min_angle_spin") and rec_mode != "via"
            else self.min_polygon_angle_spin.value(),
            min_area=self.min_area_spin.value(),
            max_area=None if max_area <= 0 else max_area,
            min_perimeter=self.min_perimeter_spin.value(),
            min_points=self.metal_min_points_spin.value()
            if hasattr(self, "metal_min_points_spin") and rec_mode != "via"
            else self.min_points_spin.value(),
            max_perimeter=None if max_perimeter <= 0 else max_perimeter,
            min_bbox_width=self.min_bbox_width_spin.value(),
            max_bbox_width=None if max_bbox_width <= 0 else max_bbox_width,
            min_bbox_height=self.min_bbox_height_spin.value(),
            max_bbox_height=None if max_bbox_height <= 0 else max_bbox_height,
            min_aspect_ratio=self.min_aspect_ratio_spin.value(),
            max_aspect_ratio=None if max_aspect_ratio <= 0 else max_aspect_ratio,
            exclude_border_touching=self.exclude_border_touching_checkbox.isChecked(),
            min_solidity=self.min_solidity_spin.value(),
            min_extent=self.min_extent_spin.value(),
            min_polygon_width_px=self.min_polygon_width_spin.value(),
            conductor_gradient_enabled=False,
            conductor_gradient_min_strength=self.conductor_gradient_min_strength_spin.value()
            if hasattr(self, "conductor_gradient_min_strength_spin")
            else 18.0,
            conductor_gradient_band_radius=self.conductor_gradient_band_radius_spin.value()
            if hasattr(self, "conductor_gradient_band_radius_spin")
            else 3,
            via_size_mode=via_size_mode,
            via_search_mode=via_search_mode_effective,
            via_white_range_enabled=self.via_white_range_checkbox.isChecked(),
            via_white_range_min=self.via_white_range_min_spin.value(),
            via_white_range_max=self.via_white_range_max_spin.value(),
            via_black_range_enabled=self.via_black_range_checkbox.isChecked(),
            via_black_range_min=self.via_black_range_min_spin.value(),
            via_black_range_max=self.via_black_range_max_spin.value(),
            via_min_score=self.via_min_score_spin.value(),
            via_min_contrast=self.via_min_contrast_spin.value(),
            via_min_edge_coverage=self.via_min_edge_coverage_spin.value(),
            via_template_min_score=self.via_template_min_score_spin.value(),
            via_spot_line_suppression=self.via_spot_line_suppression_spin.value(),
            bright_via_diameter_min=bright_via_diameter_min,
            bright_via_diameter_max=bright_via_diameter_max,
            via_output_diameter=int(self.via_output_diameter_spin.value()),
            bright_via_clahe_clip_limit=self.bright_via_clahe_clip_spin.value(),
            bright_via_clahe_tile_grid_size=self.bright_via_clahe_tile_spin.value(),
            bright_via_median_blur_kernel=self.bright_via_median_kernel_spin.value(),
            bright_via_tophat_kernel_size=self.bright_via_tophat_kernel_spin.value(),
            bright_via_dog_sigma_small=self.bright_via_dog_small_spin.value(),
            bright_via_dog_sigma_large=self.bright_via_dog_large_spin.value(),
            bright_via_threshold_percentile=self.bright_via_threshold_percentile_spin.value(),
            bright_via_mask_combine_mode=str(self.bright_via_mask_combine_combo.currentData() or "OR"),
            bright_via_min_area_factor=self.bright_via_min_area_factor_spin.value(),
            bright_via_max_area_factor=self.bright_via_max_area_factor_spin.value(),
            bright_via_min_circularity=self.bright_via_min_circularity_spin.value(),
            bright_via_min_aspect=self.bright_via_min_aspect_spin.value(),
            bright_via_max_aspect=self.bright_via_max_aspect_spin.value(),
            bright_via_bright_center_min_score=(
                float(self.via_white_range_min_spin.value())
                if self.via_white_range_checkbox.isChecked()
                else float(self.bright_via_bright_center_score_spin.value())
            ),
            bright_via_metal_constraint_mode=_normalize_bright_via_metal_constraint_mode(
                self.bright_via_metal_constraint_combo.currentData()
            ),
            bright_via_use_metal_mask=str(self.bright_via_metal_constraint_combo.currentData()) != "disabled",
            bright_via_metal_fraction_min=self.bright_via_metal_fraction_spin.value(),
            bright_via_max_radial_asymmetry=self.bright_via_max_radial_asymmetry_spin.value(),
            bright_via_max_edge_likeness=self.bright_via_max_edge_likeness_spin.value(),
            bright_via_max_line_likeness=self.bright_via_max_line_likeness_spin.value(),
            bright_via_nms_distance=self.bright_via_nms_distance_spin.value(),
            bright_via_min_final_score=self.bright_via_min_final_score_spin.value(),
            bright_via_show_rejected=self.bright_via_show_rejected_checkbox.isChecked(),
            bright_via_hard_reject_on_asymmetry=self.bright_via_hard_asym_checkbox.isChecked(),
            bright_via_hard_reject_on_edge=self.bright_via_hard_edge_checkbox.isChecked(),
            bright_via_hard_reject_on_line=self.bright_via_hard_line_checkbox.isChecked(),
            bright_via_min_isolation_score=self.bright_via_min_isolation_spin.value()
            if hasattr(self, "bright_via_min_isolation_spin")
            else 0.38,
            via_template_images=[template.copy() for template in self._via_template_images],
            via_template_min_scores=list(self._via_template_min_scores),
            via_template_diameters=list(self._via_template_diameters),
            via_template_nms_distance=max(
                (int(max(template.shape[:2])) + 2 for template in self._via_template_images),
                default=0,
            ),
            via_template_scale_min=self.via_template_scale_min_spin.value()
            if hasattr(self, "via_template_scale_min_spin")
            else 0.9,
            via_template_scale_max=self.via_template_scale_max_spin.value()
            if hasattr(self, "via_template_scale_max_spin")
            else 1.1,
            via_template_scale_step=self.via_template_scale_step_spin.value()
            if hasattr(self, "via_template_scale_step_spin")
            else 0.1,
            via_heuristic_polarity=str(self.via_heuristic_polarity_combo.currentData() or "auto")
            if hasattr(self, "via_heuristic_polarity_combo")
            else "auto",
            via_fixed_diameters_text=str(self.via_fixed_diameters_edit.text() or "6, 8, 10")
            if hasattr(self, "via_fixed_diameters_edit")
            else "6, 8, 10",
            heuristic_background_sigma=self.heuristic_background_sigma_spin.value()
            if hasattr(self, "heuristic_background_sigma_spin")
            else 25.0,
            heuristic_analysis_window_scale=self.heuristic_analysis_window_scale_spin.value()
            if hasattr(self, "heuristic_analysis_window_scale_spin")
            else 3.0,
            heuristic_min_center_brightness=self.heuristic_min_center_brightness_spin.value()
            if hasattr(self, "heuristic_min_center_brightness_spin")
            else 0.0,
            heuristic_min_center_contrast=self.heuristic_min_center_contrast_spin.value()
            if hasattr(self, "heuristic_min_center_contrast_spin")
            else 6.0,
            heuristic_min_peak_prominence=self.heuristic_min_peak_prominence_spin.value()
            if hasattr(self, "heuristic_min_peak_prominence_spin")
            else 50.0,
            heuristic_min_compactness=self.heuristic_min_compactness_spin.value()
            if hasattr(self, "heuristic_min_compactness_spin")
            else 0.12,
            heuristic_min_circularity=self.heuristic_min_circularity_spin.value()
            if hasattr(self, "heuristic_min_circularity_spin")
            else 0.0,
            heuristic_max_elongation=self.heuristic_max_elongation_spin.value()
            if hasattr(self, "heuristic_max_elongation_spin")
            else 3.2,
            heuristic_line_penalty_scale=self.heuristic_line_penalty_spin.value()
            if hasattr(self, "heuristic_line_penalty_spin")
            else 1.0,
            heuristic_border_penalty_scale=self.heuristic_border_penalty_spin.value()
            if hasattr(self, "heuristic_border_penalty_spin")
            else 1.0,
            heuristic_local_binarize_percentile=self.heuristic_local_binarize_percentile_spin.value()
            if hasattr(self, "heuristic_local_binarize_percentile_spin")
            else 88.0,
            heuristic_min_abs_peak=self.heuristic_min_abs_peak_spin.value()
            if hasattr(self, "heuristic_min_abs_peak_spin")
            else 0.0,
            heuristic_use_bilateral=self.heuristic_use_bilateral_checkbox.isChecked()
            if hasattr(self, "heuristic_use_bilateral_checkbox")
            else False,
            heuristic_size_tolerance_range=self.heuristic_size_tolerance_range_spin.value(),
            heuristic_size_tolerance_fixed=self.heuristic_size_tolerance_fixed_spin.value(),
            heuristic_max_center_drift_ratio=self.heuristic_max_center_drift_ratio_spin.value(),
            heuristic_max_line_coherence=self.heuristic_max_line_coherence_spin.value(),
            heuristic_min_edge_sharpness=self.heuristic_min_edge_sharpness_spin.value(),
            heuristic_contrast_score_min=self.heuristic_contrast_score_min_spin.value(),
            heuristic_contrast_score_max=self.heuristic_contrast_score_max_spin.value(),
            heuristic_prominence_score_min=self.heuristic_prominence_score_min_spin.value(),
            heuristic_prominence_score_max=self.heuristic_prominence_score_max_spin.value(),
            heuristic_edge_snr_score_min=self.heuristic_edge_snr_score_min_spin.value(),
            heuristic_edge_snr_score_max=self.heuristic_edge_snr_score_max_spin.value(),
            heuristic_edge_quality_floor=self.heuristic_edge_quality_floor_spin.value(),
            heuristic_border_balance_scale=self.heuristic_border_balance_scale_spin.value(),
            heuristic_seed_percentile=self.heuristic_seed_percentile_spin.value(),
            heuristic_w_contrast=self.heuristic_w_contrast_spin.value(),
            heuristic_w_prominence=self.heuristic_w_prominence_spin.value(),
            heuristic_w_size=self.heuristic_w_size_spin.value(),
            heuristic_w_compact=self.heuristic_w_compact_spin.value(),
            heuristic_w_round=self.heuristic_w_round_spin.value(),
            heuristic_w_balance=self.heuristic_w_balance_spin.value(),
            heuristic_w_line=self.heuristic_w_line_spin.value(),
            heuristic_w_border=self.heuristic_w_border_spin.value(),
            debug_enabled=self.debug_candidates_checkbox.isChecked(),
            # Diagnostic layers are built lazily when selected in the
            # display combo; there is no separate debug-image mode.
            debug_gradient_map_enabled=False,
            recognition_mode=raw_rec,
            via_display_show_detected=(
                self.via_show_detected_checkbox.isChecked() if hasattr(self, "via_show_detected_checkbox") else True
            ),
            via_display_show_candidates=self.debug_candidates_checkbox.isChecked(),
            metal_structural_pipeline=(raw_rec == "conductors"),
            metal_preset=self._current_metal_preset_key() if hasattr(self, "metal_preset_combo") else "standard",
            metal_min_contrast=float(self.metal_min_contrast_slider.value())
            if hasattr(self, "metal_min_contrast_slider")
            else 0.0,
            metal_min_object_source_contrast=float(
                self.metal_min_object_source_contrast_spin.value()
            )
            if hasattr(self, "metal_min_object_source_contrast_spin")
            else 12.0,
            metal_min_object_rim_contrast=float(self.metal_min_object_rim_contrast_spin.value())
            if hasattr(self, "metal_min_object_rim_contrast_spin")
            else 36.0,
            metal_min_object_rim_area_fraction=float(
                self.metal_min_object_rim_area_fraction_spin.value()
            )
            if hasattr(self, "metal_min_object_rim_area_fraction_spin")
            else 0.001,
            metal_min_hole_source_contrast=float(self.metal_min_hole_source_contrast_spin.value())
            if hasattr(self, "metal_min_hole_source_contrast_spin")
            else 8.0,
            metal_min_hole_source_contrast_fraction=float(
                self.metal_min_hole_source_contrast_fraction_spin.value()
            )
            if hasattr(self, "metal_min_hole_source_contrast_fraction_spin")
            else 0.35,
            metal_segmentation_strategy=self._metal_strategy_from_combo(),
            metal_strategy_parameters=self._metal_strategy_parameters_from_widgets(),
            metal_auto_contrast_step=float(self.metal_auto_contrast_step_spin.value())
            if hasattr(self, "metal_auto_contrast_step_spin")
            else 10.0,
            metal_auto_source_contrast_step=float(
                self.metal_auto_source_contrast_step_spin.value()
            )
            if hasattr(self, "metal_auto_source_contrast_step_spin")
            else 4.0,
            metal_auto_directional_gap_bridge_px=int(
                self.metal_auto_directional_gap_bridge_spin.value()
            )
            if hasattr(self, "metal_auto_directional_gap_bridge_spin")
            else 3,
            metal_auto_directional_gap_min_source_intensity=float(
                self.metal_auto_directional_gap_min_source_spin.value()
            )
            if hasattr(self, "metal_auto_directional_gap_min_source_spin")
            else 45.0,
            metal_gap_bridge_px=int(self.metal_gap_bridge_spin.value())
            if hasattr(self, "metal_gap_bridge_spin")
            else 1,
            metal_speckle_removal_px=int(self.metal_speckle_removal_spin.value())
            if hasattr(self, "metal_speckle_removal_spin")
            else 1,
            metal_min_object_area=self.metal_min_area_spin.value() if hasattr(self, "metal_min_area_spin") else 60.0,
            metal_min_trace_width_px=float(self.metal_min_width_spin.value())
            if hasattr(self, "metal_min_width_spin")
            else 8.0,
            metal_conductor_size_offset_px=float(self.metal_conductor_size_offset_slider.value())
            if hasattr(self, "metal_conductor_size_offset_slider")
            else 0.0,
            metal_max_trace_width_px=None
            if not hasattr(self, "metal_max_width_spin") or self.metal_max_width_spin.value() <= 0
            else float(self.metal_max_width_spin.value()),
            metal_min_trace_length_px=float(self.metal_min_length_spin.value())
            if hasattr(self, "metal_min_length_spin")
            else 8.0,
            metal_border_handling=str(self.metal_border_handling_combo.currentData() or "mark")
            if hasattr(self, "metal_border_handling_combo")
            else "mark",
            metal_hierarchy_mode=str(self.metal_hierarchy_combo.currentData() or "full")
            if hasattr(self, "metal_hierarchy_combo")
            else "full",
            metal_min_area=self.metal_min_area_spin.value() if hasattr(self, "metal_min_area_spin") else 60.0,
            metal_max_area=None
            if not hasattr(self, "metal_max_area_spin") or self.metal_max_area_spin.value() <= 0
            else float(self.metal_max_area_spin.value()),
            metal_min_perimeter=float(self.metal_min_perimeter_spin.value())
            if hasattr(self, "metal_min_perimeter_spin")
            else 32.0,
            metal_max_perimeter=None
            if not hasattr(self, "metal_max_perimeter_spin") or self.metal_max_perimeter_spin.value() <= 0
            else float(self.metal_max_perimeter_spin.value()),
            metal_approximation_enabled=self.metal_approximation_checkbox.isChecked()
            if hasattr(self, "metal_approximation_checkbox")
            else True,
            metal_use_wide_conductor_gradient=self._metal_strategy_from_combo() == "gradient_watershed",
            metal_watershed_smoothing_sigma=float(self.metal_ws_smoothing_spin.value())
            if hasattr(self, "metal_ws_smoothing_spin")
            else 1.2,
            metal_watershed_core_margin=float(self.metal_ws_core_margin_spin.value())
            if hasattr(self, "metal_ws_core_margin_spin")
            else 23.0,
            metal_watershed_groove_margin=float(self.metal_ws_groove_margin_spin.value())
            if hasattr(self, "metal_ws_groove_margin_spin")
            else 19.0,
            metal_watershed_rim_probe_px=int(self.metal_ws_rim_probe_spin.value())
            if hasattr(self, "metal_ws_rim_probe_spin")
            else 2,
            metal_watershed_seed_speckle_px=int(self.metal_ws_seed_speckle_spin.value())
            if hasattr(self, "metal_ws_seed_speckle_spin")
            else 1,
            metal_watershed_valley_span_px=int(self.metal_ws_valley_span_spin.value())
            if hasattr(self, "metal_ws_valley_span_spin")
            else 5,
            metal_watershed_valley_depth=float(self.metal_ws_valley_depth_spin.value())
            if hasattr(self, "metal_ws_valley_depth_spin")
            else 65.0,
            metal_random_walker_beta=float(self.metal_rw_beta_spin.value())
            if hasattr(self, "metal_rw_beta_spin")
            else 90.0,
            metal_random_walker_iterations=int(self.metal_rw_iterations_spin.value())
            if hasattr(self, "metal_rw_iterations_spin")
            else 160,
            metal_graph_cut_iterations=int(self.metal_gc_iterations_spin.value())
            if hasattr(self, "metal_gc_iterations_spin")
            else 5,
            metal_reconstruction_erode_px=int(self.metal_recon_erode_spin.value())
            if hasattr(self, "metal_recon_erode_spin")
            else 0,
            metal_boundary_relief=float(self.metal_boundary_relief_spin.value())
            if hasattr(self, "metal_boundary_relief_spin")
            else 16.0,
            metal_boundary_background_sigma=float(self.metal_boundary_background_spin.value())
            if hasattr(self, "metal_boundary_background_spin")
            else 12.0,
            metal_structural_variant="s2",
            metal_adaptive_block_size=int(self.metal_adaptive_block_spin.value())
            if hasattr(self, "metal_adaptive_block_spin")
            else 0,
            metal_adaptive_c=float(self.metal_adaptive_c_spin.value())
            if hasattr(self, "metal_adaptive_c_spin")
            else 0.0,
            metal_adaptive_method=str(self.metal_adaptive_method_combo.currentData() or "gaussian")
            if hasattr(self, "metal_adaptive_method_combo")
            else "gaussian",
            metal_morph_close_radius=int(self.metal_gap_bridge_spin.value())
            if hasattr(self, "metal_gap_bridge_spin")
            else (int(self.metal_morph_close_spin.value()) if hasattr(self, "metal_morph_close_spin") else 1),
            metal_morph_open_radius=int(self.metal_speckle_removal_spin.value())
            if hasattr(self, "metal_speckle_removal_spin")
            else (int(self.metal_morph_open_spin.value()) if hasattr(self, "metal_morph_open_spin") else 1),
            metal_display_show_conductors=self.metal_show_conductors_checkbox.isChecked()
            if hasattr(self, "metal_show_conductors_checkbox")
            else True,
            metal_display_show_mask=self.metal_show_mask_checkbox.isChecked()
            if hasattr(self, "metal_show_mask_checkbox")
            else False,
            metal_display_show_contours=self.metal_show_conductors_checkbox.isChecked()
            if hasattr(self, "metal_show_conductors_checkbox")
            else True,
            metal_display_show_rejected=self.metal_show_rejected_checkbox.isChecked()
            if hasattr(self, "metal_show_rejected_checkbox")
            else False,
            metal_display_show_suspicious=self.metal_show_suspicious_checkbox.isChecked()
            if hasattr(self, "metal_show_suspicious_checkbox")
            else True,
            metal_display_show_border_highlight=self.metal_show_border_checkbox.isChecked()
            if hasattr(self, "metal_show_border_checkbox")
            else True,
            metal_debug_visual=str(self.metal_debug_visual_combo.currentData() or "overlay")
            if hasattr(self, "metal_debug_visual_combo")
            else "overlay",
            metal_overlay_opacity=float(self.metal_overlay_opacity_spin.value())
            if hasattr(self, "metal_overlay_opacity_spin")
            else 0.45,
            via_min_roundness=self.via_roundness_spin.value(),
            min_via_width=self.min_via_width_spin.value(),
            max_via_width=None if max_via_width <= 0 else max_via_width,
            min_via_height=self.min_via_height_spin.value(),
            max_via_height=None if max_via_height <= 0 else max_via_height,
            fixed_via_widths=fixed_via_widths,
            fixed_via_heights=fixed_via_heights,
            min_hierarchy_depth=self.min_hierarchy_depth_spin.value(),
            min_inner_hole_area=self.min_inner_hole_area_spin.value(),
            max_hierarchy_depth=None if max_hierarchy_depth <= 0 else max_hierarchy_depth,
            max_hole_area_ratio=None if max_hole_area_ratio <= 0 else max_hole_area_ratio,
        )
