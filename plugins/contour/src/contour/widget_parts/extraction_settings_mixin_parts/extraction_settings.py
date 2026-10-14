"""extraction_settings_mixin: extraction settings."""
from __future__ import annotations
from contour.widget_parts._imports import (
    VIA_SEARCH_MODE_HYBRID,
    VIA_SEARCH_MODE_TEMPLATE,
    ContourExtractionSettings,
    QSignalBlocker,
    _normalize_bright_via_metal_constraint_mode,
    normalize_recognition_mode,
    normalize_via_search_mode,
)


class ExtractionSettingsMixin:
    def _set_extraction_settings(self, settings: ContourExtractionSettings) -> None:
        blockers = [
            QSignalBlocker(self.retrieval_mode_combo),
            QSignalBlocker(self.approximation_mode_combo),
            QSignalBlocker(self.epsilon_spin),
            QSignalBlocker(self.epsilon_slider),
            QSignalBlocker(self.epsilon_relative_checkbox),
            QSignalBlocker(self.min_area_spin),
            QSignalBlocker(self.max_area_spin),
            QSignalBlocker(self.min_perimeter_spin),
            QSignalBlocker(self.min_points_spin),
            QSignalBlocker(self.min_polygon_width_spin),
            QSignalBlocker(self.max_perimeter_spin),
            QSignalBlocker(self.min_bbox_width_spin),
            QSignalBlocker(self.max_bbox_width_spin),
            QSignalBlocker(self.min_bbox_height_spin),
            QSignalBlocker(self.max_bbox_height_spin),
            QSignalBlocker(self.min_aspect_ratio_spin),
            QSignalBlocker(self.max_aspect_ratio_spin),
            QSignalBlocker(self.exclude_border_touching_checkbox),
            QSignalBlocker(self.min_solidity_spin),
            QSignalBlocker(self.min_extent_spin),
            QSignalBlocker(self.min_polygon_angle_spin),
            QSignalBlocker(self.conductor_gradient_checkbox),
            QSignalBlocker(self.conductor_gradient_min_strength_spin),
            QSignalBlocker(self.conductor_gradient_band_radius_spin),
            QSignalBlocker(self.via_search_mode_combo),
            QSignalBlocker(self.via_output_diameter_spin),
            QSignalBlocker(self.via_white_range_checkbox),
            QSignalBlocker(self.via_white_range_min_spin),
            QSignalBlocker(self.via_white_range_max_spin),
            QSignalBlocker(self.via_black_range_checkbox),
            QSignalBlocker(self.via_black_range_min_spin),
            QSignalBlocker(self.via_black_range_max_spin),
            QSignalBlocker(self.via_min_score_spin),
            QSignalBlocker(self.via_min_contrast_spin),
            QSignalBlocker(self.via_min_edge_coverage_spin),
            QSignalBlocker(self.via_spot_line_suppression_spin),
            QSignalBlocker(self.via_template_min_score_spin),
            QSignalBlocker(self.debug_candidates_checkbox),
            QSignalBlocker(self.via_roundness_spin),
            QSignalBlocker(self.min_via_width_spin),
            QSignalBlocker(self.max_via_width_spin),
            QSignalBlocker(self.min_via_height_spin),
            QSignalBlocker(self.max_via_height_spin),
            QSignalBlocker(self.min_hierarchy_depth_spin),
            QSignalBlocker(self.min_inner_hole_area_spin),
            QSignalBlocker(self.max_hierarchy_depth_spin),
            QSignalBlocker(self.max_hole_area_ratio_spin),
            QSignalBlocker(self.advanced_extraction_checkbox),
        ]
        for _mw in (
            "metal_preset_combo",
            "metal_min_contrast_slider",
            "metal_auto_contrast_step_spin",
            "metal_auto_source_contrast_step_spin",
            "metal_auto_directional_gap_bridge_spin",
            "metal_auto_directional_gap_min_source_spin",
            "metal_gap_bridge_spin",
            "metal_speckle_removal_spin",
            "metal_epsilon_spin",
            "metal_min_width_spin",
            "metal_max_width_spin",
            "metal_conductor_size_offset_slider",
            "metal_show_conductors_checkbox",
            "metal_show_rejected_checkbox",
            "metal_show_suspicious_checkbox",
            "metal_show_border_checkbox",
            "metal_show_mask_checkbox",
            "metal_debug_visual_combo",
            "metal_segmentation_strategy_combo",
            "metal_overlay_opacity_spin",
            "metal_min_area_spin",
            "metal_max_area_spin",
            "metal_min_perimeter_spin",
            "metal_max_perimeter_spin",
            "metal_min_length_spin",
            "metal_min_points_spin",
            "metal_min_angle_spin",
            "metal_min_object_source_contrast_spin",
            "metal_min_object_rim_contrast_spin",
            "metal_min_object_rim_area_fraction_spin",
            "metal_min_hole_source_contrast_spin",
            "metal_min_hole_source_contrast_fraction_spin",
            "metal_ws_smoothing_spin",
            "metal_ws_core_margin_spin",
            "metal_ws_groove_margin_spin",
            "metal_ws_rim_probe_spin",
            "metal_ws_seed_speckle_spin",
            "metal_ws_valley_span_spin",
            "metal_ws_valley_depth_spin",
            "metal_rw_beta_spin",
            "metal_rw_iterations_spin",
            "metal_gc_iterations_spin",
            "metal_recon_erode_spin",
            "metal_boundary_relief_spin",
            "metal_boundary_background_spin",
            "metal_adaptive_block_spin",
            "metal_adaptive_c_spin",
            "metal_adaptive_method_combo",
            "metal_approximation_checkbox",
            "metal_hierarchy_combo",
            "metal_border_handling_combo",
            "metal_advanced_group",
        ):
            _w = getattr(self, _mw, None)
            if _w is not None:
                blockers.append(QSignalBlocker(_w))
        if hasattr(self, "recognition_mode_combo"):
            blockers.append(QSignalBlocker(self.recognition_mode_combo))
        if hasattr(self, "via_show_detected_checkbox"):
            blockers.append(QSignalBlocker(self.via_show_detected_checkbox))
        try:
            prof = str(getattr(settings, "extraction_profile", "conductors") or "conductors")
            rm = normalize_recognition_mode(getattr(settings, "recognition_mode", "conductors"))
            if prof == "vias" or (getattr(settings, "object_type", "conductor") == "via" and rm != "disabled"):
                rdata = "via"
            elif rm == "disabled":
                rdata = "disabled"
            else:
                rdata = "conductors"
            if hasattr(self, "recognition_mode_combo"):
                ridx = self.recognition_mode_combo.findData(rdata)
                if ridx >= 0:
                    self.recognition_mode_combo.setCurrentIndex(ridx)
            if hasattr(self, "recognition_stack"):
                if rdata == "via":
                    self.recognition_stack.setVisible(False)
                else:
                    self.recognition_stack.setVisible(True)
                    self.recognition_stack.setCurrentIndex(0 if rdata == "disabled" else 1)
            self._active_extraction_profile = "vias" if rdata == "via" else "conductors"
            retrieval_index = self.retrieval_mode_combo.findData(settings.retrieval_mode)
            if retrieval_index >= 0:
                self.retrieval_mode_combo.setCurrentIndex(retrieval_index)
            approximation_index = self.approximation_mode_combo.findData(settings.approximation_mode)
            if approximation_index >= 0:
                self.approximation_mode_combo.setCurrentIndex(approximation_index)
            self.epsilon_spin.setValue(float(settings.epsilon))
            if hasattr(self, "epsilon_slider"):
                self.epsilon_slider.setValue(min(1000, max(0, round(float(settings.epsilon) * 100.0))))
            self.epsilon_relative_checkbox.setChecked(bool(settings.epsilon_relative))
            self.min_area_spin.setValue(float(settings.min_area))
            self.max_area_spin.setValue(0.0 if settings.max_area is None else float(settings.max_area))
            self.min_perimeter_spin.setValue(float(settings.min_perimeter))
            self.max_perimeter_spin.setValue(0.0 if settings.max_perimeter is None else float(settings.max_perimeter))
            self.min_points_spin.setValue(int(settings.min_points))
            self.min_polygon_width_spin.setValue(float(getattr(settings, "min_polygon_width_px", 0.0) or 0.0))
            self.min_bbox_width_spin.setValue(int(settings.min_bbox_width))
            self.max_bbox_width_spin.setValue(0 if settings.max_bbox_width is None else int(settings.max_bbox_width))
            self.min_bbox_height_spin.setValue(int(settings.min_bbox_height))
            self.max_bbox_height_spin.setValue(0 if settings.max_bbox_height is None else int(settings.max_bbox_height))
            self.min_aspect_ratio_spin.setValue(float(settings.min_aspect_ratio))
            self.max_aspect_ratio_spin.setValue(
                0.0 if settings.max_aspect_ratio is None else float(settings.max_aspect_ratio)
            )
            self.exclude_border_touching_checkbox.setChecked(bool(settings.exclude_border_touching))
            self.min_solidity_spin.setValue(float(settings.min_solidity))
            self.min_extent_spin.setValue(float(settings.min_extent))
            self.min_polygon_angle_spin.setValue(float(settings.min_polygon_angle))
            self.conductor_gradient_checkbox.setChecked(bool(settings.conductor_gradient_enabled))
            self.conductor_gradient_min_strength_spin.setValue(float(settings.conductor_gradient_min_strength))
            self.conductor_gradient_band_radius_spin.setValue(int(settings.conductor_gradient_band_radius))
            via_search_mode_index = self.via_search_mode_combo.findData(
                normalize_via_search_mode(settings.via_search_mode)
            )
            if via_search_mode_index >= 0:
                self.via_search_mode_combo.setCurrentIndex(via_search_mode_index)
            if hasattr(self, "via_heuristic_polarity_combo"):
                _po = str(getattr(settings, "via_heuristic_polarity", "auto") or "auto")
                _pidx = self.via_heuristic_polarity_combo.findData(_po)
                if _pidx >= 0:
                    self.via_heuristic_polarity_combo.setCurrentIndex(_pidx)
            if hasattr(self, "via_fixed_diameters_edit"):
                self.via_fixed_diameters_edit.setText(
                    str(getattr(settings, "via_fixed_diameters_text", "6, 8, 10") or "6, 8, 10")
                )
            if hasattr(self, "via_template_nms_distance_spin"):
                self.via_template_nms_distance_spin.setValue(
                    int(getattr(settings, "via_template_nms_distance", 4) or 4)
                )
            if hasattr(self, "via_template_scale_min_spin"):
                self.via_template_scale_min_spin.setValue(
                    float(getattr(settings, "via_template_scale_min", 0.9) or 0.9)
                )
            if hasattr(self, "via_template_scale_max_spin"):
                self.via_template_scale_max_spin.setValue(
                    float(getattr(settings, "via_template_scale_max", 1.1) or 1.1)
                )
            if hasattr(self, "via_template_scale_step_spin"):
                self.via_template_scale_step_spin.setValue(
                    float(getattr(settings, "via_template_scale_step", 0.1) or 0.1)
                )
            if hasattr(self, "heuristic_background_sigma_spin"):
                self.heuristic_background_sigma_spin.setValue(
                    float(getattr(settings, "heuristic_background_sigma", 25.0) or 25.0)
                )
            if hasattr(self, "heuristic_analysis_window_scale_spin"):
                self.heuristic_analysis_window_scale_spin.setValue(
                    float(getattr(settings, "heuristic_analysis_window_scale", 3.0) or 3.0)
                )
            if hasattr(self, "heuristic_min_center_brightness_spin"):
                self.heuristic_min_center_brightness_spin.setValue(
                    float(getattr(settings, "heuristic_min_center_brightness", 0.0) or 0.0)
                )
            if hasattr(self, "heuristic_min_center_contrast_spin"):
                self.heuristic_min_center_contrast_spin.setValue(
                    float(getattr(settings, "heuristic_min_center_contrast", 50.0) or 0.0)
                )
            if hasattr(self, "heuristic_min_peak_prominence_spin"):
                self.heuristic_min_peak_prominence_spin.setValue(
                    float(getattr(settings, "heuristic_min_peak_prominence", 50.0) or 0.0)
                )
            if hasattr(self, "heuristic_min_compactness_spin"):
                self.heuristic_min_compactness_spin.setValue(
                    float(getattr(settings, "heuristic_min_compactness", 0.9) or 0.0)
                )
            if hasattr(self, "heuristic_min_circularity_spin"):
                self.heuristic_min_circularity_spin.setValue(
                    float(getattr(settings, "heuristic_min_circularity", 0.40))
                )
            if hasattr(self, "heuristic_max_elongation_spin"):
                self.heuristic_max_elongation_spin.setValue(
                    float(getattr(settings, "heuristic_max_elongation", 2.5) or 2.5)
                )
            if hasattr(self, "heuristic_line_penalty_spin"):
                self.heuristic_line_penalty_spin.setValue(
                    float(getattr(settings, "heuristic_line_penalty_scale", 3.0) or 3.0)
                )
            if hasattr(self, "heuristic_border_penalty_spin"):
                self.heuristic_border_penalty_spin.setValue(
                    float(getattr(settings, "heuristic_border_penalty_scale", 1.0) or 1.0)
                )
            if hasattr(self, "heuristic_local_binarize_percentile_spin"):
                self.heuristic_local_binarize_percentile_spin.setValue(
                    float(getattr(settings, "heuristic_local_binarize_percentile", 88.0) or 88.0)
                )
            if hasattr(self, "heuristic_min_abs_peak_spin"):
                self.heuristic_min_abs_peak_spin.setValue(
                    float(getattr(settings, "heuristic_min_abs_peak", 0.0) or 0.0)
                )
            if hasattr(self, "heuristic_use_bilateral_checkbox"):
                self.heuristic_use_bilateral_checkbox.setChecked(
                    bool(getattr(settings, "heuristic_use_bilateral", False))
                )
            for setting_name, widget_name, default_value in (
                ("heuristic_size_tolerance_range", "heuristic_size_tolerance_range_spin", 0.36),
                ("heuristic_size_tolerance_fixed", "heuristic_size_tolerance_fixed_spin", 0.26),
                ("heuristic_max_center_drift_ratio", "heuristic_max_center_drift_ratio_spin", 0.72),
                ("heuristic_max_line_coherence", "heuristic_max_line_coherence_spin", 0.82),
                ("heuristic_min_edge_sharpness", "heuristic_min_edge_sharpness_spin", 0.20),
                ("heuristic_contrast_score_min", "heuristic_contrast_score_min_spin", 3.0),
                ("heuristic_contrast_score_max", "heuristic_contrast_score_max_spin", 20.0),
                ("heuristic_prominence_score_min", "heuristic_prominence_score_min_spin", 2.0),
                ("heuristic_prominence_score_max", "heuristic_prominence_score_max_spin", 25.0),
                ("heuristic_edge_snr_score_min", "heuristic_edge_snr_score_min_spin", 0.70),
                ("heuristic_edge_snr_score_max", "heuristic_edge_snr_score_max_spin", 2.80),
                ("heuristic_edge_quality_floor", "heuristic_edge_quality_floor_spin", 0.55),
                ("heuristic_border_balance_scale", "heuristic_border_balance_scale_spin", 2.0),
                ("heuristic_seed_percentile", "heuristic_seed_percentile_spin", 90.0),
                ("heuristic_w_contrast", "heuristic_w_contrast_spin", 25.0),
                ("heuristic_w_prominence", "heuristic_w_prominence_spin", 20.0),
                ("heuristic_w_size", "heuristic_w_size_spin", 20.0),
                ("heuristic_w_compact", "heuristic_w_compact_spin", 15.0),
                ("heuristic_w_round", "heuristic_w_round_spin", 10.0),
                ("heuristic_w_balance", "heuristic_w_balance_spin", 10.0),
                ("heuristic_w_line", "heuristic_w_line_spin", 20.0),
                ("heuristic_w_border", "heuristic_w_border_spin", 20.0),
            ):
                widget = getattr(self, widget_name, None)
                if widget is not None:
                    widget.setValue(float(getattr(settings, setting_name, default_value)))
            if hasattr(self, "bright_via_mode_stack") and hasattr(self, "via_search_mode_combo"):
                mode = normalize_via_search_mode(self.via_search_mode_combo.currentData())
                self.bright_via_mode_stack.setCurrentIndex(
                    2 if mode in (VIA_SEARCH_MODE_TEMPLATE, VIA_SEARCH_MODE_HYBRID) else 1
                )
            self.via_white_range_checkbox.setChecked(bool(settings.via_white_range_enabled))
            self.via_white_range_min_spin.setValue(int(settings.via_white_range_min))
            self.via_white_range_max_spin.setValue(int(settings.via_white_range_max))
            self.via_black_range_checkbox.setChecked(bool(settings.via_black_range_enabled))
            self.via_black_range_min_spin.setValue(int(settings.via_black_range_min))
            self.via_black_range_max_spin.setValue(int(settings.via_black_range_max))
            if hasattr(self, "bright_via_bright_center_score_spin"):
                self.bright_via_bright_center_score_spin.setValue(
                    float(settings.via_white_range_min)
                    if bool(settings.via_white_range_enabled)
                    else float(settings.bright_via_bright_center_min_score)
                )
            self.via_min_score_spin.setValue(float(settings.via_min_score))
            self.via_min_contrast_spin.setValue(float(settings.via_min_contrast))
            self.via_min_edge_coverage_spin.setValue(float(settings.via_min_edge_coverage))
            self.via_template_min_score_spin.setValue(float(settings.via_template_min_score))
            self.via_spot_line_suppression_spin.setValue(float(settings.via_spot_line_suppression))
            self.bright_via_diameter_min_spin.setValue(int(settings.bright_via_diameter_min))
            self.bright_via_diameter_max_spin.setValue(int(settings.bright_via_diameter_max))
            self.via_output_diameter_spin.setValue(
                int(
                    getattr(
                        settings,
                        "via_output_diameter",
                        round(
                            (
                                int(settings.bright_via_diameter_min)
                                + int(settings.bright_via_diameter_max)
                            )
                            * 0.5
                        ),
                    )
                )
            )
            self.bright_via_clahe_clip_spin.setValue(float(settings.bright_via_clahe_clip_limit))
            self.bright_via_clahe_tile_spin.setValue(int(settings.bright_via_clahe_tile_grid_size))
            self.bright_via_median_kernel_spin.setValue(int(settings.bright_via_median_blur_kernel))
            self.bright_via_tophat_kernel_spin.setValue(int(settings.bright_via_tophat_kernel_size))
            self.bright_via_dog_small_spin.setValue(float(settings.bright_via_dog_sigma_small))
            self.bright_via_dog_large_spin.setValue(float(settings.bright_via_dog_sigma_large))
            self.bright_via_threshold_percentile_spin.setValue(float(settings.bright_via_threshold_percentile))
            combine_index = self.bright_via_mask_combine_combo.findData(settings.bright_via_mask_combine_mode)
            if combine_index >= 0:
                self.bright_via_mask_combine_combo.setCurrentIndex(combine_index)
            self.bright_via_min_area_factor_spin.setValue(float(settings.bright_via_min_area_factor))
            self.bright_via_max_area_factor_spin.setValue(float(settings.bright_via_max_area_factor))
            self.bright_via_min_circularity_spin.setValue(float(settings.bright_via_min_circularity))
            self.bright_via_min_aspect_spin.setValue(float(settings.bright_via_min_aspect))
            self.bright_via_max_aspect_spin.setValue(float(settings.bright_via_max_aspect))
            self.bright_via_bright_center_score_spin.setValue(float(settings.bright_via_bright_center_min_score))
            metal_index = self.bright_via_metal_constraint_combo.findData(
                _normalize_bright_via_metal_constraint_mode(settings.bright_via_metal_constraint_mode)
            )
            if metal_index >= 0:
                self.bright_via_metal_constraint_combo.setCurrentIndex(metal_index)
            self.bright_via_metal_fraction_spin.setValue(float(settings.bright_via_metal_fraction_min))
            self.bright_via_max_radial_asymmetry_spin.setValue(float(settings.bright_via_max_radial_asymmetry))
            self.bright_via_max_edge_likeness_spin.setValue(float(settings.bright_via_max_edge_likeness))
            self.bright_via_max_line_likeness_spin.setValue(float(settings.bright_via_max_line_likeness))
            self.bright_via_nms_distance_spin.setValue(int(settings.bright_via_nms_distance))
            self.bright_via_min_final_score_spin.setValue(float(settings.bright_via_min_final_score))
            self.bright_via_show_rejected_checkbox.setChecked(bool(settings.bright_via_show_rejected))
            self.bright_via_hard_asym_checkbox.setChecked(bool(settings.bright_via_hard_reject_on_asymmetry))
            self.bright_via_hard_edge_checkbox.setChecked(bool(settings.bright_via_hard_reject_on_edge))
            self.bright_via_hard_line_checkbox.setChecked(bool(settings.bright_via_hard_reject_on_line))
            if hasattr(self, "bright_via_min_isolation_spin"):
                self.bright_via_min_isolation_spin.setValue(float(settings.bright_via_min_isolation_score))
            self._via_template_images = self._normalize_via_template_images(settings.via_template_images)
            self._via_template_min_scores = [
                max(0.0, min(1.0, float(value))) for value in settings.via_template_min_scores
            ]
            self._via_template_diameters = [max(1, int(value)) for value in settings.via_template_diameters]
            self._refresh_via_template_list()
            self.debug_candidates_checkbox.setChecked(bool(settings.debug_enabled))
            if hasattr(self, "via_show_detected_checkbox"):
                self.via_show_detected_checkbox.setChecked(bool(getattr(settings, "via_display_show_detected", True)))
            self.via_roundness_spin.setValue(float(settings.via_min_roundness))
            self.min_via_width_spin.setValue(int(settings.min_via_width))
            self.max_via_width_spin.setValue(0 if settings.max_via_width is None else int(settings.max_via_width))
            self.min_via_height_spin.setValue(int(settings.min_via_height))
            self.max_via_height_spin.setValue(0 if settings.max_via_height is None else int(settings.max_via_height))
            self._suspend_fixed_via_updates = True
            self._clear_fixed_via_rows()
            for width, height in zip(settings.fixed_via_widths, settings.fixed_via_heights, strict=False):
                self._add_fixed_via_row(width=width, height=height)
            self._suspend_fixed_via_updates = False
            self.min_hierarchy_depth_spin.setValue(int(settings.min_hierarchy_depth))
            self.min_inner_hole_area_spin.setValue(float(getattr(settings, "min_inner_hole_area", 100.0)))
            self.max_hierarchy_depth_spin.setValue(
                0 if settings.max_hierarchy_depth is None else int(settings.max_hierarchy_depth)
            )
            self.max_hole_area_ratio_spin.setValue(
                0.0 if settings.max_hole_area_ratio is None else float(settings.max_hole_area_ratio)
            )
            if hasattr(self, "metal_preset_combo"):
                if hasattr(self, "_refresh_metal_preset_combo"):
                    self._refresh_metal_preset_combo()
                mp_key = str(getattr(settings, "metal_preset", "standard") or "standard")
                label_map = {
                    "standard": "Стандартный" if self._ui_language == "ru" else "Standard",
                }
                mp_label = label_map.get(mp_key, label_map["standard"])
                mp = self.metal_preset_combo.findText(mp_label)
                if mp >= 0:
                    self.metal_preset_combo.setCurrentIndex(mp)
                elif mp_key != "standard":
                    user_index = self.metal_preset_combo.findText(mp_key)
                    if user_index >= 0:
                        self.metal_preset_combo.setCurrentIndex(user_index)
                if hasattr(self, "metal_min_contrast_slider"):
                    self.metal_min_contrast_slider.setValue(
                        round(float(getattr(settings, "metal_min_contrast", 50.0)))
                    )
                if hasattr(self, "metal_min_object_source_contrast_spin"):
                    self.metal_min_object_source_contrast_spin.setValue(
                        float(getattr(settings, "metal_min_object_source_contrast", 12.0))
                    )
                if hasattr(self, "metal_min_object_rim_contrast_spin"):
                    self.metal_min_object_rim_contrast_spin.setValue(
                        float(getattr(settings, "metal_min_object_rim_contrast", 36.0))
                    )
                if hasattr(self, "metal_min_object_rim_area_fraction_spin"):
                    self.metal_min_object_rim_area_fraction_spin.setValue(
                        float(getattr(settings, "metal_min_object_rim_area_fraction", 0.001))
                    )
                if hasattr(self, "metal_auto_contrast_step_spin"):
                    self.metal_auto_contrast_step_spin.setValue(
                        float(getattr(settings, "metal_auto_contrast_step", 10.0))
                    )
                if hasattr(self, "metal_auto_source_contrast_step_spin"):
                    self.metal_auto_source_contrast_step_spin.setValue(
                        float(getattr(settings, "metal_auto_source_contrast_step", 4.0))
                    )
                if hasattr(self, "metal_auto_directional_gap_bridge_spin"):
                    self.metal_auto_directional_gap_bridge_spin.setValue(
                        int(getattr(settings, "metal_auto_directional_gap_bridge_px", 3))
                    )
                if hasattr(self, "metal_auto_directional_gap_min_source_spin"):
                    self.metal_auto_directional_gap_min_source_spin.setValue(
                        float(
                            getattr(
                                settings,
                                "metal_auto_directional_gap_min_source_intensity",
                                45.0,
                            )
                        )
                    )
                if hasattr(self, "metal_gap_bridge_spin"):
                    self.metal_gap_bridge_spin.setValue(int(getattr(settings, "metal_gap_bridge_px", 1)))
                if hasattr(self, "metal_speckle_removal_spin"):
                    self.metal_speckle_removal_spin.setValue(int(getattr(settings, "metal_speckle_removal_px", 1)))
                if hasattr(self, "metal_epsilon_spin"):
                    self.metal_epsilon_spin.setValue(float(settings.epsilon))
                self.metal_min_width_spin.setValue(float(getattr(settings, "metal_min_trace_width_px", 8.0) or 8.0))
                mw = getattr(settings, "metal_max_trace_width_px", None)
                self.metal_max_width_spin.setValue(0.0 if mw is None else float(mw))
                if hasattr(self, "metal_conductor_size_offset_slider"):
                    self.metal_conductor_size_offset_slider.setValue(
                        round(float(getattr(settings, "metal_conductor_size_offset_px", 0.0) or 0.0))
                    )
                self.metal_min_area_spin.setValue(float(getattr(settings, "metal_min_area", 60.0) or 60.0))
                ma = getattr(settings, "metal_max_area", None)
                self.metal_max_area_spin.setValue(0.0 if ma is None else float(ma))
                self.metal_min_perimeter_spin.setValue(float(getattr(settings, "metal_min_perimeter", 32.0) or 32.0))
                mp2 = getattr(settings, "metal_max_perimeter", None)
                self.metal_max_perimeter_spin.setValue(0.0 if mp2 is None else float(mp2))
                self.metal_min_length_spin.setValue(float(getattr(settings, "metal_min_trace_length_px", 8.0) or 8.0))
                self.metal_min_points_spin.setValue(int(settings.min_points))
                self.metal_min_angle_spin.setValue(float(settings.min_polygon_angle))
                self.metal_min_hole_source_contrast_spin.setValue(
                    float(getattr(settings, "metal_min_hole_source_contrast", 8.0))
                )
                self.metal_min_hole_source_contrast_fraction_spin.setValue(
                    float(getattr(settings, "metal_min_hole_source_contrast_fraction", 0.35))
                )
                self.metal_approximation_checkbox.setChecked(
                    bool(getattr(settings, "metal_approximation_enabled", True))
                )
                if hasattr(self, "metal_segmentation_strategy_combo"):
                    self._apply_metal_strategy_to_combo(settings)
                if hasattr(self, "metal_strategy_parameter_widgets"):
                    self._set_metal_strategy_parameter_widgets(
                        getattr(settings, "metal_strategy_parameters", None)
                    )
                if hasattr(self, "metal_ws_smoothing_spin"):
                    self.metal_ws_smoothing_spin.setValue(
                        float(getattr(settings, "metal_watershed_smoothing_sigma", 1.2) or 1.2)
                    )
                if hasattr(self, "metal_ws_core_margin_spin"):
                    self.metal_ws_core_margin_spin.setValue(
                        float(getattr(settings, "metal_watershed_core_margin", 23.0) or 0.0)
                    )
                if hasattr(self, "metal_ws_groove_margin_spin"):
                    self.metal_ws_groove_margin_spin.setValue(
                        float(getattr(settings, "metal_watershed_groove_margin", 19.0) or 0.0)
                    )
                if hasattr(self, "metal_ws_rim_probe_spin"):
                    self.metal_ws_rim_probe_spin.setValue(
                        int(getattr(settings, "metal_watershed_rim_probe_px", 2) or 1)
                    )
                if hasattr(self, "metal_ws_seed_speckle_spin"):
                    self.metal_ws_seed_speckle_spin.setValue(
                        int(getattr(settings, "metal_watershed_seed_speckle_px", 1) or 0)
                    )
                if hasattr(self, "metal_ws_valley_span_spin"):
                    self.metal_ws_valley_span_spin.setValue(
                        int(getattr(settings, "metal_watershed_valley_span_px", 5) or 0)
                    )
                if hasattr(self, "metal_ws_valley_depth_spin"):
                    self.metal_ws_valley_depth_spin.setValue(
                        float(getattr(settings, "metal_watershed_valley_depth", 65.0) or 0.0)
                    )
                if hasattr(self, "metal_rw_beta_spin"):
                    self.metal_rw_beta_spin.setValue(
                        float(getattr(settings, "metal_random_walker_beta", 90.0) or 90.0)
                    )
                if hasattr(self, "metal_rw_iterations_spin"):
                    self.metal_rw_iterations_spin.setValue(
                        int(getattr(settings, "metal_random_walker_iterations", 160) or 160)
                    )
                if hasattr(self, "metal_gc_iterations_spin"):
                    self.metal_gc_iterations_spin.setValue(
                        int(getattr(settings, "metal_graph_cut_iterations", 5) or 5)
                    )
                if hasattr(self, "metal_recon_erode_spin"):
                    self.metal_recon_erode_spin.setValue(
                        int(getattr(settings, "metal_reconstruction_erode_px", 0) or 0)
                    )
                if hasattr(self, "metal_boundary_relief_spin"):
                    self.metal_boundary_relief_spin.setValue(
                        float(getattr(settings, "metal_boundary_relief", 16.0) or 16.0)
                    )
                if hasattr(self, "metal_boundary_background_spin"):
                    self.metal_boundary_background_spin.setValue(
                        float(getattr(settings, "metal_boundary_background_sigma", 12.0) or 12.0)
                    )
                if hasattr(self, "metal_adaptive_block_spin"):
                    self.metal_adaptive_block_spin.setValue(
                        int(getattr(settings, "metal_adaptive_block_size", 0) or 0)
                    )
                if hasattr(self, "metal_adaptive_c_spin"):
                    self.metal_adaptive_c_spin.setValue(
                        float(getattr(settings, "metal_adaptive_c", 0.0) or 0.0)
                    )
                if hasattr(self, "metal_adaptive_method_combo"):
                    method_index = self.metal_adaptive_method_combo.findData(
                        str(getattr(settings, "metal_adaptive_method", "gaussian") or "gaussian")
                    )
                    if method_index >= 0:
                        self.metal_adaptive_method_combo.setCurrentIndex(method_index)
                _hm = self.metal_hierarchy_combo.findData(
                    str(getattr(settings, "metal_hierarchy_mode", "full") or "full")
                )
                if _hm >= 0:
                    self.metal_hierarchy_combo.setCurrentIndex(_hm)
                _bh = str(getattr(settings, "metal_border_handling", "mark") or "mark")
                _bhi = self.metal_border_handling_combo.findData(_bh)
                if _bhi >= 0:
                    self.metal_border_handling_combo.setCurrentIndex(_bhi)
                self.metal_show_conductors_checkbox.setChecked(
                    bool(getattr(settings, "metal_display_show_conductors", True))
                )
                self.metal_show_rejected_checkbox.setChecked(
                    bool(getattr(settings, "metal_display_show_rejected", False))
                )
                self.metal_show_suspicious_checkbox.setChecked(
                    bool(getattr(settings, "metal_display_show_suspicious", True))
                )
                self.metal_show_border_checkbox.setChecked(
                    bool(getattr(settings, "metal_display_show_border_highlight", True))
                )
                self.metal_show_mask_checkbox.setChecked(bool(getattr(settings, "metal_display_show_mask", False)))
                _dv = str(getattr(settings, "metal_debug_visual", "overlay") or "overlay")
                _dvi = self.metal_debug_visual_combo.findData(_dv)
                if _dvi >= 0:
                    self.metal_debug_visual_combo.setCurrentIndex(_dvi)
                self.metal_overlay_opacity_spin.setValue(
                    float(getattr(settings, "metal_overlay_opacity", 0.45) or 0.45)
                )
            self._update_via_size_controls_state()
            self._update_via_threshold_controls_state()
            self._update_extraction_profile_controls_state()
        finally:
            self._suspend_fixed_via_updates = False
            self._ignore_extraction_profile_change = False
            del blockers
