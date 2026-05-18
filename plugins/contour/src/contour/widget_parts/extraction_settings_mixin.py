from __future__ import annotations

from ._imports import *  # noqa: F403


class WidgetExtractionSettingsMixin:
    def _auto_apply_pipeline(self) -> None:
        if not self._workspace.current_image_path:
            return
        if hasattr(self, "auto_apply_checkbox") and not self.auto_apply_checkbox.isChecked():
            return
        self._abort_in_flight_interactive_processing(preview=True, prepared=True)
        self.process_current_image(debounced=False)

    def _try_extract_if_recognition_enabled(self) -> None:
        if not hasattr(self, "recognition_mode_combo"):
            return
        if str(self.recognition_mode_combo.currentData() or "") == "disabled":
            return
        current_path = self._workspace.current_image_path
        if not current_path:
            return
        state = self._workspace.current_state
        if state is None or state.image_path != current_path or state.preprocessed_image is None:
            return
        if state.pipeline_config != self.get_pipeline():
            return
        self.process_current_image(debounced=False)

    def _start_auto_tune_from_reference(self) -> None:
        current_state = self._workspace.current_state
        current_image_path = self._workspace.current_image_path
        reference_polygons = self.get_polygons()

        if current_state is None or current_state.source_image is None or current_image_path is None:
            self._append_log(
                self._tr(
                    "no_image_selected_log",
                    "Изображение не выбрано." if self._ui_language == "ru" else "No image selected.",
                )
            )
            return
        if not reference_polygons:
            self._append_log(
                self._tr(
                    "auto_tune_no_reference_log",
                    "Для автоподбора сначала нарисуйте эталонный полигон или область."
                    if self._ui_language == "ru"
                    else "Draw at least one reference polygon before running auto-fit.",
                )
            )
            return
        if self._auto_tune_running_request_id is not None:
            self._append_log(
                self._tr(
                    "auto_tune_already_running_log",
                    "Автоподбор уже выполняется." if self._ui_language == "ru" else "Auto-fit is already running.",
                )
            )
            return

        self._auto_tune_request_serial += 1
        request_id = self._auto_tune_request_serial
        self._auto_tune_running_request_id = request_id
        self._append_log(
            self._tr(
                "auto_tune_started_log",
                "Запущен автоподбор по {count} полигонам."
                if self._ui_language == "ru"
                else "Auto-fit started using {count} reference polygons.",
                count=len(reference_polygons),
            )
        )
        worker = AutoTuneRunnable(
            request_id=request_id,
            image_path=current_image_path,
            source_image=current_state.source_image,
            reference_polygons=reference_polygons,
        )
        worker.signals.result.connect(self._on_auto_tune_result)
        worker.signals.error.connect(self._on_auto_tune_error)
        worker.signals.finished.connect(self._on_auto_tune_finished)
        self._auto_tune_thread_pool.start(worker)
        self._refresh_busy_indicator()

    def _apply_auto_tune_result(self, result: AutoTuneResult) -> None:
        self._pipeline = PreprocessingPipeline.from_dict(result.pipeline_config)
        self._populate_pipeline_list()
        self._set_extraction_settings(result.contour_settings)
        self.process_current_image()

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
            QSignalBlocker(self.via_size_mode_combo),
            QSignalBlocker(self.via_search_mode_combo),
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
            "metal_sensitivity_slider",
            "metal_min_width_spin",
            "metal_max_width_spin",
            "metal_min_length_spin",
            "metal_use_wide_gradient_checkbox",
            "metal_segmentation_method_combo",
            "metal_sensitivity_combo",
            "metal_show_conductors_checkbox",
            "metal_show_rejected_checkbox",
            "metal_show_suspicious_checkbox",
            "metal_show_border_checkbox",
            "metal_show_mask_checkbox",
            "metal_debug_visual_combo",
            "metal_overlay_opacity_spin",
            "metal_min_area_spin",
            "metal_max_area_spin",
            "metal_min_perimeter_spin",
            "metal_max_perimeter_spin",
            "metal_epsilon_spin",
            "metal_min_points_spin",
            "metal_min_angle_spin",
            "metal_approximation_checkbox",
            "metal_hierarchy_combo",
            "metal_allowed_angles_combo",
            "metal_angle_tolerance_spin",
            "metal_straightness_spin",
            "metal_t_junction_checkbox",
            "metal_border_handling_combo",
            "metal_validity_checkbox",
            "metal_morph_close_spin",
            "metal_morph_open_spin",
            "metal_wide_grad_radius_spin",
            "metal_wide_grad_conf_spin",
            "metal_wide_grad_pair_len_spin",
            "metal_wide_grad_parallel_spin",
            "metal_wide_grad_gap_spin",
            "metal_wide_grad_overlap_spin",
            "metal_advanced_group",
        ):
            _w = getattr(self, _mw, None)
            if _w is not None:
                blockers.append(QSignalBlocker(_w))
        if hasattr(self, "recognition_mode_combo"):
            blockers.append(QSignalBlocker(self.recognition_mode_combo))
        if hasattr(self, "via_search_sensitivity_combo"):
            blockers.append(QSignalBlocker(self.via_search_sensitivity_combo))
        if hasattr(self, "via_show_detected_checkbox"):
            blockers.append(QSignalBlocker(self.via_show_detected_checkbox))
        if hasattr(self, "via_debug_gradient_map_checkbox"):
            blockers.append(QSignalBlocker(self.via_debug_gradient_map_checkbox))
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
            via_size_mode_index = self.via_size_mode_combo.findData(normalize_via_size_mode(settings.via_size_mode))
            if via_size_mode_index >= 0:
                self.via_size_mode_combo.setCurrentIndex(via_size_mode_index)
            via_search_mode_index = self.via_search_mode_combo.findData(
                normalize_via_search_mode(settings.via_search_mode)
            )
            if via_search_mode_index >= 0:
                self.via_search_mode_combo.setCurrentIndex(via_search_mode_index)
            if hasattr(self, "via_diameter_size_mode_combo"):
                _di = self.via_diameter_size_mode_combo.findData(normalize_via_size_mode(settings.via_size_mode))
                if _di >= 0:
                    self.via_diameter_size_mode_combo.setCurrentIndex(_di)
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
            if hasattr(self, "heuristic_min_center_contrast_spin"):
                self.heuristic_min_center_contrast_spin.setValue(
                    float(getattr(settings, "heuristic_min_center_contrast", 6.0) or 0.0)
                )
            if hasattr(self, "heuristic_min_peak_prominence_spin"):
                self.heuristic_min_peak_prominence_spin.setValue(
                    float(getattr(settings, "heuristic_min_peak_prominence", 4.0) or 0.0)
                )
            if hasattr(self, "heuristic_min_compactness_spin"):
                self.heuristic_min_compactness_spin.setValue(
                    float(getattr(settings, "heuristic_min_compactness", 0.12) or 0.0)
                )
            if hasattr(self, "heuristic_max_elongation_spin"):
                self.heuristic_max_elongation_spin.setValue(
                    float(getattr(settings, "heuristic_max_elongation", 3.2) or 3.2)
                )
            if hasattr(self, "heuristic_line_penalty_spin"):
                self.heuristic_line_penalty_spin.setValue(
                    float(getattr(settings, "heuristic_line_penalty_scale", 1.0) or 1.0)
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
            if hasattr(self, "bright_via_mode_stack") and hasattr(self, "via_search_mode_combo"):
                _ist = self.via_search_mode_combo.currentData() == VIA_SEARCH_MODE_TEMPLATE
                self.bright_via_mode_stack.setCurrentIndex(1 if _ist else 0)
            self.via_white_range_checkbox.setChecked(bool(settings.via_white_range_enabled))
            self.via_white_range_min_spin.setValue(int(settings.via_white_range_min))
            self.via_white_range_max_spin.setValue(int(settings.via_white_range_max))
            self.via_black_range_checkbox.setChecked(bool(settings.via_black_range_enabled))
            self.via_black_range_min_spin.setValue(int(settings.via_black_range_min))
            self.via_black_range_max_spin.setValue(int(settings.via_black_range_max))
            self.via_min_score_spin.setValue(float(settings.via_min_score))
            self.via_min_contrast_spin.setValue(float(settings.via_min_contrast))
            self.via_min_edge_coverage_spin.setValue(float(settings.via_min_edge_coverage))
            self.via_template_min_score_spin.setValue(float(settings.via_template_min_score))
            self.via_spot_line_suppression_spin.setValue(float(settings.via_spot_line_suppression))
            self.bright_via_diameter_min_spin.setValue(int(settings.bright_via_diameter_min))
            self.bright_via_diameter_max_spin.setValue(int(settings.bright_via_diameter_max))
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
            self._via_template_images = self._normalize_via_template_images(settings.via_template_images)
            self._refresh_via_template_list()
            self.debug_candidates_checkbox.setChecked(bool(settings.debug_enabled))
            if hasattr(self, "via_debug_gradient_map_checkbox"):
                self.via_debug_gradient_map_checkbox.setChecked(bool(settings.debug_gradient_map_enabled))
            if hasattr(self, "via_show_detected_checkbox"):
                self.via_show_detected_checkbox.setChecked(bool(getattr(settings, "via_display_show_detected", True)))
            if hasattr(self, "via_search_sensitivity_combo"):
                vs = str(getattr(settings, "via_search_sensitivity", "medium") or "medium")
                vs_idx = self.via_search_sensitivity_combo.findData(vs)
                if vs_idx >= 0:
                    self.via_search_sensitivity_combo.setCurrentIndex(vs_idx)
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
                mp = self.metal_preset_combo.findData(str(getattr(settings, "metal_preset", "standard") or "standard"))
                if mp >= 0:
                    self.metal_preset_combo.setCurrentIndex(mp)
                self.metal_sensitivity_slider.setValue(int(getattr(settings, "metal_sensitivity_0_100", 50)))
                if hasattr(self, "metal_sensitivity_value_label"):
                    self.metal_sensitivity_value_label.setText(str(self.metal_sensitivity_slider.value()))
                self.metal_min_width_spin.setValue(float(getattr(settings, "metal_min_trace_width_px", 8.0) or 8.0))
                mw = getattr(settings, "metal_max_trace_width_px", None)
                self.metal_max_width_spin.setValue(0.0 if mw is None else float(mw))
                self.metal_min_length_spin.setValue(
                    float(getattr(settings, "metal_min_trace_length_px", 8.0) or 8.0)
                )
                if hasattr(self, "metal_use_wide_gradient_checkbox"):
                    self.metal_use_wide_gradient_checkbox.setChecked(
                        bool(getattr(settings, "metal_use_wide_conductor_gradient", False))
                    )
                if hasattr(self, "metal_wide_grad_radius_spin"):
                    self.metal_wide_grad_radius_spin.setValue(
                        int(getattr(settings, "metal_wide_gradient_profile_radius_px", 8) or 8)
                    )
                if hasattr(self, "metal_wide_grad_conf_spin"):
                    self.metal_wide_grad_conf_spin.setValue(
                        float(getattr(settings, "metal_wide_gradient_min_direction_confidence", 0.15) or 0.15)
                    )
                if hasattr(self, "metal_wide_grad_pair_len_spin"):
                    self.metal_wide_grad_pair_len_spin.setValue(
                        float(getattr(settings, "metal_wide_gradient_min_pair_length_px", 24.0) or 24.0)
                    )
                if hasattr(self, "metal_wide_grad_parallel_spin"):
                    self.metal_wide_grad_parallel_spin.setValue(
                        float(getattr(settings, "metal_wide_gradient_parallel_tolerance_deg", 10.0) or 10.0)
                    )
                if hasattr(self, "metal_wide_grad_gap_spin"):
                    self.metal_wide_grad_gap_spin.setValue(
                        int(getattr(settings, "metal_wide_gradient_max_edge_gap_px", 5) or 5)
                    )
                if hasattr(self, "metal_wide_grad_overlap_spin"):
                    self.metal_wide_grad_overlap_spin.setValue(
                        float(getattr(settings, "metal_wide_gradient_min_overlap_ratio", 0.5) or 0.5)
                    )
                _smi = self.metal_segmentation_method_combo.findData(
                    str(getattr(settings, "metal_segmentation_method", "otsu") or "otsu")
                )
                if _smi >= 0:
                    self.metal_segmentation_method_combo.setCurrentIndex(_smi)
                _st = str(getattr(settings, "metal_sensitivity", "medium") or "medium")
                _sti = self.metal_sensitivity_combo.findData(_st)
                if _sti >= 0:
                    self.metal_sensitivity_combo.setCurrentIndex(_sti)
                self.metal_min_area_spin.setValue(float(getattr(settings, "metal_min_area", 60.0) or 60.0))
                ma = getattr(settings, "metal_max_area", None)
                self.metal_max_area_spin.setValue(0.0 if ma is None else float(ma))
                self.metal_min_perimeter_spin.setValue(
                    float(getattr(settings, "metal_min_perimeter", 32.0) or 32.0)
                )
                mp2 = getattr(settings, "metal_max_perimeter", None)
                self.metal_max_perimeter_spin.setValue(0.0 if mp2 is None else float(mp2))
                self.metal_epsilon_spin.setValue(float(settings.epsilon))
                self.metal_min_points_spin.setValue(int(settings.min_points))
                self.metal_min_angle_spin.setValue(float(settings.min_polygon_angle))
                self.metal_approximation_checkbox.setChecked(
                    bool(getattr(settings, "metal_approximation_enabled", True))
                )
                _hm = self.metal_hierarchy_combo.findData(
                    str(getattr(settings, "metal_hierarchy_mode", "full") or "full")
                )
                if _hm >= 0:
                    self.metal_hierarchy_combo.setCurrentIndex(_hm)
                _aa = str(getattr(settings, "metal_allowed_angles", "free") or "free")
                _aai = self.metal_allowed_angles_combo.findData(_aa)
                if _aai >= 0:
                    self.metal_allowed_angles_combo.setCurrentIndex(_aai)
                self.metal_angle_tolerance_spin.setValue(
                    float(getattr(settings, "metal_angle_tolerance_deg", 7.0) or 7.0)
                )
                self.metal_straightness_spin.setValue(
                    float(getattr(settings, "metal_min_straightness", 0.2) or 0.2)
                )
                self.metal_t_junction_checkbox.setChecked(
                    bool(getattr(settings, "metal_allow_t_junction", True))
                )
                _bh = str(getattr(settings, "metal_border_handling", "mark") or "mark")
                _bhi = self.metal_border_handling_combo.findData(_bh)
                if _bhi >= 0:
                    self.metal_border_handling_combo.setCurrentIndex(_bhi)
                self.metal_validity_checkbox.setChecked(
                    bool(getattr(settings, "metal_check_contour_validity", False))
                )
                self.metal_morph_close_spin.setValue(int(getattr(settings, "metal_morph_close_radius", 1) or 1))
                self.metal_morph_open_spin.setValue(int(getattr(settings, "metal_morph_open_radius", 0) or 0))
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
                self.metal_show_mask_checkbox.setChecked(bool(getattr(settings, "metal_display_show_mask", True)))
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

    def _current_contour_settings(self) -> ContourExtractionSettings:
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
        if rec_mode == "via" and hasattr(self, "via_diameter_size_mode_combo"):
            via_size_mode = normalize_via_size_mode(self.via_diameter_size_mode_combo.currentData())
        else:
            via_size_mode = normalize_via_size_mode(self.via_size_mode_combo.currentData())
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
            bright_via_diameter_min=self.bright_via_diameter_min_spin.value(),
            bright_via_diameter_max=self.bright_via_diameter_max_spin.value(),
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
            bright_via_bright_center_min_score=self.bright_via_bright_center_score_spin.value(),
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
            via_template_images=[template.copy() for template in self._via_template_images],
            via_template_nms_distance=self.via_template_nms_distance_spin.value()
            if hasattr(self, "via_template_nms_distance_spin")
            else 4,
            via_template_scale_min=self.via_template_scale_min_spin.value()
            if hasattr(self, "via_template_scale_min_spin")
            else 0.9,
            via_template_scale_max=self.via_template_scale_max_spin.value()
            if hasattr(self, "via_template_scale_max_spin")
            else 1.1,
            via_template_scale_step=self.via_template_scale_step_spin.value()
            if hasattr(self, "via_template_scale_step_spin")
            else 0.1,
            via_heuristic_polarity=str(
                self.via_heuristic_polarity_combo.currentData() or "auto"
            )
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
            heuristic_min_center_contrast=self.heuristic_min_center_contrast_spin.value()
            if hasattr(self, "heuristic_min_center_contrast_spin")
            else 6.0,
            heuristic_min_peak_prominence=self.heuristic_min_peak_prominence_spin.value()
            if hasattr(self, "heuristic_min_peak_prominence_spin")
            else 4.0,
            heuristic_min_compactness=self.heuristic_min_compactness_spin.value()
            if hasattr(self, "heuristic_min_compactness_spin")
            else 0.12,
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
            debug_enabled=self.debug_candidates_checkbox.isChecked(),
            debug_gradient_map_enabled=(
                self.via_debug_gradient_map_checkbox.isChecked()
                if hasattr(self, "via_debug_gradient_map_checkbox")
                else self.debug_candidates_checkbox.isChecked()
            ),
            recognition_mode=raw_rec,
            via_search_sensitivity=str(
                self.via_search_sensitivity_combo.currentData() or "medium"
            )
            if hasattr(self, "via_search_sensitivity_combo")
            else "medium",
            via_display_show_detected=(
                self.via_show_detected_checkbox.isChecked()
                if hasattr(self, "via_show_detected_checkbox")
                else True
            ),
            via_display_show_candidates=self.debug_candidates_checkbox.isChecked(),
            metal_structural_pipeline=(raw_rec == "conductors"),
            metal_preset=str(self.metal_preset_combo.currentData() or "standard")
            if hasattr(self, "metal_preset_combo")
            else "standard",
            metal_segmentation_method=str(self.metal_segmentation_method_combo.currentData() or "none")
            if hasattr(self, "metal_segmentation_method_combo")
            else "none",
            metal_sensitivity=str(self.metal_sensitivity_combo.currentData() or "medium")
            if hasattr(self, "metal_sensitivity_combo")
            else "medium",
            metal_sensitivity_0_100=int(self.metal_sensitivity_slider.value())
            if hasattr(self, "metal_sensitivity_slider")
            else 50,
            metal_min_object_area=self.metal_min_area_spin.value()
            if hasattr(self, "metal_min_area_spin")
            else 60.0,
            metal_min_trace_width_px=float(self.metal_min_width_spin.value())
            if hasattr(self, "metal_min_width_spin")
            else 8.0,
            metal_max_trace_width_px=None
            if not hasattr(self, "metal_max_width_spin") or self.metal_max_width_spin.value() <= 0
            else float(self.metal_max_width_spin.value()),
            metal_min_trace_length_px=float(self.metal_min_length_spin.value())
            if hasattr(self, "metal_min_length_spin")
            else 8.0,
            metal_use_wide_conductor_gradient=(
                self.metal_use_wide_gradient_checkbox.isChecked()
                if hasattr(self, "metal_use_wide_gradient_checkbox")
                else False
            ),
            metal_wide_gradient_profile_radius_px=int(self.metal_wide_grad_radius_spin.value())
            if hasattr(self, "metal_wide_grad_radius_spin")
            else 8,
            metal_wide_gradient_min_direction_confidence=float(self.metal_wide_grad_conf_spin.value())
            if hasattr(self, "metal_wide_grad_conf_spin")
            else 0.15,
            metal_wide_gradient_min_pair_length_px=float(self.metal_wide_grad_pair_len_spin.value())
            if hasattr(self, "metal_wide_grad_pair_len_spin")
            else 24.0,
            metal_wide_gradient_parallel_tolerance_deg=float(self.metal_wide_grad_parallel_spin.value())
            if hasattr(self, "metal_wide_grad_parallel_spin")
            else 10.0,
            metal_wide_gradient_max_edge_gap_px=int(self.metal_wide_grad_gap_spin.value())
            if hasattr(self, "metal_wide_grad_gap_spin")
            else 5,
            metal_wide_gradient_min_overlap_ratio=float(self.metal_wide_grad_overlap_spin.value())
            if hasattr(self, "metal_wide_grad_overlap_spin")
            else 0.5,
            metal_allowed_angles=str(self.metal_allowed_angles_combo.currentData() or "free")
            if hasattr(self, "metal_allowed_angles_combo")
            else "free",
            metal_angle_tolerance_deg=float(self.metal_angle_tolerance_spin.value())
            if hasattr(self, "metal_angle_tolerance_spin")
            else 7.0,
            metal_min_straightness=float(self.metal_straightness_spin.value())
            if hasattr(self, "metal_straightness_spin")
            else 0.2,
            metal_allow_t_junction=self.metal_t_junction_checkbox.isChecked()
            if hasattr(self, "metal_t_junction_checkbox")
            else True,
            metal_border_handling=str(self.metal_border_handling_combo.currentData() or "mark")
            if hasattr(self, "metal_border_handling_combo")
            else "mark",
            metal_check_contour_validity=self.metal_validity_checkbox.isChecked()
            if hasattr(self, "metal_validity_checkbox")
            else True,
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
            metal_morph_close_radius=self.metal_morph_close_spin.value()
            if hasattr(self, "metal_morph_close_spin")
            else 1,
            metal_morph_open_radius=self.metal_morph_open_spin.value()
            if hasattr(self, "metal_morph_open_spin")
            else 0,
            metal_display_show_conductors=self.metal_show_conductors_checkbox.isChecked()
            if hasattr(self, "metal_show_conductors_checkbox")
            else True,
            metal_display_show_mask=self.metal_show_mask_checkbox.isChecked()
            if hasattr(self, "metal_show_mask_checkbox")
            else True,
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

    def _on_recognition_mode_changed(self, *_args) -> None:
        if not hasattr(self, "recognition_mode_combo") or not hasattr(self, "recognition_stack"):
            return
        data = str(self.recognition_mode_combo.currentData() or "conductors")
        if data == "disabled":
            self._active_extraction_profile = "conductors"
            self._sync_recognition_stack_visibility()
        elif data == "conductors":
            self._active_extraction_profile = "conductors"
            self._sync_recognition_stack_visibility()
            self._set_extraction_settings(self._contour_settings_profiles["conductors"])
        else:
            self._active_extraction_profile = "vias"
            self.recognition_stack.setVisible(False)
            self._set_extraction_settings(self._contour_settings_profiles["vias"])
        if hasattr(self, "via_group"):
            self.via_group.setVisible(self._active_extraction_profile == "vias" and data == "disabled")
        self.polygon_editor.set_debug_candidates([])
        if hasattr(self, "_update_extraction_profile_controls_state"):
            self._update_extraction_profile_controls_state()
        self._on_extraction_settings_changed()

    def _on_via_search_sensitivity_changed(self, *_args) -> None:
        self._apply_via_search_sensitivity_profile()
        self._on_extraction_settings_changed()

    def _apply_via_search_sensitivity_profile(self) -> None:
        if not hasattr(self, "via_search_sensitivity_combo"):
            return
        level = str(self.via_search_sensitivity_combo.currentData() or "medium")
        profiles = {
            "low": (99.5, 8.0, 55.0, 0.40, True, True, True, True),
            "medium": (99.0, 6.0, 38.0, 0.30, False, False, False, False),
            "high": (98.0, 4.0, 32.0, 0.22, False, False, False, False),
        }
        pct, bright, final, circ, ha, he, hl, _ = profiles.get(level, profiles["medium"])
        blockers = [
            QSignalBlocker(self.bright_via_threshold_percentile_spin),
            QSignalBlocker(self.bright_via_bright_center_score_spin),
            QSignalBlocker(self.bright_via_min_final_score_spin),
            QSignalBlocker(self.bright_via_min_circularity_spin),
            QSignalBlocker(self.bright_via_hard_asym_checkbox),
            QSignalBlocker(self.bright_via_hard_edge_checkbox),
            QSignalBlocker(self.bright_via_hard_line_checkbox),
        ]
        try:
            self.bright_via_threshold_percentile_spin.setValue(pct)
            self.bright_via_bright_center_score_spin.setValue(bright)
            self.bright_via_min_final_score_spin.setValue(final)
            self.bright_via_min_circularity_spin.setValue(circ)
            self.bright_via_hard_asym_checkbox.setChecked(ha)
            self.bright_via_hard_edge_checkbox.setChecked(he)
            self.bright_via_hard_line_checkbox.setChecked(hl)
        finally:
            del blockers

    def _on_via_display_settings_changed(self, *_args) -> None:
        if hasattr(self, "polygon_editor") and hasattr(self, "via_show_detected_checkbox"):
            self.polygon_editor.set_polygon_category_visible("via", self.via_show_detected_checkbox.isChecked())
        self._on_extraction_settings_changed()

    def _on_metal_overlay_opacity_changed(self, value: float) -> None:
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_gradient_overlay_opacity(float(value))
        self._on_extraction_settings_changed()

    def _on_metal_sensitivity_slider_changed(self, value: int) -> None:
        if hasattr(self, "metal_sensitivity_value_label"):
            self.metal_sensitivity_value_label.setText(str(int(value)))
        self._on_extraction_settings_changed()

    def _metal_preset_table(self) -> dict[str, dict[str, float | int | str]]:
        return {
            "standard": {
                "sens": 50,
                "close": 1,
                "open": 0,
                "min_w": 8.0,
                "max_w": 0.0,
                "min_l": 8.0,
                "min_a": 60.0,
                "max_a": 0.0,
                "min_p": 32.0,
                "max_p": 0.0,
                "str": 0.2,
                "tol": 7.0,
                "tok": "medium",
                "angles": "free",
            },
            "dense": {
                "sens": 42,
                "close": 5,
                "open": 0,
                "min_w": 6.0,
                "max_w": 85.0,
                "min_l": 18.0,
                "min_a": 45.0,
                "max_a": 0.0,
                "min_p": 28.0,
                "max_p": 0.0,
                "str": 0.52,
                "tol": 9.0,
                "tok": "high",
            },
            "thin_traces": {
                "sens": 58,
                "close": 2,
                "open": 1,
                "min_w": 4.0,
                "max_w": 24.0,
                "min_l": 28.0,
                "min_a": 35.0,
                "max_a": 0.0,
                "min_p": 26.0,
                "max_p": 0.0,
                "str": 0.64,
                "tol": 6.0,
                "tok": "medium",
            },
            "wide_traces": {
                "sens": 44,
                "close": 4,
                "open": 0,
                "min_w": 14.0,
                "max_w": 0.0,
                "min_l": 24.0,
                "min_a": 100.0,
                "max_a": 0.0,
                "min_p": 42.0,
                "max_p": 0.0,
                "str": 0.5,
                "tol": 8.0,
                "tok": "medium",
            },
            "weak_contrast": {
                "sens": 68,
                "close": 2,
                "open": 0,
                "min_w": 6.0,
                "max_w": 0.0,
                "min_l": 14.0,
                "min_a": 40.0,
                "max_a": 0.0,
                "min_p": 24.0,
                "max_p": 0.0,
                "str": 0.35,
                "tol": 10.0,
                "tok": "high",
            },
            "noisy_sem": {
                "sens": 36,
                "close": 4,
                "open": 1,
                "min_w": 10.0,
                "max_w": 0.0,
                "min_l": 32.0,
                "min_a": 85.0,
                "max_a": 0.0,
                "min_p": 40.0,
                "max_p": 0.0,
                "str": 0.68,
                "tol": 10.0,
                "tok": "low",
            },
            "conservative": {
                "sens": 62,
                "close": 2,
                "open": 0,
                "min_w": 10.0,
                "max_w": 48.0,
                "min_l": 36.0,
                "min_a": 100.0,
                "max_a": 0.0,
                "min_p": 44.0,
                "max_p": 0.0,
                "str": 0.72,
                "tol": 5.0,
                "tok": "low",
            },
            "angles_45_90": {
                "sens": 52,
                "close": 1,
                "open": 0,
                "min_w": 8.0,
                "max_w": 0.0,
                "min_l": 12.0,
                "min_a": 60.0,
                "max_a": 0.0,
                "min_p": 32.0,
                "max_p": 0.0,
                "str": 0.35,
                "tol": 6.0,
                "tok": "medium",
                "angles": "45_90",
            },
        }

    def _on_metal_preset_changed(self, *_args) -> None:
        if not hasattr(self, "metal_preset_combo"):
            return
        key = str(self.metal_preset_combo.currentData() or "standard")
        pr = self._metal_preset_table().get(key)
        if not pr:
            self._on_extraction_settings_changed()
            return
        self.metal_sensitivity_slider.setValue(int(pr["sens"]))
        self.metal_morph_close_spin.setValue(int(pr["close"]))
        self.metal_morph_open_spin.setValue(int(pr["open"]))
        self.metal_min_width_spin.setValue(float(pr["min_w"]))
        self.metal_max_width_spin.setValue(float(pr["max_w"]))
        self.metal_min_length_spin.setValue(float(pr["min_l"]))
        self.metal_min_area_spin.setValue(float(pr["min_a"]))
        self.metal_max_area_spin.setValue(float(pr["max_a"]))
        self.metal_min_perimeter_spin.setValue(float(pr["min_p"]))
        self.metal_max_perimeter_spin.setValue(float(pr["max_p"]))
        self.metal_straightness_spin.setValue(float(pr["str"]))
        self.metal_angle_tolerance_spin.setValue(float(pr["tol"]))
        _ti = self.metal_sensitivity_combo.findData(str(pr["tok"]))
        if _ti >= 0:
            self.metal_sensitivity_combo.setCurrentIndex(_ti)
        if hasattr(self, "metal_allowed_angles_combo") and "angles" in pr:
            _ai = self.metal_allowed_angles_combo.findData(str(pr["angles"]))
            if _ai >= 0:
                self.metal_allowed_angles_combo.setCurrentIndex(_ai)
        self._on_extraction_settings_changed()

    def _preview_metal_mask(self, *_args) -> None:
        if hasattr(self, "metal_debug_visual_combo"):
            idx = self.metal_debug_visual_combo.findData("metal_binary_mask")
            if idx >= 0:
                self.metal_debug_visual_combo.setCurrentIndex(idx)
        if hasattr(self, "metal_show_mask_checkbox"):
            self.metal_show_mask_checkbox.setChecked(True)
        self._refresh_gradient_overlay()

    def _reset_metal_parameters(self, *_args) -> None:
        defaults = ContourExtractionSettings()
        if hasattr(self, "metal_preset_combo"):
            self.metal_preset_combo.setCurrentIndex(self.metal_preset_combo.findData("standard"))
        if hasattr(self, "metal_sensitivity_slider"):
            self.metal_sensitivity_slider.setValue(int(defaults.metal_sensitivity_0_100))
        if hasattr(self, "metal_sensitivity_value_label"):
            self.metal_sensitivity_value_label.setText(str(int(defaults.metal_sensitivity_0_100)))
        if hasattr(self, "metal_min_width_spin"):
            self.metal_min_width_spin.setValue(float(defaults.metal_min_trace_width_px))
        if hasattr(self, "metal_max_width_spin"):
            mw = defaults.metal_max_trace_width_px
            self.metal_max_width_spin.setValue(0.0 if mw is None else float(mw))
        if hasattr(self, "metal_min_length_spin"):
            self.metal_min_length_spin.setValue(float(defaults.metal_min_trace_length_px))
        if hasattr(self, "metal_segmentation_method_combo"):
            ix = self.metal_segmentation_method_combo.findData(defaults.metal_segmentation_method)
            if ix >= 0:
                self.metal_segmentation_method_combo.setCurrentIndex(ix)
        if hasattr(self, "metal_sensitivity_combo"):
            ix = self.metal_sensitivity_combo.findData(defaults.metal_sensitivity)
            if ix >= 0:
                self.metal_sensitivity_combo.setCurrentIndex(ix)
        if hasattr(self, "metal_min_area_spin"):
            self.metal_min_area_spin.setValue(float(defaults.metal_min_area))
        if hasattr(self, "metal_max_area_spin"):
            ma = defaults.metal_max_area
            self.metal_max_area_spin.setValue(0.0 if ma is None else float(ma))
        if hasattr(self, "metal_min_perimeter_spin"):
            self.metal_min_perimeter_spin.setValue(float(defaults.metal_min_perimeter))
        if hasattr(self, "metal_max_perimeter_spin"):
            mp = defaults.metal_max_perimeter
            self.metal_max_perimeter_spin.setValue(0.0 if mp is None else float(mp))
        if hasattr(self, "metal_epsilon_spin"):
            self.metal_epsilon_spin.setValue(float(defaults.epsilon))
        if hasattr(self, "metal_min_points_spin"):
            self.metal_min_points_spin.setValue(int(defaults.min_points))
        if hasattr(self, "metal_min_angle_spin"):
            self.metal_min_angle_spin.setValue(float(defaults.min_polygon_angle))
        if hasattr(self, "metal_approximation_checkbox"):
            self.metal_approximation_checkbox.setChecked(bool(defaults.metal_approximation_enabled))
        if hasattr(self, "metal_hierarchy_combo"):
            ix = self.metal_hierarchy_combo.findData(defaults.metal_hierarchy_mode)
            if ix >= 0:
                self.metal_hierarchy_combo.setCurrentIndex(ix)
        if hasattr(self, "metal_allowed_angles_combo"):
            ix = self.metal_allowed_angles_combo.findData(defaults.metal_allowed_angles)
            if ix >= 0:
                self.metal_allowed_angles_combo.setCurrentIndex(ix)
        if hasattr(self, "metal_angle_tolerance_spin"):
            self.metal_angle_tolerance_spin.setValue(float(defaults.metal_angle_tolerance_deg))
        if hasattr(self, "metal_straightness_spin"):
            self.metal_straightness_spin.setValue(float(defaults.metal_min_straightness))
        if hasattr(self, "metal_t_junction_checkbox"):
            self.metal_t_junction_checkbox.setChecked(bool(defaults.metal_allow_t_junction))
        if hasattr(self, "metal_border_handling_combo"):
            ix = self.metal_border_handling_combo.findData(defaults.metal_border_handling)
            if ix >= 0:
                self.metal_border_handling_combo.setCurrentIndex(ix)
        if hasattr(self, "metal_validity_checkbox"):
            self.metal_validity_checkbox.setChecked(bool(defaults.metal_check_contour_validity))
        if hasattr(self, "metal_morph_close_spin"):
            self.metal_morph_close_spin.setValue(int(defaults.metal_morph_close_radius))
        if hasattr(self, "metal_morph_open_spin"):
            self.metal_morph_open_spin.setValue(int(defaults.metal_morph_open_radius))
        if hasattr(self, "metal_use_wide_gradient_checkbox"):
            self.metal_use_wide_gradient_checkbox.setChecked(bool(defaults.metal_use_wide_conductor_gradient))
        if hasattr(self, "metal_wide_grad_radius_spin"):
            self.metal_wide_grad_radius_spin.setValue(int(defaults.metal_wide_gradient_profile_radius_px))
        if hasattr(self, "metal_wide_grad_conf_spin"):
            self.metal_wide_grad_conf_spin.setValue(float(defaults.metal_wide_gradient_min_direction_confidence))
        if hasattr(self, "metal_wide_grad_pair_len_spin"):
            self.metal_wide_grad_pair_len_spin.setValue(float(defaults.metal_wide_gradient_min_pair_length_px))
        if hasattr(self, "metal_wide_grad_parallel_spin"):
            self.metal_wide_grad_parallel_spin.setValue(float(defaults.metal_wide_gradient_parallel_tolerance_deg))
        if hasattr(self, "metal_wide_grad_gap_spin"):
            self.metal_wide_grad_gap_spin.setValue(int(defaults.metal_wide_gradient_max_edge_gap_px))
        if hasattr(self, "metal_wide_grad_overlap_spin"):
            self.metal_wide_grad_overlap_spin.setValue(float(defaults.metal_wide_gradient_min_overlap_ratio))
        if hasattr(self, "metal_show_conductors_checkbox"):
            self.metal_show_conductors_checkbox.setChecked(bool(defaults.metal_display_show_conductors))
        if hasattr(self, "metal_show_rejected_checkbox"):
            self.metal_show_rejected_checkbox.setChecked(bool(defaults.metal_display_show_rejected))
        if hasattr(self, "metal_show_suspicious_checkbox"):
            self.metal_show_suspicious_checkbox.setChecked(bool(defaults.metal_display_show_suspicious))
        if hasattr(self, "metal_show_border_checkbox"):
            self.metal_show_border_checkbox.setChecked(bool(defaults.metal_display_show_border_highlight))
        if hasattr(self, "metal_show_mask_checkbox"):
            self.metal_show_mask_checkbox.setChecked(bool(defaults.metal_display_show_mask))
        if hasattr(self, "metal_debug_visual_combo"):
            ix = self.metal_debug_visual_combo.findData(defaults.metal_debug_visual)
            if ix >= 0:
                self.metal_debug_visual_combo.setCurrentIndex(ix)
        if hasattr(self, "metal_overlay_opacity_spin"):
            self.metal_overlay_opacity_spin.setValue(float(defaults.metal_overlay_opacity))
        self._on_extraction_settings_changed()

    def _set_recognition_status(self, kind: str, message: str | None = None) -> None:
        if not hasattr(self, "recognition_status_label"):
            return
        if kind == "disabled":
            self.recognition_status_label.clear()
            return
        if self._ui_language == "ru":
            texts = {
                "idle": "Готово",
                "disabled": "Извлечение отключено",
                "updating": "Выполняется обработка…",
                "error": "Ошибка",
            }
        else:
            texts = {
                "idle": "Ready",
                "disabled": "Recognition off",
                "updating": "Updating…",
                "error": "Error",
            }
        text = message or texts.get(kind, texts["idle"])
        self.recognition_status_label.setText(text)


