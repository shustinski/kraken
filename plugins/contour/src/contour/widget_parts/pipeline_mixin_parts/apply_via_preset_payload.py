"""pipeline_mixin: apply via preset payload."""
from __future__ import annotations
from contour.value_conversion import to_float
from contour.widget_parts._imports import (
    QCheckBox,
    QSignalBlocker,
    _normalize_bright_via_metal_constraint_mode,
    normalize_via_search_mode,
)


class ApplyViaPresetPayloadMixin:
    def _apply_via_preset_payload(self, payload: dict[str, object]) -> None:
        # A via preset is deliberately limited to the expert detector
        # parameters. Method, polarity, sizes, sensitivity, templates and
        # display controls belong to the recognition configuration itself.
        payload = {
            key: value
            for key, value in payload.items()
            if str(key).startswith("heuristic_")
        }
        if not payload:
            return
        widget_names = {
            "heuristic_line_penalty_scale": "heuristic_line_penalty_spin",
            "heuristic_border_penalty_scale": "heuristic_border_penalty_spin",
            "heuristic_use_bilateral": "heuristic_use_bilateral_checkbox",
        }
        preset_widgets = {
            key: getattr(
                self,
                widget_names.get(key, f"{key}_spin"),
                None,
            )
            for key in payload
        }
        preset_widgets = {
            key: widget
            for key, widget in preset_widgets.items()
            if widget is not None
        }
        preset_blockers = [QSignalBlocker(widget) for widget in preset_widgets.values()]
        try:
            for key, widget in preset_widgets.items():
                assert widget is not None
                value = payload[key]
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                else:
                    widget.setValue(to_float(value))
        finally:
            del preset_blockers
        if hasattr(self, "bright_via_advanced_outer"):
            self.bright_via_advanced_outer.setChecked(True)
        self._on_extraction_settings_changed()
        return

        blockers = [
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
            QSignalBlocker(self.bright_via_diameter_min_spin),
            QSignalBlocker(self.bright_via_diameter_max_spin),
            QSignalBlocker(self.bright_via_clahe_clip_spin),
            QSignalBlocker(self.bright_via_clahe_tile_spin),
            QSignalBlocker(self.bright_via_median_kernel_spin),
            QSignalBlocker(self.bright_via_tophat_kernel_spin),
            QSignalBlocker(self.bright_via_dog_small_spin),
            QSignalBlocker(self.bright_via_dog_large_spin),
            QSignalBlocker(self.bright_via_threshold_percentile_spin),
            QSignalBlocker(self.bright_via_mask_combine_combo),
            QSignalBlocker(self.bright_via_min_area_factor_spin),
            QSignalBlocker(self.bright_via_max_area_factor_spin),
            QSignalBlocker(self.bright_via_min_circularity_spin),
            QSignalBlocker(self.bright_via_min_aspect_spin),
            QSignalBlocker(self.bright_via_max_aspect_spin),
            QSignalBlocker(self.bright_via_bright_center_score_spin),
            QSignalBlocker(self.bright_via_metal_constraint_combo),
            QSignalBlocker(self.bright_via_metal_fraction_spin),
            QSignalBlocker(self.bright_via_max_radial_asymmetry_spin),
            QSignalBlocker(self.bright_via_max_edge_likeness_spin),
            QSignalBlocker(self.bright_via_max_line_likeness_spin),
            QSignalBlocker(self.bright_via_nms_distance_spin),
            QSignalBlocker(self.bright_via_min_final_score_spin),
            QSignalBlocker(self.bright_via_show_rejected_checkbox),
            QSignalBlocker(self.bright_via_hard_asym_checkbox),
            QSignalBlocker(self.bright_via_hard_edge_checkbox),
            QSignalBlocker(self.bright_via_hard_line_checkbox),
            QSignalBlocker(self.debug_candidates_checkbox),
            QSignalBlocker(self.via_roundness_spin),
        ]
        heuristic_widgets = {
            "heuristic_background_sigma": "heuristic_background_sigma_spin",
            "heuristic_analysis_window_scale": "heuristic_analysis_window_scale_spin",
            "heuristic_min_center_brightness": "heuristic_min_center_brightness_spin",
            "heuristic_min_center_contrast": "heuristic_min_center_contrast_spin",
            "heuristic_min_peak_prominence": "heuristic_min_peak_prominence_spin",
            "heuristic_min_compactness": "heuristic_min_compactness_spin",
            "heuristic_min_circularity": "heuristic_min_circularity_spin",
            "heuristic_max_elongation": "heuristic_max_elongation_spin",
            "heuristic_line_penalty_scale": "heuristic_line_penalty_spin",
            "heuristic_border_penalty_scale": "heuristic_border_penalty_spin",
            "heuristic_local_binarize_percentile": "heuristic_local_binarize_percentile_spin",
            "heuristic_min_abs_peak": "heuristic_min_abs_peak_spin",
            **{
                name: f"{name}_spin"
                for name in payload
                if name.startswith("heuristic_") and hasattr(self, f"{name}_spin")
            },
        }
        blockers.extend(
            QSignalBlocker(widget)
            for widget_name in heuristic_widgets.values()
            if (widget := getattr(self, widget_name, None)) is not None
        )
        try:
            mode_index = self.via_search_mode_combo.findData(
                normalize_via_search_mode(payload.get("via_search_mode", self.via_search_mode_combo.currentData()))
            )
            if mode_index >= 0:
                self.via_search_mode_combo.setCurrentIndex(mode_index)
            self.via_white_range_checkbox.setChecked(
                bool(payload.get("via_white_range_enabled", self.via_white_range_checkbox.isChecked()))
            )
            self.via_white_range_min_spin.setValue(
                int(payload.get("via_white_range_min", self.via_white_range_min_spin.value()))
            )
            self.via_white_range_max_spin.setValue(
                int(payload.get("via_white_range_max", self.via_white_range_max_spin.value()))
            )
            self.via_black_range_checkbox.setChecked(
                bool(payload.get("via_black_range_enabled", self.via_black_range_checkbox.isChecked()))
            )
            self.via_black_range_min_spin.setValue(
                int(payload.get("via_black_range_min", self.via_black_range_min_spin.value()))
            )
            self.via_black_range_max_spin.setValue(
                int(payload.get("via_black_range_max", self.via_black_range_max_spin.value()))
            )
            self.via_min_score_spin.setValue(float(payload.get("via_min_score", self.via_min_score_spin.value())))
            self.via_min_contrast_spin.setValue(
                float(payload.get("via_min_contrast", self.via_min_contrast_spin.value()))
            )
            self.via_min_edge_coverage_spin.setValue(
                float(payload.get("via_min_edge_coverage", self.via_min_edge_coverage_spin.value()))
            )
            self.via_spot_line_suppression_spin.setValue(
                float(payload.get("via_spot_line_suppression", self.via_spot_line_suppression_spin.value()))
            )
            self.via_template_min_score_spin.setValue(
                float(payload.get("via_template_min_score", self.via_template_min_score_spin.value()))
            )
            self.via_roundness_spin.setValue(float(payload.get("via_min_roundness", self.via_roundness_spin.value())))
            self.bright_via_diameter_min_spin.setValue(
                int(payload.get("bright_via_diameter_min", self.bright_via_diameter_min_spin.value()))
            )
            self.bright_via_diameter_max_spin.setValue(
                int(payload.get("bright_via_diameter_max", self.bright_via_diameter_max_spin.value()))
            )
            self.bright_via_clahe_clip_spin.setValue(
                float(payload.get("bright_via_clahe_clip_limit", self.bright_via_clahe_clip_spin.value()))
            )
            self.bright_via_clahe_tile_spin.setValue(
                int(payload.get("bright_via_clahe_tile_grid_size", self.bright_via_clahe_tile_spin.value()))
            )
            self.bright_via_median_kernel_spin.setValue(
                int(payload.get("bright_via_median_blur_kernel", self.bright_via_median_kernel_spin.value()))
            )
            self.bright_via_tophat_kernel_spin.setValue(
                int(payload.get("bright_via_tophat_kernel_size", self.bright_via_tophat_kernel_spin.value()))
            )
            self.bright_via_dog_small_spin.setValue(
                float(payload.get("bright_via_dog_sigma_small", self.bright_via_dog_small_spin.value()))
            )
            self.bright_via_dog_large_spin.setValue(
                float(payload.get("bright_via_dog_sigma_large", self.bright_via_dog_large_spin.value()))
            )
            self.bright_via_threshold_percentile_spin.setValue(
                float(payload.get("bright_via_threshold_percentile", self.bright_via_threshold_percentile_spin.value()))
            )
            combine_index = self.bright_via_mask_combine_combo.findData(
                str(payload.get("bright_via_mask_combine_mode", self.bright_via_mask_combine_combo.currentData()))
            )
            if combine_index >= 0:
                self.bright_via_mask_combine_combo.setCurrentIndex(combine_index)
            self.bright_via_min_area_factor_spin.setValue(
                float(payload.get("bright_via_min_area_factor", self.bright_via_min_area_factor_spin.value()))
            )
            self.bright_via_max_area_factor_spin.setValue(
                float(payload.get("bright_via_max_area_factor", self.bright_via_max_area_factor_spin.value()))
            )
            self.bright_via_min_circularity_spin.setValue(
                float(payload.get("bright_via_min_circularity", self.bright_via_min_circularity_spin.value()))
            )
            self.bright_via_min_aspect_spin.setValue(
                float(payload.get("bright_via_min_aspect", self.bright_via_min_aspect_spin.value()))
            )
            self.bright_via_max_aspect_spin.setValue(
                float(payload.get("bright_via_max_aspect", self.bright_via_max_aspect_spin.value()))
            )
            self.bright_via_bright_center_score_spin.setValue(
                float(
                    payload.get(
                        "bright_via_bright_center_min_score",
                        self.bright_via_bright_center_score_spin.value(),
                    )
                )
            )
            metal_mode = _normalize_bright_via_metal_constraint_mode(
                payload.get("bright_via_metal_constraint_mode", self.bright_via_metal_constraint_combo.currentData())
            )
            metal_index = self.bright_via_metal_constraint_combo.findData(metal_mode)
            if metal_index >= 0:
                self.bright_via_metal_constraint_combo.setCurrentIndex(metal_index)
            self.bright_via_metal_fraction_spin.setValue(
                float(payload.get("bright_via_metal_fraction_min", self.bright_via_metal_fraction_spin.value()))
            )
            self.bright_via_max_radial_asymmetry_spin.setValue(
                float(
                    payload.get(
                        "bright_via_max_radial_asymmetry",
                        self.bright_via_max_radial_asymmetry_spin.value(),
                    )
                )
            )
            self.bright_via_max_edge_likeness_spin.setValue(
                float(payload.get("bright_via_max_edge_likeness", self.bright_via_max_edge_likeness_spin.value()))
            )
            self.bright_via_max_line_likeness_spin.setValue(
                float(payload.get("bright_via_max_line_likeness", self.bright_via_max_line_likeness_spin.value()))
            )
            self.bright_via_nms_distance_spin.setValue(
                int(payload.get("bright_via_nms_distance", self.bright_via_nms_distance_spin.value()))
            )
            self.bright_via_min_final_score_spin.setValue(
                float(payload.get("bright_via_min_final_score", self.bright_via_min_final_score_spin.value()))
            )
            self.bright_via_show_rejected_checkbox.setChecked(
                bool(payload.get("bright_via_show_rejected", self.bright_via_show_rejected_checkbox.isChecked()))
            )
            self.bright_via_hard_asym_checkbox.setChecked(
                bool(payload.get("bright_via_hard_reject_on_asymmetry", self.bright_via_hard_asym_checkbox.isChecked()))
            )
            self.bright_via_hard_edge_checkbox.setChecked(
                bool(payload.get("bright_via_hard_reject_on_edge", self.bright_via_hard_edge_checkbox.isChecked()))
            )
            self.bright_via_hard_line_checkbox.setChecked(
                bool(payload.get("bright_via_hard_reject_on_line", self.bright_via_hard_line_checkbox.isChecked()))
            )
            self.debug_candidates_checkbox.setChecked(
                bool(payload.get("debug_enabled", self.debug_candidates_checkbox.isChecked()))
            )
            if hasattr(self, "via_heuristic_polarity_combo") and "via_heuristic_polarity" in payload:
                polarity_index = self.via_heuristic_polarity_combo.findData(str(payload["via_heuristic_polarity"]))
                if polarity_index >= 0:
                    self.via_heuristic_polarity_combo.setCurrentIndex(polarity_index)
            for setting_name, widget_name in heuristic_widgets.items():
                widget = getattr(self, widget_name, None)
                if widget is not None and setting_name in payload:
                    widget.setValue(float(payload[setting_name]))
            if hasattr(self, "heuristic_use_bilateral_checkbox") and "heuristic_use_bilateral" in payload:
                self.heuristic_use_bilateral_checkbox.setChecked(bool(payload["heuristic_use_bilateral"]))
        finally:
            del blockers
        if hasattr(self, "bright_via_advanced_outer"):
            self.bright_via_advanced_outer.setChecked(True)
        self._update_bright_via_diameter_controls_state()
        self._update_via_threshold_controls_state()
        self._on_extraction_settings_changed()
