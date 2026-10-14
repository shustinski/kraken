"""pipeline_mixin: reset bright via parameters."""
from __future__ import annotations
from contour.widget_parts._imports import (
    QSignalBlocker,
)


class ResetBrightViaParametersMixin:
    def _reset_bright_via_parameters(self, *_args) -> None:
        heuristic_defaults = self._heuristic_default_values()
        blockers = [
            QSignalBlocker(self.bright_via_diameter_min_spin),
            QSignalBlocker(self.bright_via_diameter_max_spin),
            QSignalBlocker(self.via_output_diameter_spin),
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
        ]
        blockers.extend(
            QSignalBlocker(widget)
            for name in heuristic_defaults
            if (widget := getattr(self, name, None)) is not None
        )
        try:
            self.bright_via_diameter_min_spin.setValue(8)
            self.bright_via_diameter_max_spin.setValue(8)
            self.via_output_diameter_spin.setValue(8)
            self.bright_via_clahe_clip_spin.setValue(2.0)
            self.bright_via_clahe_tile_spin.setValue(8)
            self.bright_via_median_kernel_spin.setValue(3)
            self.bright_via_tophat_kernel_spin.setValue(11)
            self.bright_via_dog_small_spin.setValue(0.8)
            self.bright_via_dog_large_spin.setValue(2.0)
            self.bright_via_threshold_percentile_spin.setValue(99.0)
            combine_index = self.bright_via_mask_combine_combo.findData("OR")
            if combine_index >= 0:
                self.bright_via_mask_combine_combo.setCurrentIndex(combine_index)
            self.bright_via_min_area_factor_spin.setValue(0.45)
            self.bright_via_max_area_factor_spin.setValue(1.8)
            self.bright_via_min_circularity_spin.setValue(0.30)
            self.bright_via_min_aspect_spin.setValue(0.45)
            self.bright_via_max_aspect_spin.setValue(2.2)
            self.bright_via_bright_center_score_spin.setValue(140.0)
            self.via_white_range_checkbox.setChecked(True)
            self.via_white_range_min_spin.setValue(140)
            self.via_white_range_max_spin.setValue(255)
            self.via_black_range_checkbox.setChecked(False)
            metal_index = self.bright_via_metal_constraint_combo.findData("soft")
            if metal_index >= 0:
                self.bright_via_metal_constraint_combo.setCurrentIndex(metal_index)
            self.bright_via_metal_fraction_spin.setValue(0.3)
            self.bright_via_max_radial_asymmetry_spin.setValue(32.0)
            self.bright_via_max_edge_likeness_spin.setValue(35.0)
            self.bright_via_max_line_likeness_spin.setValue(65.0)
            self.bright_via_nms_distance_spin.setValue(5)
            self.bright_via_min_final_score_spin.setValue(38.0)
            self.bright_via_show_rejected_checkbox.setChecked(True)
            self.bright_via_hard_asym_checkbox.setChecked(True)
            self.bright_via_hard_edge_checkbox.setChecked(False)
            self.bright_via_hard_line_checkbox.setChecked(False)
            for name, value in heuristic_defaults.items():
                widget = getattr(self, name, None)
                if widget is not None:
                    widget.setValue(value)
            if hasattr(self, "heuristic_use_bilateral_checkbox"):
                self.heuristic_use_bilateral_checkbox.setChecked(False)
        finally:
            del blockers
        self._on_extraction_settings_changed()
