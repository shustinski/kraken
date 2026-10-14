"""pipeline_mixin: heuristic default values."""
from __future__ import annotations
from contour.widget_parts._imports import (
    QSignalBlocker,
)


class HeuristicDefaultValuesMixin:
    def _apply_noisy_traces_via_preset(self, *_args) -> None:
        self._apply_via_preset_payload(self._noisy_traces_via_preset_payload())

    def _apply_blurred_via_preset(self, *_args) -> None:
        self._apply_via_preset_payload(self._blurred_via_preset_payload())

    def _reset_via_search_parameters(self, *_args) -> None:
        blockers = [
            QSignalBlocker(self.via_search_mode_combo),
            QSignalBlocker(self.via_min_score_spin),
            QSignalBlocker(self.via_min_contrast_spin),
            QSignalBlocker(self.via_min_edge_coverage_spin),
            QSignalBlocker(self.via_spot_line_suppression_spin),
            QSignalBlocker(self.via_template_min_score_spin),
            QSignalBlocker(self.via_roundness_spin),
        ]
        try:
            mode_index = self.via_search_mode_combo.findData("template")
            if mode_index >= 0:
                self.via_search_mode_combo.setCurrentIndex(mode_index)
            self.via_min_score_spin.setValue(0.35)
            self.via_min_contrast_spin.setValue(14.0)
            self.via_min_edge_coverage_spin.setValue(0.45)
            self.via_spot_line_suppression_spin.setValue(0.65)
            self.via_template_min_score_spin.setValue(0.35)
            self.via_roundness_spin.setValue(40.0)
        finally:
            del blockers
        self._update_via_threshold_controls_state()
        self._on_extraction_settings_changed()

    @staticmethod
    def _heuristic_default_values() -> dict[str, float]:
        return {
            "heuristic_background_sigma_spin": 25.0,
            "heuristic_analysis_window_scale_spin": 3.0,
            "heuristic_min_center_brightness_spin": 0.0,
            "heuristic_min_center_contrast_spin": 50.0,
            "heuristic_min_peak_prominence_spin": 50.0,
            "heuristic_min_compactness_spin": 0.9,
            "heuristic_min_circularity_spin": 0.40,
            "heuristic_max_elongation_spin": 2.5,
            "heuristic_line_penalty_spin": 3.0,
            "heuristic_border_penalty_spin": 1.0,
            "heuristic_local_binarize_percentile_spin": 88.0,
            "heuristic_min_abs_peak_spin": 0.0,
            "heuristic_size_tolerance_range_spin": 0.36,
            "heuristic_size_tolerance_fixed_spin": 0.26,
            "heuristic_max_center_drift_ratio_spin": 0.72,
            "heuristic_max_line_coherence_spin": 0.82,
            "heuristic_min_edge_sharpness_spin": 0.20,
            "heuristic_contrast_score_min_spin": 3.0,
            "heuristic_contrast_score_max_spin": 20.0,
            "heuristic_prominence_score_min_spin": 2.0,
            "heuristic_prominence_score_max_spin": 25.0,
            "heuristic_edge_snr_score_min_spin": 0.70,
            "heuristic_edge_snr_score_max_spin": 2.80,
            "heuristic_edge_quality_floor_spin": 0.55,
            "heuristic_border_balance_scale_spin": 2.0,
            "heuristic_seed_percentile_spin": 90.0,
            "heuristic_w_contrast_spin": 25.0,
            "heuristic_w_prominence_spin": 20.0,
            "heuristic_w_size_spin": 20.0,
            "heuristic_w_compact_spin": 15.0,
            "heuristic_w_round_spin": 10.0,
            "heuristic_w_balance_spin": 10.0,
            "heuristic_w_line_spin": 20.0,
            "heuristic_w_border_spin": 20.0,
        }

    def _reset_heuristic_parameters(self, *_args) -> None:
        defaults = self._heuristic_default_values()
        blockers = [
            QSignalBlocker(widget)
            for name in defaults
            if (widget := getattr(self, name, None)) is not None
        ]
        if hasattr(self, "heuristic_use_bilateral_checkbox"):
            blockers.append(QSignalBlocker(self.heuristic_use_bilateral_checkbox))
        try:
            for name, value in defaults.items():
                widget = getattr(self, name, None)
                if widget is not None:
                    widget.setValue(value)
            if hasattr(self, "heuristic_use_bilateral_checkbox"):
                self.heuristic_use_bilateral_checkbox.setChecked(False)
        finally:
            del blockers
        self._on_extraction_settings_changed()
