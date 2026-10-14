"""processing_mixin: apply contact feedback adjustments."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    QSignalBlocker,
    QSpinBox,
)


class ApplyContactFeedbackAdjustmentsMixin:
    def _apply_contact_feedback_adjustments(self: Any, adjustments: list[object]) -> bool:
        widget_by_field = {
            "bright_via_min_final_score": "bright_via_min_final_score_spin",
            "heuristic_min_circularity": "heuristic_min_circularity_spin",
            "heuristic_min_compactness": "heuristic_min_compactness_spin",
            "heuristic_min_center_contrast": "heuristic_min_center_contrast_spin",
            "heuristic_min_peak_prominence": "heuristic_min_peak_prominence_spin",
            "heuristic_min_edge_sharpness": "heuristic_min_edge_sharpness_spin",
            "heuristic_max_elongation": "heuristic_max_elongation_spin",
            "heuristic_max_line_coherence": "heuristic_max_line_coherence_spin",
            "heuristic_min_center_brightness": "heuristic_min_center_brightness_spin",
            "heuristic_max_center_drift_ratio": "heuristic_max_center_drift_ratio_spin",
        }
        labels = {
            "bright_via_min_final_score": ("Минимальная итоговая оценка", "Minimum final score"),
            "heuristic_min_circularity": ("Минимальная округлость", "Minimum circularity"),
            "heuristic_min_compactness": ("Минимальная компактность", "Minimum compactness"),
            "heuristic_min_center_contrast": ("Минимальный контраст центра", "Minimum center contrast"),
            "heuristic_min_peak_prominence": ("Минимальная выраженность пика", "Minimum peak prominence"),
            "heuristic_min_edge_sharpness": ("Минимальная резкость края", "Minimum edge sharpness"),
            "heuristic_max_elongation": ("Максимальная вытянутость", "Maximum elongation"),
            "heuristic_max_line_coherence": ("Максимальная направленность границ", "Maximum line coherence"),
            "heuristic_min_center_brightness": ("Минимальная яркость центра", "Minimum center brightness"),
            "heuristic_max_center_drift_ratio": ("Допустимое смещение центра", "Maximum center drift"),
        }
        excluded_fields = {
            "bright_via_min_final_score",
            "via_white_range_enabled",
            "via_white_range_min",
            "via_white_range_max",
            "via_black_range_enabled",
            "via_black_range_min",
            "via_black_range_max",
        }
        changes_by_field = {}
        for adjustment in adjustments:
            for change in getattr(adjustment, "changes", ()):
                if change.field in excluded_fields:
                    continue
                changes_by_field[change.field] = change
        if not changes_by_field:
            return False
        if any(field.startswith("heuristic_") for field in changes_by_field) and hasattr(
            self, "bright_via_advanced_outer"
        ):
            self.bright_via_advanced_outer.setChecked(True)
        changed_text: list[str] = []
        for field, change in changes_by_field.items():
            widget = getattr(self, widget_by_field[field])
            blocker = QSignalBlocker(widget)
            try:
                if isinstance(widget, QSpinBox):
                    widget.setValue(round(float(change.new_value)))
                else:
                    widget.setValue(float(change.new_value))
            finally:
                del blocker
            self._highlight_contact_feedback_widget(widget)
            ru_label, en_label = labels[field]
            label = ru_label if getattr(self, "_ui_language", "en") == "ru" else en_label
            changed_text.append(
                f"{label}: {self._format_contact_feedback_value(change.old_value)} → "
                f"{self._format_contact_feedback_value(widget.value())}"
            )
        self._store_active_extraction_profile_settings()
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_debug_candidates([])
        self._append_log(
            self._contact_feedback_text(
                "Автоподбор изменил параметры: ",
                "Contact feedback changed parameters: ",
            )
            + "; ".join(changed_text)
        )
        self._restart_after_contact_feedback()
        return True
