"""extraction_controls_mixin: update extraction profile controls."""
from __future__ import annotations


class UpdateExtractionProfileControlsMixin:
    def _update_extraction_profile_controls_state(self) -> None:
        rec = (
            str(self.recognition_mode_combo.currentData() or "conductors")
            if hasattr(self, "recognition_mode_combo")
            else "conductors"
        )
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_contact_recognition_mode(rec == "via")
            self.polygon_editor.set_conductor_recognition_mode(rec == "conductors")
        is_via_profile = self._active_extraction_profile == "vias"
        advanced = self._advanced_extraction_enabled()
        if hasattr(self, "advanced_extraction_checkbox"):
            self.advanced_extraction_checkbox.hide()
        self.basic_filters_group.hide()
        self.geometry_filters_group.hide()
        self.topology_group.hide()
        self.conductor_group.setEnabled(False)
        self.conductor_group.hide()
        self.via_group.setEnabled(False)
        self.via_group.hide()
        advanced_via_widgets = [
            (self.via_range_checkboxes_label_widget, self.via_range_checkboxes_widget),
            (self.via_min_score_label_widget, self.via_min_score_spin),
            (self.via_min_contrast_label_widget, self.via_min_contrast_spin),
            (self.via_min_edge_coverage_label_widget, self.via_min_edge_coverage_spin),
            (self.via_spot_line_suppression_label_widget, self.via_spot_line_suppression_spin),
            (self.via_roundness_label_widget, self.via_roundness_spin),
        ]
        in_via_extraction = rec in ("via", "disabled")
        if hasattr(self, "contour_group"):
            self.contour_group.hide()
        for label_widget, field_widget in advanced_via_widgets:
            if label_widget is not None:
                label_widget.setVisible(advanced and is_via_profile and in_via_extraction and rec == "disabled")
            field_widget.setVisible(advanced and is_via_profile and in_via_extraction and rec == "disabled")
        if hasattr(self, "bright_via_group"):
            self.bright_via_group.setVisible(is_via_profile and rec == "via")
        if hasattr(self, "metal_display_group"):
            self.metal_display_group.setEnabled(rec == "conductors")
        self._sync_recognition_stack_visibility()
        self._update_via_threshold_controls_state()
        self._update_via_brightness_range_controls_state()

    def _sync_recognition_stack_visibility(self) -> None:
        if not hasattr(self, "recognition_mode_combo") or not hasattr(self, "recognition_stack"):
            return
        data = str(self.recognition_mode_combo.currentData() or "conductors")
        if data == "via":
            self.recognition_stack.setVisible(False)
        else:
            self.recognition_stack.setVisible(True)
            self.recognition_stack.setCurrentIndex(0 if data == "disabled" else 1)

    def _advanced_extraction_enabled(self) -> bool:
        return bool(hasattr(self, "advanced_extraction_checkbox") and self.advanced_extraction_checkbox.isChecked())

    def _on_advanced_extraction_toggled(self, *_args) -> None:
        self._update_extraction_profile_controls_state()
