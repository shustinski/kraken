"""extraction_controls_mixin: update via threshold controls."""
from __future__ import annotations
from contour.widget_parts._imports import (
    VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG,
    VIA_SEARCH_MODE_HEURISTIC,
    VIA_SEARCH_MODE_HYBRID,
    VIA_SEARCH_MODE_TEMPLATE,
    normalize_via_search_mode,
)


class UpdateViaThresholdControlsMixin:
    def _update_via_threshold_controls_state(self) -> None:
        mode = normalize_via_search_mode(self.via_search_mode_combo.currentData())
        advanced = self._advanced_extraction_enabled()
        _unused_bright_enabled = mode == VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG
        heuristic_mode = mode in (VIA_SEARCH_MODE_HEURISTIC, VIA_SEARCH_MODE_HYBRID)
        blob_enabled = False
        template_enabled = mode in (VIA_SEARCH_MODE_TEMPLATE, VIA_SEARCH_MODE_HYBRID)
        in_via_recognition = (
            hasattr(self, "recognition_mode_combo")
            and str(self.recognition_mode_combo.currentData() or "") == "via"
        )
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_ctrl_image_region_selection_enabled(
                in_via_recognition and template_enabled
            )
        for label_widget, field_widget in (
            (self.via_min_score_label_widget, self.via_min_score_spin),
            (self.via_min_contrast_label_widget, self.via_min_contrast_spin),
            (self.via_min_edge_coverage_label_widget, self.via_min_edge_coverage_spin),
            (self.via_spot_line_suppression_label_widget, self.via_spot_line_suppression_spin),
        ):
            if label_widget is not None:
                label_widget.setVisible(advanced and blob_enabled)
            field_widget.setVisible(advanced and blob_enabled)
        if self.via_template_min_score_label_widget is not None:
            self.via_template_min_score_label_widget.setVisible(advanced and template_enabled)
        self.via_template_min_score_spin.setVisible(advanced and template_enabled)
        if self.via_templates_label_widget is not None:
            self.via_templates_label_widget.setVisible(template_enabled)
        self.via_templates_widget.setVisible(template_enabled)
        if hasattr(self, "via_range_checkboxes_label_widget") and self.via_range_checkboxes_label_widget is not None:
            self.via_range_checkboxes_label_widget.setVisible(False)
        if hasattr(self, "via_range_checkboxes_widget"):
            self.via_range_checkboxes_widget.setVisible(False)

        self._update_via_brightness_range_controls_state()
        if hasattr(self, "bright_via_group") and hasattr(self, "recognition_mode_combo"):
            self.bright_via_group.setVisible(
                self._active_extraction_profile == "vias"
                and str(self.recognition_mode_combo.currentData() or "") == "via"
            )
        template_only = mode == VIA_SEARCH_MODE_TEMPLATE
        if hasattr(self, "bright_via_basics_form"):
            for row in range(self.bright_via_basics_form.rowCount()):
                self.bright_via_basics_form.setRowVisible(row, row < 2 or not template_only)
            if hasattr(self, "bright_via_mode_stack"):
                self.bright_via_basics_form.setRowVisible(self.bright_via_mode_stack, template_enabled)
        # The generic row visibility above must not re-open the diameter range
        # that is irrelevant in fixed-size mode.
        if not template_only:
            self._update_bright_via_diameter_controls_state()
        if hasattr(self, "via_heuristic_polarity_combo"):
            self.via_heuristic_polarity_combo.setVisible(heuristic_mode)
        if (label_widget := getattr(self, "bright_via_polarity_label_widget", None)) is not None:
            label_widget.setVisible(heuristic_mode)
        if (label_widget := getattr(self, "bright_via_mode_stack_label_widget", None)) is not None:
            label_widget.setVisible(not template_only)
        sem_mode = mode == VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG
        if hasattr(self, "bright_via_quality_group"):
            self.bright_via_quality_group.setVisible(sem_mode and not template_only)
        if hasattr(self, "bright_via_display_group"):
            self.bright_via_display_group.setVisible(not template_only)
        if hasattr(self, "bright_via_advanced_outer"):
            self.bright_via_advanced_outer.setVisible(heuristic_mode and not template_only)
        if sem_mode:
            self._update_bright_via_diameter_controls_state()

    def _update_via_brightness_range_controls_state(self) -> None:
        if not hasattr(self, "via_white_range_checkbox"):
            return

        white_checked = self.via_white_range_checkbox.isChecked()
        self.via_white_range_min_spin.setEnabled(white_checked)
        self.via_white_range_max_spin.setEnabled(white_checked)
        black_checked = self.via_black_range_checkbox.isChecked()
        self.via_black_range_min_spin.setEnabled(black_checked)
        self.via_black_range_max_spin.setEnabled(black_checked)

        in_via_recognition = (
            hasattr(self, "recognition_mode_combo") and str(self.recognition_mode_combo.currentData() or "") == "via"
        )
        show_in_basics = in_via_recognition
        self.via_white_range_checkbox.setVisible(show_in_basics)
        self.via_white_range_widget.setVisible(show_in_basics)
        self.via_black_range_checkbox.setVisible(show_in_basics)
        self.via_black_range_widget.setVisible(show_in_basics)
        if (label_widget := getattr(self, "bright_via_white_range_label_widget", None)) is not None:
            label_widget.setVisible(show_in_basics)
        if (label_widget := getattr(self, "bright_via_black_range_label_widget", None)) is not None:
            label_widget.setVisible(show_in_basics)

        # Legacy via panel duplicates (recognition disabled).
        legacy_visible = (
            not in_via_recognition and self._advanced_extraction_enabled() and self._active_extraction_profile == "vias"
        )
        if self.via_white_range_label_widget is not None:
            self.via_white_range_label_widget.setVisible(legacy_visible and white_checked)
        if self.via_black_range_label_widget is not None:
            self.via_black_range_label_widget.setVisible(legacy_visible and black_checked)
