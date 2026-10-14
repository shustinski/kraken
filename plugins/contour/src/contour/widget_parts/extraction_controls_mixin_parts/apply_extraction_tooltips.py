"""extraction_controls_mixin: apply extraction tooltips."""
from __future__ import annotations
from contour.widget_parts._imports import (
    EXTRACTION_HELP_TEXTS,
    QPushButton,
    QSpinBox,
    _localized_text,
)


class ApplyExtractionTooltipsMixin:
    def _apply_extraction_tooltips(self) -> None:
        self._set_field_tooltip(self.retrieval_mode_label_widget, self.retrieval_mode_combo, "retrieval_mode")
        self._set_field_tooltip(
            self.approximation_mode_label_widget, self.approximation_mode_combo, "approximation_mode"
        )
        self._set_field_tooltip(
            self.epsilon_label_widget,
            self.epsilon_row_widget if hasattr(self, "epsilon_row_widget") else self.epsilon_spin,
            "epsilon",
        )
        self._set_field_tooltip(self.epsilon_mode_label_widget, self.epsilon_relative_checkbox, "epsilon_mode")
        self._set_field_tooltip(self.min_area_label_widget, self.min_area_spin, "min_area")
        self._set_field_tooltip(self.max_area_label_widget, self.max_area_spin, "max_area")
        self._set_field_tooltip(self.min_perimeter_label_widget, self.min_perimeter_spin, "min_perimeter")
        self._set_field_tooltip(self.max_perimeter_label_widget, self.max_perimeter_spin, "max_perimeter")
        self._set_field_tooltip(self.min_point_count_label_widget, self.min_points_spin, "min_points")
        self._set_field_tooltip(self.min_polygon_width_label_widget, self.min_polygon_width_spin, "min_polygon_width")
        self._set_field_tooltip(self.min_bbox_width_label_widget, self.min_bbox_width_spin, "min_bbox_width")
        self._set_field_tooltip(self.max_bbox_width_label_widget, self.max_bbox_width_spin, "max_bbox_width")
        self._set_field_tooltip(self.min_bbox_height_label_widget, self.min_bbox_height_spin, "min_bbox_height")
        self._set_field_tooltip(self.max_bbox_height_label_widget, self.max_bbox_height_spin, "max_bbox_height")
        self._set_field_tooltip(self.min_aspect_ratio_label_widget, self.min_aspect_ratio_spin, "min_aspect_ratio")
        self._set_field_tooltip(self.max_aspect_ratio_label_widget, self.max_aspect_ratio_spin, "max_aspect_ratio")
        self._set_field_tooltip(
            self.border_handling_label_widget, self.exclude_border_touching_checkbox, "exclude_border_touching"
        )
        self._set_field_tooltip(self.min_solidity_label_widget, self.min_solidity_spin, "min_solidity")
        self._set_field_tooltip(self.min_extent_label_widget, self.min_extent_spin, "min_extent")
        if getattr(self, "via_search_mode_label_widget", None) is not None:
            self._set_field_tooltip(self.via_search_mode_label_widget, self.via_search_mode_combo, "via_search_mode")
        if hasattr(self, "bright_via_viamode_label_widget"):
            self._set_field_tooltip(self.bright_via_viamode_label_widget, self.via_search_mode_combo, "via_search_mode")
        if hasattr(self, "bright_via_polarity_label_widget"):
            self._set_field_tooltip(
                self.bright_via_polarity_label_widget,
                self.via_heuristic_polarity_combo,
                "via_polarity",
            )
        self._set_field_tooltip(self.via_white_range_label_widget, self.via_white_range_widget, "via_white_range")
        self._set_field_tooltip(self.via_black_range_label_widget, self.via_black_range_widget, "via_black_range")
        self._set_field_tooltip(self.via_min_score_label_widget, self.via_min_score_spin, "via_min_score")
        self._set_field_tooltip(self.via_min_contrast_label_widget, self.via_min_contrast_spin, "via_min_contrast")
        self._set_field_tooltip(
            self.via_min_edge_coverage_label_widget,
            self.via_min_edge_coverage_spin,
            "via_min_edge_coverage",
        )
        self._set_field_tooltip(
            self.via_spot_line_suppression_label_widget,
            self.via_spot_line_suppression_spin,
            "via_spot_line_suppression",
        )
        self._set_field_tooltip(
            self.via_template_min_score_label_widget, self.via_template_min_score_spin, "via_template_min_score"
        )
        self._set_field_tooltip(self.via_templates_label_widget, self.via_templates_widget, "via_templates")
        self._set_field_tooltip(self.via_preset_label_widget, self.via_preset_widget, "via_preset_selector")
        self._set_field_tooltip(self.reset_via_search_label_widget, self.reset_via_search_button, "reset_via_search")
        for checkbox, tooltip_key in (
            (self.via_white_range_checkbox, "via_white_range"),
            (self.via_black_range_checkbox, "via_black_range"),
        ):
            detector_tooltip = _localized_text(EXTRACTION_HELP_TEXTS, tooltip_key, self._ui_language)
            checkbox.setToolTip(detector_tooltip)
            checkbox.setStatusTip(detector_tooltip)
        self._set_field_tooltip(self.debug_candidates_label_widget, self.debug_candidates_checkbox, "debug_candidates")
        self._set_field_tooltip(self.via_roundness_label_widget, self.via_roundness_spin, "via_min_roundness")
        self._set_field_tooltip(self.min_via_width_label_widget, self.min_via_width_spin, "min_via_width")
        self._set_field_tooltip(self.max_via_width_label_widget, self.max_via_width_spin, "max_via_width")
        self._set_field_tooltip(self.min_via_height_label_widget, self.min_via_height_spin, "min_via_height")
        self._set_field_tooltip(self.max_via_height_label_widget, self.max_via_height_spin, "max_via_height")
        self._set_field_tooltip(self.fixed_vias_label_widget, self.fixed_vias_widget, "fixed_via_widths")
        self.fixed_via_add_button.setToolTip(
            "Добавляет ещё одну допустимую пару ширины и высоты контакта."
            if self._ui_language == "ru"
            else "Adds another allowed contact width and height pair."
        )
        for row in self._fixed_via_rows:
            width_spin = row["width_spin"]
            height_spin = row["height_spin"]
            remove_button = row["remove_button"]
            if isinstance(width_spin, QSpinBox):
                width_spin.setToolTip(_localized_text(EXTRACTION_HELP_TEXTS, "fixed_via_widths", self._ui_language))
            if isinstance(height_spin, QSpinBox):
                height_spin.setToolTip(_localized_text(EXTRACTION_HELP_TEXTS, "fixed_via_heights", self._ui_language))
            if isinstance(remove_button, QPushButton):
                remove_button.setToolTip(
                    "Удаляет эту строку с допустимым размером контакта из списка."
                    if self._ui_language == "ru"
                    else "Removes this allowed contact-size row from the list."
                )
        self._set_field_tooltip(
            self.min_hierarchy_depth_label_widget, self.min_hierarchy_depth_spin, "min_hierarchy_depth"
        )
        self._set_field_tooltip(
            self.min_inner_hole_area_label_widget, self.min_inner_hole_area_spin, "min_inner_hole_area"
        )
        self._set_field_tooltip(
            self.max_hierarchy_depth_label_widget, self.max_hierarchy_depth_spin, "max_hierarchy_depth"
        )
        self._set_field_tooltip(
            self.max_hole_area_ratio_label_widget, self.max_hole_area_ratio_spin, "max_hole_area_ratio"
        )
        self._apply_bright_via_tooltips()
