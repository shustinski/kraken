"""extraction_controls_mixin: update via size controls."""
from __future__ import annotations


class UpdateViaSizeControlsMixin:
    def _update_via_size_controls_state(self) -> None:
        range_widgets = [
            (self.min_via_width_label_widget, self.via_width_range_widget),
            (self.min_via_height_label_widget, self.via_height_range_widget),
        ]
        fixed_widgets = [
            (self.fixed_vias_label_widget, self.fixed_vias_widget),
        ]
        for label_widget, field_widget in range_widgets:
            if label_widget is not None:
                label_widget.setVisible(True)
            field_widget.setVisible(True)
        for label_widget, field_widget in fixed_widgets:
            if label_widget is not None:
                label_widget.setVisible(False)
            field_widget.setVisible(False)
        self._update_bright_via_diameter_controls_state()
        self._update_via_threshold_controls_state()

    def _update_bright_via_diameter_controls_state(self) -> None:
        if (label_widget := getattr(self, "bright_via_diameter_range_label_widget", None)) is not None:
            label_widget.setVisible(True)
        if hasattr(self, "bright_via_diameter_range_widget"):
            self.bright_via_diameter_range_widget.setVisible(True)
        if (label_widget := getattr(self, "via_output_diameter_label_widget", None)) is not None:
            label_widget.setVisible(True)
        self.via_output_diameter_spin.setVisible(True)
