"""extraction_controls_mixin: delete fixed via row."""
from __future__ import annotations
from contour.widget_parts._imports import (
    QLabel,
    QSpinBox,
    QWidget,
)


class DeleteFixedViaRowMixin:
    def _renumber_fixed_via_rows(self) -> None:
        for index, row in enumerate(self._fixed_via_rows, start=1):
            label = row["label"]
            if isinstance(label, QLabel):
                label.setText(f"Контакт {index}" if self._ui_language == "ru" else f"Contact {index}")

    def _clear_fixed_via_rows(self) -> None:
        while self._fixed_via_rows:
            row = self._fixed_via_rows.pop()
            widget = row["widget"]
            if isinstance(widget, QWidget):
                self.fixed_via_rows_layout.removeWidget(widget)
                widget.deleteLater()

    def _fixed_via_pairs(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for row in self._fixed_via_rows:
            width_spin = row["width_spin"]
            height_spin = row["height_spin"]
            if isinstance(width_spin, QSpinBox) and isinstance(height_spin, QSpinBox):
                pairs.append((int(width_spin.value()), int(height_spin.value())))
        return pairs

    def _delete_fixed_via_row(self, row_widget: QWidget) -> None:
        for index, row in enumerate(self._fixed_via_rows):
            if row["widget"] is row_widget:
                self._fixed_via_rows.pop(index)
                self.fixed_via_rows_layout.removeWidget(row_widget)
                row_widget.deleteLater()
                self._renumber_fixed_via_rows()
                if not self._suspend_fixed_via_updates:
                    self._on_extraction_settings_changed()
                return
