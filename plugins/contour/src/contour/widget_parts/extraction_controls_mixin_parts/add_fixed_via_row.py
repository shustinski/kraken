"""extraction_controls_mixin: add fixed via row."""
from __future__ import annotations
from contour.widget_parts._imports import (
    EXTRACTION_HELP_TEXTS,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
    _localized_text,
)


class AddFixedViaRowMixin:
    def _add_fixed_via_row(self, *_args, width: int = 1, height: int = 1) -> None:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        via_label = QLabel("")
        via_label.setMinimumWidth(44)
        width_spin = QSpinBox()
        width_spin.setRange(1, 100_000)
        width_spin.setValue(max(1, int(width)))
        width_spin.setPrefix("X ")
        height_spin = QSpinBox()
        height_spin.setRange(1, 100_000)
        height_spin.setValue(max(1, int(height)))
        height_spin.setPrefix("Y ")
        remove_button = QPushButton("-")
        remove_button.setFixedWidth(36)
        remove_button.setMinimumHeight(30)
        remove_button.setStyleSheet(
            "QPushButton { background-color: #d64545; color: white; font-size: 18px; font-weight: 700; border-radius: 6px; }"
            "QPushButton:hover { background-color: #bf3838; }"
            "QPushButton:pressed { background-color: #a93030; }"
        )

        width_spin.valueChanged.connect(self._on_extraction_settings_changed)
        height_spin.valueChanged.connect(self._on_extraction_settings_changed)
        remove_button.clicked.connect(lambda _checked=False, widget=row_widget: self._delete_fixed_via_row(widget))

        self._fixed_via_rows.append(
            {
                "widget": row_widget,
                "label": via_label,
                "width_spin": width_spin,
                "height_spin": height_spin,
                "remove_button": remove_button,
            }
        )

        row_layout.addWidget(via_label)
        row_layout.addWidget(width_spin, 1)
        row_layout.addWidget(height_spin, 1)
        row_layout.addWidget(remove_button)
        self.fixed_via_rows_layout.addWidget(row_widget)
        self._renumber_fixed_via_rows()

        width_spin.setToolTip(_localized_text(EXTRACTION_HELP_TEXTS, "fixed_via_widths", self._ui_language))
        height_spin.setToolTip(_localized_text(EXTRACTION_HELP_TEXTS, "fixed_via_heights", self._ui_language))
        remove_button.setToolTip(
            "Удаляет эту строку с допустимым размером контакта из списка."
            if self._ui_language == "ru"
            else "Removes this allowed contact-size row from the list."
        )

        if not self._suspend_fixed_via_updates:
            self._on_extraction_settings_changed()
