"""matrix_view: gradient range selector widget."""
from __future__ import annotations
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from neuralimage.Validation_gradient_widget_lite.ui.i18n import Translator
from neuralimage.Validation_gradient_widget_lite.ui.ui_constants import (
    CARD_CONTENT_SPACING,
    DEFAULT_ERROR_WINDOW,
    DEFAULT_GRADIENT_NAME,
    GRADIENT_PRESETS,
    GRADIENT_RANGE_SELECTOR_MIN_HEIGHT,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.gradient_window_bar import _GradientWindowBar

class GradientRangeSelectorWidget(QWidget):
    """Edit the low and high bounds of the active error gradient window."""

    rangeChanged = pyqtSignal(float, float)

    def __init__(self, parent=None) -> None:
        """Initialize the error-window selector."""
        super().__init__(parent)
        self._gradient_name = DEFAULT_GRADIENT_NAME
        self._i18n = Translator()
        self._t = self._i18n.tr
        self._low_spin = QDoubleSpinBox(self)
        self._high_spin = QDoubleSpinBox(self)
        self._bar = _GradientWindowBar(self)
        for spin, value in ((self._low_spin, DEFAULT_ERROR_WINDOW[0]), (self._high_spin, DEFAULT_ERROR_WINDOW[1])):
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.01)
            spin.setDecimals(2)
            spin.setValue(value)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(CARD_CONTENT_SPACING)
        self._title_label = QLabel(self._t("matrix.error_window"), self)
        layout.addWidget(self._title_label)
        layout.addWidget(self._bar)
        row = QHBoxLayout()
        self._low_label = QLabel(self._t("matrix.low"), self)
        row.addWidget(self._low_label)
        row.addWidget(self._low_spin)
        self._high_label = QLabel(self._t("matrix.high"), self)
        row.addWidget(self._high_label)
        row.addWidget(self._high_spin)
        layout.addLayout(row)
        self.setMinimumHeight(GRADIENT_RANGE_SELECTOR_MIN_HEIGHT)
        self._low_spin.valueChanged.connect(self._emit_range_changed)
        self._high_spin.valueChanged.connect(self._emit_range_changed)
        self._bar.rangeEdited.connect(self._on_bar_range_edited)
        self._bar.set_gradient_name(self._gradient_name)
        self._bar.set_error_window(*DEFAULT_ERROR_WINDOW)

    def error_window(self) -> tuple[float, float]:
        """Return the currently selected error window."""
        return float(self._low_spin.value()), float(self._high_spin.value())

    def set_error_window(self, low_bound: float, high_bound: float) -> None:
        """Set the error window without emitting change signals."""
        low_value = max(0.0, min(float(low_bound), 1.0))
        high_value = max(0.0, min(float(high_bound), 1.0))
        self._low_spin.blockSignals(True)
        self._high_spin.blockSignals(True)
        self._low_spin.setValue(low_value)
        self._high_spin.setValue(high_value)
        self._low_spin.blockSignals(False)
        self._high_spin.blockSignals(False)
        self._bar.set_error_window(low_value, high_value)

    def set_gradient_name(self, gradient_name: str) -> None:
        """Set the gradient used by the error-window preview bar."""
        self._gradient_name = gradient_name if gradient_name in GRADIENT_PRESETS else DEFAULT_GRADIENT_NAME
        self._bar.set_gradient_name(self._gradient_name)

    def _emit_range_changed(self, _value: float) -> None:
        """Propagate range changes coming from the numeric editors."""
        low_value, high_value = self.error_window()
        self._bar.set_error_window(low_value, high_value)
        self.rangeChanged.emit(low_value, high_value)

    def _on_bar_range_edited(self, low_bound: float, high_bound: float) -> None:
        """Apply range changes coming from the draggable gradient bar."""
        self.set_error_window(low_bound, high_bound)
        self.rangeChanged.emit(*self.error_window())

    def retranslate_ui(self) -> None:
        """Update translated captions inside the range selector."""
        self._title_label.setText(self._t("matrix.error_window"))
        self._low_label.setText(self._t("matrix.low"))
        self._high_label.setText(self._t("matrix.high"))
