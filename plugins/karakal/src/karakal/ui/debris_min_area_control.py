"""Minimum debris size in pixels: a slider to pick by eye and a box to type a known value."""

from __future__ import annotations

from PyQt6.QtCore import QSignalBlocker, Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QSlider, QSpinBox, QWidget

from ..core.grid_anomaly import DEBRIS_MIN_AREA_MAX_PX, DEBRIS_MIN_AREA_PX


class DebrisMinAreaControl(QWidget):
    valueChanged = pyqtSignal(int)

    def __init__(self, translator, value: int = int(DEBRIS_MIN_AREA_PX), parent=None) -> None:
        super().__init__(parent)
        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setRange(0, DEBRIS_MIN_AREA_MAX_PX)
        self._spin = QSpinBox(self)
        self._spin.setRange(0, DEBRIS_MIN_AREA_MAX_PX)
        self._spin.setSuffix(" px")
        # Typing "120" must not run an analysis for 1 and 12 on the way.
        self._spin.setKeyboardTracking(False)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._spin)
        self.set_value(value)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._spin.valueChanged.connect(self._on_spin_changed)
        self.retranslate(translator)

    def retranslate(self, translator) -> None:
        hint = translator("grid_tuning.debris_min_area_hint")
        self.setToolTip(hint)
        self._slider.setToolTip(hint)
        self._spin.setToolTip(hint)

    def value(self) -> int:
        return int(self._spin.value())

    def set_value(self, value: int) -> None:
        """Show a value without emitting valueChanged."""

        value = max(0, min(DEBRIS_MIN_AREA_MAX_PX, int(value)))
        slider_blocker = QSignalBlocker(self._slider)
        spin_blocker = QSignalBlocker(self._spin)
        self._slider.setValue(value)
        self._spin.setValue(value)
        del slider_blocker, spin_blocker

    def _on_slider_changed(self, value: int) -> None:
        blocker = QSignalBlocker(self._spin)
        self._spin.setValue(int(value))
        del blocker
        self.valueChanged.emit(int(value))

    def _on_spin_changed(self, value: int) -> None:
        blocker = QSignalBlocker(self._slider)
        self._slider.setValue(int(value))
        del blocker
        self.valueChanged.emit(int(value))
