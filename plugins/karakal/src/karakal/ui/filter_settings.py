"""Pieces of the defect filter panel: a slider beside a number box, a row of choice buttons, and the
indented block of a filter's own settings that sits right under the filter.

The number boxes stay the source of truth (the presenter reads and sets them); a slider only moves
them. While a slider is dragged the box shows the value and the change is sent once, on release,
so a drag does not start a recount for every step.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import QEvent, QObject, QSignalBlocker, Qt
from PyQt6.QtWidgets import QAbstractSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

SLIDER_STEPS = 1000


class SpinSlider(QWidget):
    """A slider driving ``spin`` (QSpinBox or QDoubleSpinBox), put beside it. ``log``: the slider
    runs on a logarithmic scale, so small values of a wide range are easy to pick."""

    def __init__(self, spin: QAbstractSpinBox, *, log: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spin = spin
        self._log = bool(log) and float(spin.minimum()) >= 0.0
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, SLIDER_STEPS)
        self.slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        spin.setParent(self)
        spin.setMinimumWidth(78)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(spin)
        self._sync_slider(spin.value())
        self.slider.valueChanged.connect(self._slider_moved)
        self.slider.sliderReleased.connect(self._slider_released)
        spin.valueChanged.connect(self._sync_slider)
        # A number box switched off (e.g. «auto») switches its slider off too.
        spin.installEventFilter(self)
        self.slider.setEnabled(spin.isEnabled())

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.spin and event.type() == QEvent.Type.EnabledChange:
            self.slider.setEnabled(self.spin.isEnabled())
        return super().eventFilter(watched, event)

    # Scale ---------------------------------------------------------------------------------------

    def _to_position(self, value: float) -> int:
        low, high = float(self.spin.minimum()), float(self.spin.maximum())
        if high <= low:
            return 0
        if self._log:
            share = (math.log1p(float(value) - low)) / max(1e-9, math.log1p(high - low))
        else:
            share = (float(value) - low) / (high - low)
        return int(round(max(0.0, min(1.0, share)) * SLIDER_STEPS))

    def _to_value(self, position: int) -> float:
        low, high = float(self.spin.minimum()), float(self.spin.maximum())
        share = max(0.0, min(1.0, position / SLIDER_STEPS))
        value = low + math.expm1(share * math.log1p(high - low)) if self._log else low + share * (high - low)
        step = float(self.spin.singleStep()) or 1.0
        return round(round((value - low) / step) * step + low, 6)

    def value(self) -> float:
        return float(self.spin.value())

    # Signals -------------------------------------------------------------------------------------

    def _sync_slider(self, value) -> None:
        if self.slider.isSliderDown():
            return
        blocker = QSignalBlocker(self.slider)
        self.slider.setValue(self._to_position(float(value)))
        del blocker

    def _set_spin(self, value: float, *, emit: bool) -> None:
        cast = int(round(value)) if not hasattr(self.spin, "decimals") else float(value)
        if emit:
            self.spin.setValue(cast)
            return
        blocker = QSignalBlocker(self.spin)
        self.spin.setValue(cast)
        del blocker

    def _slider_moved(self, position: int) -> None:
        value = self._to_value(position)
        # Dragged: the number follows; the change goes out once, on release.
        self._set_spin(value, emit=not self.slider.isSliderDown())

    def _slider_released(self) -> None:
        value = self.spin.value()
        self.spin.valueChanged.emit(value)


class ChoiceButtons(QWidget):
    """A row of buttons for a few whole values, driving ``spin`` (hidden: it keeps the value)."""

    def __init__(self, spin: QAbstractSpinBox, values, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spin = spin
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.buttons: dict[int, QPushButton] = {}
        for value in values:
            button = QPushButton(str(value), self)
            button.setCheckable(True)
            button.setMinimumWidth(34)
            button.clicked.connect(lambda _checked=False, value=int(value): self.spin.setValue(value))
            layout.addWidget(button)
            self.buttons[int(value)] = button
        layout.addStretch(1)
        spin.setParent(self)
        spin.setVisible(False)
        spin.valueChanged.connect(self._sync)
        self._sync(spin.value())

    def _sync(self, value) -> None:
        for key, button in self.buttons.items():
            button.setChecked(key == int(value))


class FilterSettingsSection(QFrame):
    """The settings of one filter, indented under it with a line on the left."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("filterSettingsSection")
        self.setStyleSheet(
            "QFrame#filterSettingsSection { border: none; border-left: 2px solid #3a414c; margin-left: 10px; }"
        )
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 2, 0, 6)
        self._layout.setSpacing(4)
        self.labels: list[tuple[QLabel, str]] = []

    def add_row(self, label: QLabel | None, control: QWidget) -> None:
        row = QWidget(self)
        grid = QGridLayout(row)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        if label is not None:
            label.setParent(row)
            label.setStyleSheet("color: #9aa5b4;")
            grid.addWidget(label, 0, 0)
        control.setParent(row)
        grid.addWidget(control, 1 if label is not None else 0, 0)
        self._layout.addWidget(row)

    def add_widget(self, widget: QWidget) -> None:
        widget.setParent(self)
        self._layout.addWidget(widget)
