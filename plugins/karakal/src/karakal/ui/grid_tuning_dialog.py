"""Slider window for cell-defect thresholds. Preview stays on the open detail frame."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QDialog, QFormLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from .ui_constants import GRID_INSPECTION_PRESET_VALUES, GRID_INSPECTION_TUNING_KEYS


class GridTuningDialog(QDialog):
    tuningChanged = pyqtSignal(dict)

    def __init__(self, translator, values: dict[str, int], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(translator("grid_tuning.window"))
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self._sliders: dict[str, QSlider] = {}
        self._value_labels: dict[str, QLabel] = {}
        self._emit_timer = QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.setInterval(200)
        self._emit_timer.timeout.connect(self._emit_values)
        root = QVBoxLayout(self)
        form = QFormLayout()
        for key in GRID_INSPECTION_TUNING_KEYS:
            slider = QSlider(Qt.Orientation.Horizontal, self)
            slider.setRange(0, 100)
            slider.setValue(int(values.get(key, GRID_INSPECTION_PRESET_VALUES["balanced"][key])))
            value_label = QLabel(str(slider.value()), self)
            row = QWidget(self)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(slider, stretch=1)
            row_layout.addWidget(value_label)
            slider.valueChanged.connect(lambda value, label=value_label: label.setText(str(int(value))))
            slider.valueChanged.connect(self._schedule_emit)
            self._sliders[key] = slider
            self._value_labels[key] = value_label
            form.addRow(translator(f"grid_tuning.{key}"), row)
        root.addLayout(form)
        note = QLabel(translator("grid_tuning.preview_hint"), self)
        note.setWordWrap(True)
        root.addWidget(note)
        close_button = QPushButton(translator("grid_tuning.close"), self)
        close_button.clicked.connect(self.close)
        root.addWidget(close_button)
        self.resize(460, 320)

    def values(self) -> dict[str, int]:
        return {key: int(slider.value()) for key, slider in self._sliders.items()}

    def set_values(self, values: dict[str, int]) -> None:
        for key, slider in self._sliders.items():
            if key not in values:
                continue
            blocker = slider.blockSignals(True)
            slider.setValue(int(values[key]))
            slider.blockSignals(blocker)
            label = self._value_labels.get(key)
            if label is not None:
                label.setText(str(int(slider.value())))

    def _schedule_emit(self, *_args) -> None:
        self._emit_timer.start()

    def _emit_values(self) -> None:
        self.tuningChanged.emit(self.values())
