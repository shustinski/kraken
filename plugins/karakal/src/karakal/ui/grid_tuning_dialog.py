"""Slider window for cell-defect thresholds. Preview stays on the open detail frame."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QDialog, QFormLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from .ui_constants import GRID_INSPECTION_PRESET_VALUES, GRID_INSPECTION_TUNING_KEYS


class GridTuningDialog(QDialog):
    tuningChanged = pyqtSignal(dict)
    tuningConfirmed = pyqtSignal(dict)
    fitRequested = pyqtSignal()
    fitMarkupRequested = pyqtSignal()
    chooseMarkupRequested = pyqtSignal()

    def __init__(self, translator, values: dict[str, int], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(translator("grid_tuning.window"))
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self._sliders: dict[str, QSlider] = {}
        self._value_labels: dict[str, QLabel] = {}
        self._emit_timer = QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.setInterval(30)
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
        self._comparison_note = QLabel("", self)
        self._comparison_note.setWordWrap(True)
        self._comparison_note.hide()
        form.addRow(self._comparison_note)
        root.addLayout(form)
        note = QLabel(translator("grid_tuning.preview_hint"), self)
        note.setWordWrap(True)
        root.addWidget(note)
        self._summary = QLabel("", self)
        self._summary.setWordWrap(True)
        self._summary.hide()
        root.addWidget(self._summary)
        fit_button = QPushButton(translator("grid_tuning.fit_examples"), self)
        fit_button.clicked.connect(self.fitRequested.emit)
        root.addWidget(fit_button)
        self._markup_note = QLabel(translator("grid_tuning.no_markup"), self)
        self._markup_note.setWordWrap(True)
        root.addWidget(self._markup_note)
        choose_markup = QPushButton(translator("grid_tuning.choose_markup"), self)
        choose_markup.clicked.connect(self.chooseMarkupRequested.emit)
        root.addWidget(choose_markup)
        self._markup_button = QPushButton(translator("grid_tuning.fit_markup"), self)
        self._markup_button.setEnabled(False)
        self._markup_button.clicked.connect(self.fitMarkupRequested.emit)
        root.addWidget(self._markup_button)
        confirm_button = QPushButton(translator("grid_tuning.confirm"), self)
        confirm_button.clicked.connect(self._confirm)
        root.addWidget(confirm_button)
        close_button = QPushButton(translator("grid_tuning.close"), self)
        close_button.clicked.connect(self.close)
        root.addWidget(close_button)
        self.resize(460, 320)

    def values(self) -> dict[str, int]:
        return {key: int(slider.value()) for key, slider in self._sliders.items()}

    def set_summary(self, text: str) -> None:
        self._summary.setText(text)
        self._summary.setVisible(bool(text))

    def set_markup_available(self, available: bool, note: str = "") -> None:
        self._markup_button.setEnabled(bool(available))
        self._markup_note.setText(note)
        self._markup_note.setVisible(not available)

    def set_comparison_available(self, available: bool, note: str = "") -> None:
        for key in ("mismatch_sensitivity", "disagreement_sensitivity"):
            slider = self._sliders.get(key)
            if slider is not None:
                slider.setEnabled(available)
        self._comparison_note.setText(note)
        self._comparison_note.setVisible(not available and bool(note))

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

    def _confirm(self) -> None:
        self.tuningConfirmed.emit(self.values())
