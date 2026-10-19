"""Dirty regions in the cell check settings: on/off, merge distance, sensitivity, a run button and
the run's summary."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.dirt_regions import DirtConfig

Translate = Callable[..., str]


class DirtControls(QWidget):
    runRequested = pyqtSignal()
    cancelRequested = pyqtSignal()
    settingsChanged = pyqtSignal()

    def __init__(self, translate: Translate, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._t = translate
        defaults = DirtConfig()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)
        self.title = QLabel(self)
        self.title.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.title)
        self.enabled_check = QCheckBox(self)
        self.enabled_check.setChecked(True)
        layout.addWidget(self.enabled_check)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self.merge_gap_spin = QDoubleSpinBox(self)
        self.merge_gap_spin.setRange(0.0, 10.0)
        self.merge_gap_spin.setSingleStep(0.5)
        self.merge_gap_spin.setDecimals(1)
        self.merge_gap_spin.setValue(defaults.merge_gap_cells)
        self.sensitivity_spin = QDoubleSpinBox(self)
        self.sensitivity_spin.setRange(0.5, 3.0)
        self.sensitivity_spin.setSingleStep(0.1)
        self.sensitivity_spin.setDecimals(1)
        self.sensitivity_spin.setValue(defaults.sensitivity)
        self.merge_gap_label = QLabel(self)
        self.sensitivity_label = QLabel(self)
        form.addRow(self.merge_gap_label, self.merge_gap_spin)
        form.addRow(self.sensitivity_label, self.sensitivity_spin)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.run_button = QPushButton(self)
        self.cancel_button = QPushButton(self)
        self.cancel_button.setVisible(False)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.status = QLabel(self)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.run_button.clicked.connect(self.runRequested)
        self.cancel_button.clicked.connect(self.cancelRequested)
        self.merge_gap_spin.valueChanged.connect(lambda _value: self.settingsChanged.emit())
        self.sensitivity_spin.valueChanged.connect(lambda _value: self.settingsChanged.emit())
        self.enabled_check.toggled.connect(lambda _checked: self.settingsChanged.emit())
        self._status_key = "dirt.summary_empty"
        self._status_args: dict = {}
        self.retranslate(translate)

    def retranslate(self, translate: Translate) -> None:
        self._t = translate
        self.title.setText(self._t("dirt.title"))
        self.enabled_check.setText(self._t("dirt.enabled"))
        self.enabled_check.setToolTip(self._t("dirt.enabled_hint"))
        self.merge_gap_label.setText(self._t("dirt.merge_gap"))
        self.merge_gap_label.setToolTip(self._t("dirt.merge_gap_hint"))
        self.merge_gap_spin.setToolTip(self._t("dirt.merge_gap_hint"))
        self.sensitivity_label.setText(self._t("dirt.sensitivity"))
        self.sensitivity_label.setToolTip(self._t("dirt.sensitivity_hint"))
        self.sensitivity_spin.setToolTip(self._t("dirt.sensitivity_hint"))
        self.run_button.setText(self._t("dirt.run"))
        self.cancel_button.setText(self._t("dirt.cancel"))
        self.status.setText(self._t(self._status_key, **self._status_args))

    def enabled(self) -> bool:
        return bool(self.enabled_check.isChecked())

    def config(self) -> DirtConfig:
        return DirtConfig(
            merge_gap_cells=float(self.merge_gap_spin.value()),
            sensitivity=float(self.sensitivity_spin.value()),
        )

    def settings_payload(self) -> dict:
        return {
            "enabled": self.enabled(),
            "merge_gap_cells": float(self.merge_gap_spin.value()),
            "sensitivity": float(self.sensitivity_spin.value()),
        }

    def apply_settings(self, payload: dict | None) -> None:
        values = dict(payload or {})
        widgets = (self.enabled_check, self.merge_gap_spin, self.sensitivity_spin)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            if "enabled" in values:
                self.enabled_check.setChecked(bool(values["enabled"]))
            for key, spin in (("merge_gap_cells", self.merge_gap_spin), ("sensitivity", self.sensitivity_spin)):
                try:
                    if values.get(key) is not None:
                        spin.setValue(float(values[key]))
                except (TypeError, ValueError):
                    pass
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.cancel_button.setVisible(bool(running))

    def set_status(self, key: str, **args) -> None:
        self._status_key = str(key)
        self._status_args = dict(args)
        self.status.setText(self._t(self._status_key, **self._status_args))
