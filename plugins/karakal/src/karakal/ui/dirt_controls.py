"""Dirty regions in the cell check settings: on/off, merge distance, sensitivity, a run button,
the run's summary, a matrix filter by dirt level and the list of dirty frames."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.dirt_regions import LEVEL_HEAVY, LEVEL_LIGHT, LEVEL_MEDIUM, DirtConfig

# The level a frame's dirt must reach to be shown and listed.
FILTER_LEVELS = (LEVEL_LIGHT, LEVEL_MEDIUM, LEVEL_HEAVY)

Translate = Callable[..., str]


class DirtControls(QWidget):
    runRequested = pyqtSignal()
    cancelRequested = pyqtSignal()
    settingsChanged = pyqtSignal()
    filterChanged = pyqtSignal()
    frameChosen = pyqtSignal(str)
    frameOpened = pyqtSignal(str)

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
        filter_row = QHBoxLayout()
        self.filter_check = QCheckBox(self)
        self.level_combo = QComboBox(self)
        for level in FILTER_LEVELS:
            self.level_combo.addItem("", level)
        filter_row.addWidget(self.filter_check, 1)
        filter_row.addWidget(self.level_combo)
        layout.addLayout(filter_row)
        self.frames_list = QListWidget(self)
        self.frames_list.setMaximumHeight(170)
        self.frames_list.setVisible(False)
        layout.addWidget(self.frames_list)
        steps = QHBoxLayout()
        self.previous_button = QPushButton(self)
        self.next_button = QPushButton(self)
        steps.addWidget(self.previous_button)
        steps.addWidget(self.next_button)
        steps.addStretch(1)
        layout.addLayout(steps)
        self.previous_button.setVisible(False)
        self.next_button.setVisible(False)
        self.filter_check.toggled.connect(lambda _checked: self.filterChanged.emit())
        self.level_combo.currentIndexChanged.connect(lambda _index: self.filterChanged.emit())
        self.frames_list.itemClicked.connect(self._item_chosen)
        self.frames_list.itemActivated.connect(self._item_opened)
        self.previous_button.clicked.connect(lambda: self.step(-1))
        self.next_button.clicked.connect(lambda: self.step(1))
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
        self.filter_check.setText(self._t("dirt.filter"))
        self.filter_check.setToolTip(self._t("dirt.filter_hint"))
        for index, level in enumerate(FILTER_LEVELS):
            self.level_combo.setItemText(index, self._t(f"dirt.filter_level.{level}"))
        self.previous_button.setText(self._t("dirt.previous"))
        self.next_button.setText(self._t("dirt.next"))
        self.frames_list.setToolTip(self._t("dirt.list_hint"))
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
            "filter_level": self.filter_level(),
        }

    def apply_settings(self, payload: dict | None) -> None:
        values = dict(payload or {})
        widgets = (self.enabled_check, self.merge_gap_spin, self.sensitivity_spin, self.level_combo)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            if "enabled" in values:
                self.enabled_check.setChecked(bool(values["enabled"]))
            index = self.level_combo.findData(str(values.get("filter_level") or ""))
            if index >= 0:
                self.level_combo.setCurrentIndex(index)
            for key, spin in (("merge_gap_cells", self.merge_gap_spin), ("sensitivity", self.sensitivity_spin)):
                try:
                    if values.get(key) is not None:
                        spin.setValue(float(values[key]))
                except (TypeError, ValueError):
                    pass
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def filter_on(self) -> bool:
        return bool(self.filter_check.isChecked())

    def filter_level(self) -> str:
        return str(self.level_combo.currentData() or LEVEL_LIGHT)

    def set_frames(self, rows: list[tuple[str, str]]) -> None:
        """Dirty frames to list: (frame key, text), dirtiest first; the chosen one stays chosen."""

        current = self.current_frame()
        self.frames_list.blockSignals(True)
        try:
            self.frames_list.clear()
            for key, text in rows:
                item = QListWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, str(key))
                self.frames_list.addItem(item)
                if str(key) == current:
                    self.frames_list.setCurrentItem(item)
        finally:
            self.frames_list.blockSignals(False)
        has_rows = bool(rows)
        for widget in (self.frames_list, self.previous_button, self.next_button):
            widget.setVisible(has_rows)

    def current_frame(self) -> str:
        item = self.frames_list.currentItem()
        return str(item.data(Qt.ItemDataRole.UserRole) or "") if item is not None else ""

    def step(self, delta: int) -> None:
        """Choose the previous or next frame of the list (the first one when none is chosen)."""

        count = self.frames_list.count()
        if not count:
            return
        row = self.frames_list.currentRow()
        row = 0 if row < 0 else (row + int(delta)) % count
        self.frames_list.setCurrentRow(row)
        self._item_chosen(self.frames_list.currentItem())

    def _item_chosen(self, item: QListWidgetItem | None) -> None:
        if item is not None:
            self.frameChosen.emit(str(item.data(Qt.ItemDataRole.UserRole) or ""))

    def _item_opened(self, item: QListWidgetItem | None) -> None:
        if item is not None:
            self.frameOpened.emit(str(item.data(Qt.ItemDataRole.UserRole) or ""))

    def set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.cancel_button.setVisible(bool(running))

    def set_status(self, key: str, **args) -> None:
        self._status_key = str(key)
        self._status_args = dict(args)
        self.status.setText(self._t(self._status_key, **self._status_args))
