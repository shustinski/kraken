"""The «Dirt» tab of the cell check: search and its settings, the matrix filter, the table of dirty
frames (sortable) and a preview of the chosen frame with its regions."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.dirt_regions import LEVEL_HEAVY, LEVEL_LIGHT, LEVEL_MEDIUM, LEVELS, DirtConfig

Translate = Callable[..., str]
FILTER_LEVELS = (LEVEL_LIGHT, LEVEL_MEDIUM, LEVEL_HEAVY)
_MUTED = "color: #8a94a3;"


class _NumberItem(QTableWidgetItem):
    """A cell sorted by its number, not by its text."""

    def __init__(self, text: str, value: float) -> None:
        super().__init__(text)
        self._value = float(value)
        self.setTextAlignment(int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))

    def __lt__(self, other) -> bool:
        if isinstance(other, _NumberItem):
            return self._value < other._value
        return super().__lt__(other)


class _Preview(QLabel):
    """The chosen frame's picture, scaled to the space it has."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #0d1117; border-radius: 6px;" + _MUTED)

    def set_picture(self, pixmap: QPixmap | None, empty_text: str = "") -> None:
        self._pixmap = pixmap if pixmap is not None and not pixmap.isNull() else None
        if self._pixmap is None:
            self.setPixmap(QPixmap())
            self.setText(empty_text)
        else:
            self.setText("")
            self._rescale()

    def picture(self) -> QPixmap | None:
        return self._pixmap

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is not None:
            self.setPixmap(
                self._pixmap.scaled(
                    self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
            )


class DirtTab(QWidget):
    runRequested = pyqtSignal()
    cancelRequested = pyqtSignal()
    settingsChanged = pyqtSignal()
    filterChanged = pyqtSignal()
    frameChosen = pyqtSignal(str)
    frameOpened = pyqtSignal(str)
    showOnMatrixRequested = pyqtSignal(str)
    titleChanged = pyqtSignal(str)

    def __init__(self, translate: Translate, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._t = translate
        defaults = DirtConfig()
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        search = QHBoxLayout()
        self.run_button = QPushButton(self)
        self.run_button.setStyleSheet("font-weight: 700;")
        self.cancel_button = QPushButton(self)
        self.cancel_button.setVisible(False)
        self.status = QLabel(self)
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        search.addWidget(self.run_button)
        search.addWidget(self.cancel_button)
        search.addWidget(self.status, 1)
        root.addLayout(search)

        settings = QHBoxLayout()
        self.enabled_check = QCheckBox(self)
        self.enabled_check.setChecked(True)
        self.merge_gap_label = QLabel(self)
        self.merge_gap_spin = QDoubleSpinBox(self)
        self.merge_gap_spin.setRange(0.0, 10.0)
        self.merge_gap_spin.setSingleStep(0.5)
        self.merge_gap_spin.setDecimals(1)
        self.merge_gap_spin.setValue(defaults.merge_gap_cells)
        self.sensitivity_label = QLabel(self)
        self.sensitivity_spin = QDoubleSpinBox(self)
        self.sensitivity_spin.setRange(0.5, 3.0)
        self.sensitivity_spin.setSingleStep(0.1)
        self.sensitivity_spin.setDecimals(1)
        self.sensitivity_spin.setValue(defaults.sensitivity)
        settings.addWidget(self.enabled_check)
        settings.addSpacing(16)
        settings.addWidget(self.merge_gap_label)
        settings.addWidget(self.merge_gap_spin)
        settings.addSpacing(12)
        settings.addWidget(self.sensitivity_label)
        settings.addWidget(self.sensitivity_spin)
        settings.addStretch(1)
        root.addLayout(settings)

        view = QHBoxLayout()
        self.filter_check = QCheckBox(self)
        self.level_combo = QComboBox(self)
        for level in FILTER_LEVELS:
            self.level_combo.addItem("", level)
        self.previous_button = QPushButton(self)
        self.next_button = QPushButton(self)
        self.show_button = QPushButton(self)
        view.addWidget(self.filter_check)
        view.addWidget(self.level_combo)
        view.addStretch(1)
        view.addWidget(self.previous_button)
        view.addWidget(self.next_button)
        view.addWidget(self.show_button)
        root.addLayout(view)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.frames_table = QTableWidget(0, 4, splitter)
        self.frames_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.frames_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.frames_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.frames_table.verticalHeader().setVisible(False)
        self.frames_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            self.frames_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.frames_table.setSortingEnabled(True)
        preview_box = QWidget(splitter)
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.setContentsMargins(6, 0, 0, 0)
        self.preview = _Preview(preview_box)
        self.preview_caption = QLabel(preview_box)
        self.preview_caption.setWordWrap(True)
        preview_layout.addWidget(self.preview, 1)
        preview_layout.addWidget(self.preview_caption)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, 1)

        self.run_button.clicked.connect(self.runRequested)
        self.cancel_button.clicked.connect(self.cancelRequested)
        self.merge_gap_spin.valueChanged.connect(lambda _value: self.settingsChanged.emit())
        self.sensitivity_spin.valueChanged.connect(lambda _value: self.settingsChanged.emit())
        self.enabled_check.toggled.connect(lambda _checked: self.settingsChanged.emit())
        self.filter_check.toggled.connect(lambda _checked: self.filterChanged.emit())
        self.level_combo.currentIndexChanged.connect(lambda _index: self.filterChanged.emit())
        self.frames_table.itemSelectionChanged.connect(self._selection_changed)
        self.frames_table.itemDoubleClicked.connect(lambda item: self.frameOpened.emit(self._key_of_row(item.row())))
        self.previous_button.clicked.connect(lambda: self.step(-1))
        self.next_button.clicked.connect(lambda: self.step(1))
        self.show_button.clicked.connect(lambda: self.showOnMatrixRequested.emit(self.current_frame()))
        self._status_key = "dirt.summary_empty"
        self._status_args: dict = {}
        self._filling = False
        self.retranslate(translate)
        self.set_frames([])

    # Texts --------------------------------------------------------------------------------------

    def retranslate(self, translate: Translate) -> None:
        self._t = translate
        self.run_button.setText(self._t("dirt.run"))
        self.cancel_button.setText(self._t("dirt.cancel"))
        self.enabled_check.setText(self._t("dirt.with_cell_check"))
        self.enabled_check.setToolTip(self._t("dirt.enabled_hint"))
        self.merge_gap_label.setText(self._t("dirt.merge_gap"))
        for widget in (self.merge_gap_label, self.merge_gap_spin):
            widget.setToolTip(self._t("dirt.merge_gap_hint"))
        self.sensitivity_label.setText(self._t("dirt.sensitivity"))
        for widget in (self.sensitivity_label, self.sensitivity_spin):
            widget.setToolTip(self._t("dirt.sensitivity_hint"))
        self.filter_check.setText(self._t("dirt.filter"))
        self.filter_check.setToolTip(self._t("dirt.filter_hint"))
        for index, level in enumerate(FILTER_LEVELS):
            self.level_combo.setItemText(index, self._t(f"dirt.filter_level.{level}"))
        self.previous_button.setText(self._t("dirt.previous"))
        self.next_button.setText(self._t("dirt.next"))
        self.show_button.setText(self._t("dirt.show_on_matrix"))
        self.show_button.setToolTip(self._t("dirt.show_on_matrix_hint"))
        self.frames_table.setHorizontalHeaderLabels(
            [self._t("dirt.col.frame"), self._t("dirt.col.level"), self._t("dirt.col.dirtiness"), self._t("dirt.col.regions")]
        )
        self.frames_table.setToolTip(self._t("dirt.list_hint"))
        self.status.setText(self._t(self._status_key, **self._status_args))
        if self.preview.picture() is None:
            self.preview.set_picture(None, self._t("dirt.preview_empty"))

    # Settings -----------------------------------------------------------------------------------

    def enabled(self) -> bool:
        return bool(self.enabled_check.isChecked())

    def config(self) -> DirtConfig:
        return DirtConfig(merge_gap_cells=float(self.merge_gap_spin.value()), sensitivity=float(self.sensitivity_spin.value()))

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

    # State --------------------------------------------------------------------------------------

    def set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.cancel_button.setVisible(bool(running))

    def set_status(self, key: str, **args) -> None:
        self._status_key = str(key)
        self._status_args = dict(args)
        self.status.setText(self._t(self._status_key, **self._status_args))

    def set_title(self, text: str) -> None:
        self.titleChanged.emit(str(text))

    # The table ----------------------------------------------------------------------------------

    def set_frames(self, rows: list[dict]) -> None:
        """Dirty frames: dicts with key, level, dirtiness (0..1), regions; the chosen one stays chosen."""

        current = self.current_frame()
        self._filling = True
        table = self.frames_table
        sorting = table.isSortingEnabled()
        table.setSortingEnabled(False)
        try:
            table.setRowCount(len(rows))
            for row, entry in enumerate(rows):
                key = str(entry["key"])
                level = str(entry.get("level") or "")
                name = QTableWidgetItem(key)
                name.setData(Qt.ItemDataRole.UserRole, key)
                rank = LEVELS.index(level) if level in LEVELS else -1
                table.setItem(row, 0, name)
                table.setItem(row, 1, _NumberItem(self._t(f"dirt.level.{level}") if level else "", rank))
                dirtiness = float(entry.get("dirtiness") or 0.0)
                table.setItem(row, 2, _NumberItem(f"{int(round(dirtiness * 100))}%", dirtiness))
                regions = int(entry.get("regions") or 0)
                table.setItem(row, 3, _NumberItem(str(regions), regions))
        finally:
            table.setSortingEnabled(sorting)
            self._filling = False
        if current:
            self.select_frame(current, notify=False)

    def _key_of_row(self, row: int) -> str:
        item = self.frames_table.item(row, 0)
        return str(item.data(Qt.ItemDataRole.UserRole) or "") if item is not None else ""

    def current_frame(self) -> str:
        return self._key_of_row(self.frames_table.currentRow()) if self.frames_table.currentRow() >= 0 else ""

    def select_frame(self, key: str, *, notify: bool = True) -> bool:
        for row in range(self.frames_table.rowCount()):
            if self._key_of_row(row) == str(key):
                self._filling = not notify
                try:
                    self.frames_table.selectRow(row)
                    self.frames_table.setCurrentCell(row, 0)
                finally:
                    self._filling = False
                return True
        return False

    def step(self, delta: int) -> None:
        """Choose the previous or next frame of the table, in its order (the first when none is chosen)."""

        count = self.frames_table.rowCount()
        if not count:
            return
        row = self.frames_table.currentRow()
        row = 0 if row < 0 else (row + int(delta)) % count
        self.frames_table.selectRow(row)
        self.frames_table.setCurrentCell(row, 0)

    def _selection_changed(self) -> None:
        if self._filling:
            return
        key = self.current_frame()
        if key:
            self.frameChosen.emit(key)

    def set_preview(self, pixmap: QPixmap | None, caption: str = "") -> None:
        self.preview.set_picture(pixmap, self._t("dirt.preview_empty"))
        self.preview_caption.setText(caption)
