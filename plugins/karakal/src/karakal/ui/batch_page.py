"""«Compute and export» page: a snapshot folder, the layers found in it, the format of the error
maps, where to write them, and one button. No matrix."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

Translate = Callable[..., str]
_MUTED = "color: #8a94a3;"
FORMATS = (("png", "PNG"), ("jpg", "JPG"), ("bmp", "BMP"), ("tif", "TIFF"))


class BatchPage(QWidget):
    backRequested = pyqtSignal()
    browseRequested = pyqtSignal()
    snapshotChosen = pyqtSignal(str)
    outputBrowseRequested = pyqtSignal()
    runRequested = pyqtSignal()
    cancelRequested = pyqtSignal()
    openFolderRequested = pyqtSignal()
    openProjectRequested = pyqtSignal()

    def __init__(self, translate: Translate, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("karakalBatchPage")
        self._t = translate
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 28, 48, 28)
        root.setSpacing(10)
        top = QHBoxLayout()
        self.back_button = QPushButton(self)
        self.title = QLabel(self)
        self.title.setStyleSheet("font-size: 22px; font-weight: 700;")
        top.addWidget(self.back_button)
        top.addSpacing(12)
        top.addWidget(self.title, 1)
        root.addLayout(top)
        self.subtitle = QLabel(self)
        self.subtitle.setWordWrap(True)
        self.subtitle.setStyleSheet(_MUTED)
        root.addWidget(self.subtitle)

        self.snapshot_label = QLabel(self)
        self.snapshot_label.setStyleSheet("font-weight: 600; margin-top: 8px;")
        root.addWidget(self.snapshot_label)
        snapshot_row = QHBoxLayout()
        self.snapshot_edit = QLineEdit(self)
        self.snapshot_edit.setReadOnly(True)
        self.snapshot_button = QPushButton(self)
        snapshot_row.addWidget(self.snapshot_edit, 1)
        snapshot_row.addWidget(self.snapshot_button)
        root.addLayout(snapshot_row)

        self.layers_label = QLabel(self)
        self.layers_label.setStyleSheet("font-weight: 600; margin-top: 8px;")
        root.addWidget(self.layers_label)
        self.layers_table = QTableWidget(0, 5, self)
        self.layers_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.layers_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.layers_table.verticalHeader().setVisible(False)
        header = self.layers_table.horizontalHeader()
        for column in range(5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.layers_table.setMaximumHeight(140)
        root.addWidget(self.layers_table)
        self.layers_note = QLabel(self)
        self.layers_note.setWordWrap(True)
        self.layers_note.setStyleSheet(_MUTED)
        root.addWidget(self.layers_note)

        export_row = QHBoxLayout()
        self.format_label = QLabel(self)
        self.format_combo = QComboBox(self)
        for value, text in FORMATS:
            self.format_combo.addItem(text, value)
        export_row.addWidget(self.format_label)
        export_row.addWidget(self.format_combo)
        export_row.addStretch(1)
        root.addLayout(export_row)
        self.output_label = QLabel(self)
        self.output_label.setStyleSheet("font-weight: 600; margin-top: 8px;")
        root.addWidget(self.output_label)
        output_row = QHBoxLayout()
        self.output_edit = QLineEdit(self)
        self.output_button = QPushButton(self)
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(self.output_button)
        root.addLayout(output_row)

        run_row = QHBoxLayout()
        self.run_button = QPushButton(self)
        self.run_button.setStyleSheet("font-weight: 700; padding: 6px 18px;")
        self.cancel_button = QPushButton(self)
        self.cancel_button.setVisible(False)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.cancel_button)
        run_row.addStretch(1)
        root.addLayout(run_row)
        self.status = QLabel(self)
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        done_row = QHBoxLayout()
        self.open_folder_button = QPushButton(self)
        self.open_project_button = QPushButton(self)
        for button in (self.open_folder_button, self.open_project_button):
            button.setVisible(False)
            done_row.addWidget(button)
        done_row.addStretch(1)
        root.addLayout(done_row)
        root.addStretch(1)

        self.back_button.clicked.connect(self.backRequested)
        self.snapshot_button.clicked.connect(self.browseRequested)
        self.output_button.clicked.connect(self.outputBrowseRequested)
        self.run_button.clicked.connect(self.runRequested)
        self.cancel_button.clicked.connect(self.cancelRequested)
        self.open_folder_button.clicked.connect(self.openFolderRequested)
        self.open_project_button.clicked.connect(self.openProjectRequested)
        self._status_key = "batch.ready_hint"
        self._status_args: dict = {}
        self._has_layers = False
        self.retranslate(translate)
        self._sync()

    def retranslate(self, translate: Translate) -> None:
        self._t = translate
        self.back_button.setText(self._t("batch.back"))
        self.title.setText(self._t("batch.title"))
        self.subtitle.setText(self._t("batch.subtitle"))
        self.snapshot_label.setText(self._t("batch.snapshot"))
        self.snapshot_button.setText(self._t("batch.browse"))
        self.layers_label.setText(self._t("batch.layers"))
        self.layers_table.setHorizontalHeaderLabels(
            [
                self._t("batch.col.layer"),
                self._t("batch.col.class"),
                self._t("batch.col.masks"),
                self._t("batch.col.confidence"),
                self._t("batch.col.sources"),
            ]
        )
        self.format_label.setText(self._t("batch.format"))
        self.output_label.setText(self._t("batch.output"))
        self.output_button.setText(self._t("batch.browse"))
        self.run_button.setText(self._t("batch.run"))
        self.cancel_button.setText(self._t("batch.cancel"))
        self.open_folder_button.setText(self._t("batch.open_folder"))
        self.open_project_button.setText(self._t("batch.open_project"))
        self.status.setText(self._t(self._status_key, **self._status_args))

    # Inputs -------------------------------------------------------------------------------------

    def errors_format(self) -> str:
        return str(self.format_combo.currentData() or "png")

    def output_dir(self) -> str:
        return self.output_edit.text().strip()

    def set_output_dir(self, path: str) -> None:
        self.output_edit.setText(str(path))

    def set_layout(self, layout) -> None:
        self.snapshot_edit.setText(str(layout.root))
        table = self.layers_table
        table.setRowCount(len(layout.layers))
        for row, layer in enumerate(layout.layers):
            cells = (
                layer.label,
                self._t(f"batch.class.{layer.cell_class}") if layer.cell_class else self._t("batch.class.single"),
                str(layer.mask_count),
                f"{layer.confidence_folder.name} ({layer.confidence_count})" if layer.confidence_folder else "—",
                layer.source_folder.name if layer.source_folder else "—",
            )
            for column, text in enumerate(cells):
                table.setItem(row, column, QTableWidgetItem(text))
        notes = [self._t(f"batch.note.{note}") for note in layout.notes]
        if layout.is_pair:
            notes.insert(0, self._t("batch.note.pair"))
        self.layers_note.setText(" ".join(notes))
        self._has_layers = bool(layout.layers)
        self.set_status("batch.ready" if self._has_layers else "batch.no_layers")
        self._sync()

    # State --------------------------------------------------------------------------------------

    def _sync(self, running: bool = False) -> None:
        self.run_button.setEnabled(self._has_layers and not running)
        self.cancel_button.setVisible(running)
        for widget in (self.snapshot_button, self.output_button, self.format_combo, self.output_edit, self.back_button):
            widget.setEnabled(not running)

    def set_running(self, running: bool) -> None:
        if running:
            self.open_folder_button.setVisible(False)
            self.open_project_button.setVisible(False)
        self._sync(running)

    def set_status(self, key: str, **args) -> None:
        self._status_key = str(key)
        self._status_args = dict(args)
        self.status.setText(self._t(self._status_key, **self._status_args))

    def set_done(self, report) -> None:
        self.set_status(
            "batch.done" if not report.missing else "batch.done_missing",
            frames=report.frames,
            files=report.written,
            missing=len(report.missing),
            path=str(report.output_dir or ""),
        )
        self.open_folder_button.setVisible(report.output_dir is not None)
        self.open_project_button.setVisible(report.project is not None)
