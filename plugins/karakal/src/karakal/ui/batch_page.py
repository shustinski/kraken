"""«Compute and export» page: a snapshot folder, the layers found in it, the format of the error
maps, where to write them, and one button. No matrix."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.batch_progress import DONE, RUNNING, clock_text
from .toolbar_icons import toolbar_icon

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
    pauseRequested = pyqtSignal()
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

        # The run: the same run and pause buttons as the main window.
        run_row = QHBoxLayout()
        self.run_button = QPushButton(self)
        self.run_button.setProperty("runButton", True)
        self.run_button.setProperty("runState", "go")
        self.run_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_button.setMinimumWidth(190)
        self.pause_button = QPushButton(self)
        self.pause_button.setProperty("runButton", True)
        self.pause_button.setProperty("runState", "pause")
        self.pause_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pause_button.setVisible(False)
        self.elapsed_label = QLabel(self)
        self.elapsed_label.setStyleSheet(_MUTED)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.pause_button)
        run_row.addStretch(1)
        run_row.addWidget(self.elapsed_label)
        root.addLayout(run_row)
        # Kept for callers that stop the run by its own button.
        self.cancel_button = self.run_button
        self.status = QLabel(self)
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        # The whole run: share and time left, then every stage.
        self.progress_box = QWidget(self)
        progress_layout = QVBoxLayout(self.progress_box)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(6)
        self.total_label = QLabel(self.progress_box)
        self.total_label.setStyleSheet("font-weight: 700; font-size: 14px;")
        self.total_bar = QProgressBar(self.progress_box)
        self.total_bar.setRange(0, 1000)
        self.total_bar.setTextVisible(False)
        self.total_bar.setFixedHeight(10)
        progress_layout.addWidget(self.total_label)
        progress_layout.addWidget(self.total_bar)
        self.stages_grid = QGridLayout()
        self.stages_grid.setHorizontalSpacing(10)
        self.stages_grid.setVerticalSpacing(3)
        self.stages_grid.setColumnStretch(1, 1)
        progress_layout.addLayout(self.stages_grid)
        self.stage_bar = QProgressBar(self.progress_box)
        self.stage_bar.setRange(0, 1000)
        self.stage_bar.setTextVisible(False)
        self.stage_bar.setFixedHeight(6)
        self.stage_detail = QLabel(self.progress_box)
        self.stage_detail.setStyleSheet(_MUTED)
        progress_layout.addWidget(self.stage_bar)
        progress_layout.addWidget(self.stage_detail)
        problems_row = QHBoxLayout()
        self.problems_button = QPushButton(self.progress_box)
        self.problems_button.setCheckable(True)
        self.problems_button.setStyleSheet("text-align: left; color: #e8a33d;")
        problems_row.addWidget(self.problems_button)
        problems_row.addStretch(1)
        progress_layout.addLayout(problems_row)
        self.problems_list = QListWidget(self.progress_box)
        self.problems_list.setMaximumHeight(140)
        self.problems_list.setVisible(False)
        progress_layout.addWidget(self.problems_list)
        self.progress_box.setVisible(False)
        root.addWidget(self.progress_box)
        self._stage_rows: dict[str, tuple[QLabel, QLabel, QLabel]] = {}
        self._problems: list[tuple[str, str, str]] = []
        self._running = False
        self._paused = False
        self._last_view = None
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
        self.run_button.clicked.connect(self._run_clicked)
        self.pause_button.clicked.connect(self.pauseRequested)
        self.problems_button.toggled.connect(self.problems_list.setVisible)
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
        self._sync_buttons()
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
        # Every layer in sight, whatever else the page shows.
        table.resizeRowsToContents()
        height = table.horizontalHeader().height() + 2 * table.frameWidth() + 4
        height += sum(table.rowHeight(row) for row in range(table.rowCount()))
        table.setFixedHeight(max(height, table.horizontalHeader().height() + 36))
        notes = [self._t(f"batch.note.{note}") for note in layout.notes]
        if layout.is_pair:
            notes.insert(0, self._t("batch.note.pair"))
        self.layers_note.setText(" ".join(notes))
        self._has_layers = bool(layout.layers)
        self.set_status("batch.ready" if self._has_layers else "batch.no_layers")
        self._sync()

    # State --------------------------------------------------------------------------------------

    def _sync(self, running: bool = False) -> None:
        self._running = bool(running)
        self.run_button.setEnabled(self._has_layers or running)
        for widget in (self.snapshot_button, self.output_button, self.format_combo, self.output_edit, self.back_button):
            widget.setEnabled(not running)
        self._sync_buttons()

    def _run_clicked(self) -> None:
        if self._running:
            self.cancelRequested.emit()
        else:
            self.runRequested.emit()

    @staticmethod
    def _restyle(button: QPushButton, state: str) -> None:
        if button.property("runState") != state:
            button.setProperty("runState", state)
            button.style().unpolish(button)
            button.style().polish(button)

    def _sync_buttons(self) -> None:
        if self._running:
            share = self._last_view.share if self._last_view is not None else None
            text = self._t("batch.stop") if share is None else self._t("batch.stop_percent", percent=int(share * 100))
            self.run_button.setText(text)
            self.run_button.setIcon(toolbar_icon("stop"))
            self._restyle(self.run_button, "stop")
        else:
            self.run_button.setText(self._t("batch.run"))
            self.run_button.setIcon(toolbar_icon("play"))
            self._restyle(self.run_button, "go")
        self.pause_button.setVisible(self._running)
        self.pause_button.setText(self._t("run.resume") if self._paused else self._t("run.pause"))
        self.pause_button.setIcon(toolbar_icon("play" if self._paused else "pause"))
        self._restyle(self.pause_button, "paused" if self._paused else "pause")
        self._sync_problems()

    def set_running(self, running: bool) -> None:
        if running:
            self.open_folder_button.setVisible(False)
            self.open_project_button.setVisible(False)
            self._paused = False
            self._last_view = None
            self._problems = []
            self.problems_list.clear()
            self.problems_button.setChecked(False)
            self.progress_box.setVisible(True)
        self._sync(running)

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)
        self._sync_buttons()

    # Progress -----------------------------------------------------------------------------------

    def set_progress(self, view) -> None:
        """The whole run and every stage (a core.batch_progress.ProgressView)."""

        self._last_view = view
        self.progress_box.setVisible(True)
        self.total_bar.setValue(int(view.share * 1000))
        if view.share >= 1.0:
            total = self._t("batch.total_done", elapsed=clock_text(view.elapsed))
        elif view.eta is None:
            total = self._t("batch.total_start", percent=int(view.share * 100))
        else:
            total = self._t("batch.total", percent=int(view.share * 100), eta=clock_text(view.eta))
        if view.paused:
            total += " · " + self._t("batch.paused")
        self.total_label.setText(total)
        self.elapsed_label.setText(self._t("batch.elapsed", elapsed=clock_text(view.elapsed)))
        if set(self._stage_rows) != {stage.key for stage in view.stages}:
            self._build_stage_rows(view.stages)
        current = None
        for stage in view.stages:
            mark, title, info = self._stage_rows[stage.key]
            if stage.state == DONE:
                mark.setText("✓")
                mark.setStyleSheet("color: #3fb950;")
                info.setText(
                    (f"{stage.total} · " if stage.total else "") + clock_text(stage.seconds)
                )
                title.setStyleSheet(_MUTED)
            elif stage.state == RUNNING:
                current = stage
                mark.setText("▶")
                mark.setStyleSheet("color: #58a6ff;")
                title.setStyleSheet("font-weight: 700;")
                info.setText(f"{stage.done} / {stage.total}" if stage.total else "")
            else:
                mark.setText("○")
                mark.setStyleSheet(_MUTED)
                title.setStyleSheet(_MUTED)
                info.setText("")
        self.stage_bar.setVisible(current is not None)
        self.stage_detail.setVisible(current is not None)
        if current is not None:
            self.stage_bar.setValue(int(current.share * 1000))
            parts = []
            if current.rate > 0:
                parts.append(self._t("batch.rate", rate=f"{current.rate:.0f}" if current.rate >= 10 else f"{current.rate:.1f}"))
            if current.detail:
                parts.append(current.detail)
            self.stage_detail.setText(" · ".join(parts))
        self._sync_buttons()

    def _build_stage_rows(self, stages) -> None:
        while self.stages_grid.count():
            item = self.stages_grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._stage_rows = {}
        for row, stage in enumerate(stages):
            mark = QLabel("○", self.progress_box)
            title = QLabel(stage.title, self.progress_box)
            info = QLabel("", self.progress_box)
            info.setStyleSheet(_MUTED)
            info.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.stages_grid.addWidget(mark, row, 0)
            self.stages_grid.addWidget(title, row, 1)
            self.stages_grid.addWidget(info, row, 2)
            self._stage_rows[stage.key] = (mark, title, info)

    def set_problems(self, problems) -> None:
        """Frames that could not be checked or written: (frame, layer, why)."""

        self._problems = list(problems or ())
        self.problems_list.clear()
        for frame, layer, why in self._problems:
            self.problems_list.addItem(" · ".join(part for part in (frame, layer, why) if part))
        self._sync_problems()

    def _sync_problems(self) -> None:
        frames = len({frame for frame, _layer, _why in self._problems})
        self.problems_button.setVisible(frames > 0)
        self.problems_button.setText(self._t("batch.problems", count=frames))
        if not frames:
            self.problems_list.setVisible(False)

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
