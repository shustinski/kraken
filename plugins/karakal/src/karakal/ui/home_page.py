"""Home page shown when Karakal starts: a snapshot folder computed and exported from here, the
matrix of it opened from here, the recent selections to pick again.

Left, the run: the snapshot (dropped on the page or chosen by its path), the layers found in it,
the format of the error maps and their folder, the run button that fills as the run goes, pause.
Right, the matrix button: how ready the matrix of the chosen snapshot is (the stage of a run, the
time left, what a done run found), then the recent selections; picking one puts its paths back.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.batch_progress import DONE, RUNNING, clock_text
from ..core.recent_selections import (
    MATRIX_EMPTY,
    MATRIX_NONE,
    MATRIX_PARTIAL,
    MATRIX_READY,
    MATRIX_RUNNING,
    RUN_DONE,
    RUN_FAILED,
    RUN_STOPPED,
    Selection,
)
from .toolbar_icons import toolbar_icon

Translate = Callable[..., str]
_MUTED = "color: #8a94a3;"
_WARN = "#e8a33d"
_OK = "#3fb950"
_MUTED_COLOR = "#8a94a3"
FORMATS = (("png", "PNG"), ("jpg", "JPG"), ("bmp", "BMP"), ("tif", "TIFF"))
SELECTION_ROLE = Qt.ItemDataRole.UserRole


def _when(text: str) -> str:
    try:
        moment = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return ""
    if moment.date() == datetime.now().date():
        return moment.strftime("%H:%M")
    return moment.strftime("%d.%m.%Y %H:%M")


def fill_style(base: str, fill: str, border: str, color: str, share: float, *, extra: str = "") -> str:
    """A push button's own style: ``fill`` from the left over ``share`` of it, ``base`` after."""

    share = max(0.0, min(1.0, float(share)))
    if share <= 0.0:
        background = base
    elif share >= 1.0:
        background = fill
    else:
        edge = min(1.0, share + 0.001)
        background = (
            f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {fill}, stop:{share:.4f} {fill}, "
            f"stop:{edge:.4f} {base}, stop:1 {base})"
        )
    return (
        f"QPushButton {{ background: {background}; border: 1px solid {border}; border-radius: 8px; "
        f"color: {color}; {extra} }}"
        f"QPushButton:hover {{ border-color: #6f8fb0; }}"
        f"QPushButton:disabled {{ color: #6f7a86; border-color: #283541; }}"
    )


class DropZone(QFrame):
    """A frame a snapshot folder is dropped on."""

    folderDropped = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("snapshotDropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(86)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        self.icon = QLabel(self)
        self.icon.setPixmap(toolbar_icon("folder").pixmap(26, 26))
        self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        layout.addStretch(1)
        layout.addWidget(self.icon)
        layout.addWidget(self.label)
        layout.addStretch(1)
        self._highlight(False)

    def _highlight(self, on: bool) -> None:
        border = "#4e90ff" if on else "#46627f"
        background = "#15253a" if on else "transparent"
        self.setStyleSheet(
            f"QFrame#snapshotDropZone {{ border: 1px dashed {border}; border-radius: 10px; background: {background}; }}"
        )

    @staticmethod
    def _folder(event) -> str:
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            return ""
        for url in mime.urls():
            if url.isLocalFile():
                path = Path(url.toLocalFile())
                if path.is_dir():
                    return str(path)
                if path.is_file():
                    return str(path.parent)
        return ""

    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt)
        if self._folder(event):
            event.acceptProposedAction()
            self._highlight(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 (Qt)
        self._highlight(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt)
        self._highlight(False)
        folder = self._folder(event)
        if folder:
            event.acceptProposedAction()
            self.folderDropped.emit(folder)


class MatrixButton(QPushButton):
    """The matrix of the chosen snapshot: a card that fills with how much of it is computed."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("matrixButton")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)
        top = QHBoxLayout()
        top.setSpacing(8)
        self.icon = QLabel(self)
        self.icon.setPixmap(toolbar_icon("grid").pixmap(20, 20))
        self.title = QLabel(self)
        self.title.setStyleSheet("font-size: 15px; font-weight: 700;")
        self.corner = QLabel(self)
        self.corner.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self.icon)
        top.addWidget(self.title, 1)
        top.addWidget(self.corner)
        layout.addLayout(top)
        self.lines = [QLabel(self) for _ in range(4)]
        for line in self.lines:
            line.setWordWrap(True)
            layout.addWidget(line)
        for label in (self.icon, self.title, self.corner, *self.lines):
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            label.setStyleSheet(label.styleSheet() + "background: transparent;")
        self.state = MATRIX_EMPTY
        self.share = 0.0

    def show_card(
        self,
        state: str,
        share: float,
        title: str,
        corner: tuple[str, str] = ("", ""),
        lines: list[tuple[str, str]] = (),
    ) -> None:
        """``lines``: (text, colour) each; empty colour is the button's own."""

        self.state = str(state)
        self.share = max(0.0, min(1.0, float(share)))
        if state == MATRIX_READY:
            look = ("#111a24", "#1b4d31", "#2fa463", "#eafbef")
        elif state in (MATRIX_RUNNING, MATRIX_PARTIAL):
            look = ("#111a24", "#4d3a14", "#c08f3a", "#edf3fb")
        else:
            look = ("#111a24", "#111a24", "#30445a", "#edf3fb")
        base, fill, border, color = look
        self.setStyleSheet(fill_style(base, fill, border, color, self.share if state != MATRIX_READY else 1.0, extra="text-align: left;"))
        self.title.setText(title)
        self.title.setStyleSheet(f"font-size: 15px; font-weight: 700; background: transparent; color: {_OK if state == MATRIX_READY else color};")
        corner_text, corner_color = corner
        self.corner.setText(corner_text)
        self.corner.setStyleSheet(f"background: transparent; color: {corner_color or _MUTED_COLOR};")
        lines = list(lines)
        for index, label in enumerate(self.lines):
            text, tone = lines[index] if index < len(lines) else ("", "")
            label.setText(text)
            label.setVisible(bool(text))
            label.setStyleSheet(f"background: transparent; color: {tone or color}; font-size: 12px;")
        self._fit()

    def _fit(self) -> None:
        """As tall as its lines at its width (wrapped lines measured at the width they have)."""

        layout = self.layout()
        width = max(self.width(), 200)
        height = layout.totalHeightForWidth(width) if layout.hasHeightForWidth() else layout.totalSizeHint().height()
        if height > 0 and height != self.height():
            self.setFixedHeight(height)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt)
        super().resizeEvent(event)
        self._fit()


class HomePage(QWidget):
    snapshotBrowseRequested = pyqtSignal()
    snapshotChosen = pyqtSignal(str)
    outputBrowseRequested = pyqtSignal()
    formatChanged = pyqtSignal(str)
    runRequested = pyqtSignal()
    cancelRequested = pyqtSignal()
    pauseRequested = pyqtSignal()
    openFolderRequested = pyqtSignal()
    matrixRequested = pyqtSignal()
    selectionPicked = pyqtSignal(str)
    forgetRequested = pyqtSignal(str)
    otherProjectRequested = pyqtSignal()
    projectsFolderRequested = pyqtSignal()

    def __init__(self, translate: Translate, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("karakalHomePage")
        self._t = translate
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 28, 48, 28)
        root.setSpacing(14)
        self.title = QLabel(self)
        self.title.setStyleSheet("font-size: 24px; font-weight: 700;")
        root.addWidget(self.title)
        body = QHBoxLayout()
        body.setSpacing(28)
        root.addLayout(body, 1)

        # Left: the snapshot and its run --------------------------------------------------------
        left = QVBoxLayout()
        left.setSpacing(8)
        body.addLayout(left, 135)
        self.drop_zone = DropZone(self)
        left.addWidget(self.drop_zone)
        snapshot_row = QHBoxLayout()
        self.snapshot_label = QLabel(self)
        self.snapshot_label.setStyleSheet(_MUTED)
        self.snapshot_edit = QLineEdit(self)
        self.snapshot_edit.setClearButtonEnabled(True)
        self.snapshot_button = QPushButton(self)
        snapshot_row.addWidget(self.snapshot_label)
        snapshot_row.addWidget(self.snapshot_edit, 1)
        snapshot_row.addWidget(self.snapshot_button)
        left.addLayout(snapshot_row)
        summary_row = QHBoxLayout()
        self.summary = QLabel(self)
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(_MUTED)
        self.details_button = QPushButton(self)
        self.details_button.setCheckable(True)
        self.details_button.setFlat(True)
        self.details_button.setVisible(False)
        self.details_button.setStyleSheet("QPushButton { border: none; background: transparent; color: #58a6ff; padding: 0 4px; }")
        summary_row.addWidget(self.summary, 1)
        summary_row.addWidget(self.details_button, 0, Qt.AlignmentFlag.AlignTop)
        left.addLayout(summary_row)
        self.layers_table = QTableWidget(0, 5, self)
        self.layers_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.layers_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.layers_table.verticalHeader().setVisible(False)
        header = self.layers_table.horizontalHeader()
        for column in range(5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.layers_table.setVisible(False)
        left.addWidget(self.layers_table)
        self.layers_note = QLabel(self)
        self.layers_note.setWordWrap(True)
        self.layers_note.setStyleSheet(_MUTED)
        self.layers_note.setVisible(False)
        left.addWidget(self.layers_note)
        format_row = QHBoxLayout()
        format_row.setSpacing(0)
        self.format_label = QLabel(self)
        self.format_label.setStyleSheet(_MUTED)
        format_row.addWidget(self.format_label)
        format_row.addSpacing(10)
        self.format_group = QButtonGroup(self)
        self.format_group.setExclusive(True)
        self.format_buttons: dict[str, QPushButton] = {}
        for index, (value, text) in enumerate(FORMATS):
            button = QPushButton(text, self)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            radius = "border-radius: 0px;"
            if index == 0:
                radius = "border-top-right-radius: 0px; border-bottom-right-radius: 0px;"
            elif index == len(FORMATS) - 1:
                radius = "border-top-left-radius: 0px; border-bottom-left-radius: 0px;"
            button.setStyleSheet(f"QPushButton {{ padding: 3px 12px; {radius} }}")
            self.format_group.addButton(button, index)
            self.format_buttons[value] = button
            format_row.addWidget(button)
        format_row.addStretch(1)
        self.format_buttons["png"].setChecked(True)
        left.addLayout(format_row)
        output_row = QHBoxLayout()
        self.output_label = QLabel(self)
        self.output_label.setStyleSheet(_MUTED)
        self.output_edit = QLineEdit(self)
        self.output_button = QPushButton(self)
        output_row.addWidget(self.output_label)
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(self.output_button)
        left.addLayout(output_row)
        # The run button fills with the whole run as it goes.
        run_row = QHBoxLayout()
        run_row.setSpacing(8)
        self.run_button = QPushButton(self)
        self.run_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_button.setMinimumWidth(260)
        self.run_button.setMinimumHeight(36)
        self.pause_button = QPushButton(self)
        self.pause_button.setProperty("runButton", True)
        self.pause_button.setProperty("runState", "pause")
        self.pause_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pause_button.setMinimumHeight(36)
        self.pause_button.setVisible(False)
        self.elapsed_label = QLabel(self)
        self.elapsed_label.setStyleSheet(_MUTED)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.pause_button)
        run_row.addStretch(1)
        run_row.addWidget(self.elapsed_label)
        left.addSpacing(4)
        left.addLayout(run_row)
        # Kept for callers that stop the run by its own button.
        self.cancel_button = self.run_button
        self.status = QLabel(self)
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        problems_row = QHBoxLayout()
        self.problems_button = QPushButton(self)
        self.problems_button.setCheckable(True)
        self.problems_button.setFlat(True)
        self.problems_button.setStyleSheet(
            f"QPushButton {{ border: none; background: transparent; color: {_WARN}; text-align: left; padding: 0; }}"
        )
        problems_row.addWidget(self.problems_button)
        problems_row.addStretch(1)
        left.addLayout(problems_row)
        self.problems_list = QListWidget(self)
        self.problems_list.setMaximumHeight(140)
        self.problems_list.setVisible(False)
        left.addWidget(self.problems_list)
        done_row = QHBoxLayout()
        self.open_folder_button = QPushButton(self)
        self.open_folder_button.setVisible(False)
        done_row.addWidget(self.open_folder_button)
        done_row.addStretch(1)
        left.addLayout(done_row)
        left.addStretch(1)

        # Right: the matrix and the recent selections ------------------------------------------
        right = QVBoxLayout()
        right.setSpacing(6)
        body.addLayout(right, 100)
        self.matrix_button = MatrixButton(self)
        right.addWidget(self.matrix_button)
        links = QHBoxLayout()
        links.setSpacing(12)
        self.other_button = QPushButton(self)
        self.folder_button = QPushButton(self)
        for button in (self.other_button, self.folder_button):
            button.setFlat(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setStyleSheet("QPushButton { border: none; background: transparent; color: #58a6ff; padding: 0 2px; }")
            links.addWidget(button)
        links.addStretch(1)
        right.addLayout(links)
        right.addSpacing(10)
        self.recent_title = QLabel(self)
        self.recent_title.setStyleSheet("font-size: 14px; font-weight: 600;")
        right.addWidget(self.recent_title)
        self.selections = QListWidget(self)
        self.selections.setSpacing(2)
        right.addWidget(self.selections, 1)
        self.empty_hint = QLabel(self)
        self.empty_hint.setWordWrap(True)
        self.empty_hint.setStyleSheet(_MUTED)
        right.addWidget(self.empty_hint)
        forget_row = QHBoxLayout()
        self.forget_button = QPushButton(self)
        forget_row.addStretch(1)
        forget_row.addWidget(self.forget_button)
        right.addLayout(forget_row)

        self.drop_zone.folderDropped.connect(self.snapshotChosen)
        self.snapshot_edit.returnPressed.connect(self._snapshot_typed)
        self.snapshot_button.clicked.connect(self.snapshotBrowseRequested)
        self.details_button.toggled.connect(self._show_details)
        self.format_group.idClicked.connect(lambda index: self.formatChanged.emit(FORMATS[index][0]))
        self.output_button.clicked.connect(self.outputBrowseRequested)
        self.run_button.clicked.connect(self._run_clicked)
        self.pause_button.clicked.connect(self.pauseRequested)
        self.problems_button.toggled.connect(self.problems_list.setVisible)
        self.open_folder_button.clicked.connect(self.openFolderRequested)
        self.matrix_button.clicked.connect(self.matrixRequested)
        self.other_button.clicked.connect(self.otherProjectRequested)
        self.folder_button.clicked.connect(self.projectsFolderRequested)
        self.selections.itemClicked.connect(self._pick_item)
        self.selections.itemActivated.connect(self._pick_item)
        self.selections.currentItemChanged.connect(lambda *_args: self._sync_forget())
        self.forget_button.clicked.connect(self._forget_current)

        self._layout = None
        self._has_layers = False
        self._running = False
        self._paused = False
        self._last_view = None
        self._problems: list[tuple[str, str, str]] = []
        self._status_key = ""
        self._status_args: dict = {}
        self._matrix_args: dict = {"state": MATRIX_EMPTY}
        self._selections: list[Selection] = []
        self._current = ""
        self.retranslate(translate)
        self.set_selections([])
        self._sync()

    def retranslate(self, translate: Translate) -> None:
        self._t = translate
        self.title.setText(self._t("home.title"))
        self.drop_zone.label.setText(self._t("home.drop"))
        self.snapshot_label.setText(self._t("home.or_path"))
        self.snapshot_edit.setPlaceholderText(self._t("home.path_placeholder"))
        self.snapshot_button.setText(self._t("batch.browse"))
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
        self.output_label.setText(self._t("home.output"))
        self.output_button.setText(self._t("home.change"))
        self.open_folder_button.setText(self._t("batch.open_folder"))
        self.other_button.setText(self._t("home.other_project"))
        self.folder_button.setText(self._t("home.projects_folder"))
        self.recent_title.setText(self._t("home.recent"))
        self.empty_hint.setText(self._t("home.empty"))
        self.forget_button.setText(self._t("home.forget"))
        self._show_details(self.details_button.isChecked())
        if self._layout is not None:
            self._show_summary(self._layout)
        self.status.setText(self._t(self._status_key, **self._status_args) if self._status_key else "")
        self.set_selections(self._selections, self._current)
        self.set_matrix(**self._matrix_args)
        self._sync_buttons()

    # Inputs -------------------------------------------------------------------------------------

    def _snapshot_typed(self) -> None:
        text = self.snapshot_edit.text().strip().strip('"')
        if text:
            self.snapshotChosen.emit(text)

    def errors_format(self) -> str:
        for value, button in self.format_buttons.items():
            if button.isChecked():
                return value
        return "png"

    def set_errors_format(self, value: str) -> None:
        button = self.format_buttons.get(str(value or "").lower()) or self.format_buttons["png"]
        button.setChecked(True)

    def output_dir(self) -> str:
        return self.output_edit.text().strip().strip('"')

    def set_output_dir(self, path: str) -> None:
        self.output_edit.setText(str(path or ""))

    def snapshot_dir(self) -> str:
        return self.snapshot_edit.text().strip().strip('"')

    def set_layout(self, layout) -> None:
        self._layout = layout
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
        table.resizeRowsToContents()
        height = table.horizontalHeader().height() + 2 * table.frameWidth() + 4
        height += sum(table.rowHeight(row) for row in range(table.rowCount()))
        table.setFixedHeight(max(height, table.horizontalHeader().height() + 36))
        notes = [self._t(f"batch.note.{note}") for note in layout.notes]
        if layout.is_pair:
            notes.insert(0, self._t("batch.note.pair"))
        self.layers_note.setText(" ".join(notes))
        self._show_summary(layout)
        self._has_layers = bool(layout.layers)
        self.set_status("" if self._has_layers else "batch.no_layers")
        self.open_folder_button.setVisible(False)
        self._problems = []
        self.problems_list.clear()
        self._last_view = None
        self.elapsed_label.setText("")
        self._sync()

    def _show_summary(self, layout) -> None:
        if not layout.layers:
            self.summary.setText(self._t("batch.no_layers") if "missing" not in layout.notes else self._t("batch.note.missing"))
            self.details_button.setVisible(False)
            return
        parts = []
        for layer in layout.layers:
            name = self._t(f"batch.class.{layer.cell_class}") if layer.cell_class else layer.label
            parts.append(f"{name} {layer.mask_count}")
        yes, no = "✓", "—"
        parts.append(self._t("home.confidence", mark=yes if all(layer.confidence_folder for layer in layout.layers) else no))
        parts.append(self._t("home.sources", mark=yes if all(layer.source_folder for layer in layout.layers) else no))
        self.summary.setText(self._t("home.found", what=" · ".join(parts)))
        self.details_button.setVisible(True)

    def _show_details(self, shown: bool) -> None:
        self.details_button.setText(self._t("home.details_hide" if shown else "home.details"))
        self.layers_table.setVisible(bool(shown) and self._layout is not None and bool(self._layout.layers))
        self.layers_note.setVisible(bool(shown) and bool(self.layers_note.text()))

    # The run ------------------------------------------------------------------------------------

    def _sync(self) -> None:
        running = self._running
        self.run_button.setEnabled(self._has_layers or running)
        for widget in (
            self.snapshot_button,
            self.snapshot_edit,
            self.output_button,
            self.output_edit,
            self.drop_zone,
            self.selections,
            self.forget_button,
            *self.format_buttons.values(),
        ):
            widget.setEnabled(not running)
        self._sync_buttons()
        self._sync_forget()

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
        button = self.run_button
        extra = "font-weight: 700; padding: 6px 16px;"
        if self._running:
            share = self._last_view.share if self._last_view is not None else 0.0
            button.setText(self._t("batch.stop_percent", percent=int(share * 100)))
            button.setIcon(toolbar_icon("stop"))
            button.setProperty("runState", "stop")
            if self._paused:
                button.setStyleSheet(fill_style("#3d2f14", "#8a6424", "#c08f3a", "#fff3d6", share, extra=extra))
            else:
                button.setStyleSheet(fill_style("#4a1c24", "#8c2f3c", "#c24a5a", "#ffffff", share, extra=extra))
        else:
            button.setText(self._t("batch.run"))
            button.setIcon(toolbar_icon("play"))
            button.setProperty("runState", "go")
            button.setStyleSheet(fill_style("#1f7a46", "#1f7a46", "#2fa463", "#ffffff", 0.0, extra=extra))
        self.pause_button.setVisible(self._running)
        self.pause_button.setText(self._t("run.resume") if self._paused else self._t("run.pause"))
        self.pause_button.setIcon(toolbar_icon("play" if self._paused else "pause"))
        self._restyle(self.pause_button, "paused" if self._paused else "pause")
        self._sync_problems()

    def set_running(self, running: bool) -> None:
        if running:
            self.open_folder_button.setVisible(False)
            self._paused = False
            self._last_view = None
            self._problems = []
            self.problems_list.clear()
            self.problems_button.setChecked(False)
            self.elapsed_label.setText("")
            self.set_status("")
        self._running = bool(running)
        self._sync()

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)
        self._sync_buttons()

    def set_progress(self, view) -> None:
        """The whole run in the run button, its stage in the matrix button (a ProgressView)."""

        self._last_view = view
        self._paused = bool(view.paused)
        elapsed = self._t("batch.elapsed", elapsed=clock_text(view.elapsed))
        if view.paused:
            elapsed += " · " + self._t("batch.paused")
        self.elapsed_label.setText(elapsed)
        self._sync_buttons()
        if self._matrix_args.get("state") == MATRIX_RUNNING:
            self.set_matrix(**self._matrix_args)

    def set_problems(self, problems) -> None:
        """Frames that could not be checked or written: (frame, layer, why)."""

        self._problems = list(problems or ())
        self.problems_list.clear()
        for frame, layer, why in self._problems:
            self.problems_list.addItem(" · ".join(part for part in (frame, layer, why) if part))
        self._sync_problems()
        if self._matrix_args.get("state") == MATRIX_RUNNING:
            self.set_matrix(**self._matrix_args)

    def problem_frames(self) -> int:
        return len({frame for frame, _layer, _why in self._problems})

    def _sync_problems(self) -> None:
        frames = self.problem_frames()
        self.problems_button.setVisible(frames > 0)
        self.problems_button.setText(self._t("batch.problems", count=frames))
        if not frames:
            self.problems_list.setVisible(False)

    def set_status(self, key: str, **args) -> None:
        self._status_key = str(key or "")
        self._status_args = dict(args)
        self.status.setText(self._t(self._status_key, **self._status_args) if self._status_key else "")

    def set_done(self, report, seconds: float) -> None:
        if report.missing:
            self.set_status("home.done_missing", files=report.written, elapsed=clock_text(seconds), missing=len(report.missing))
        else:
            self.set_status("home.done", files=report.written, elapsed=clock_text(seconds))
        self.status.setStyleSheet(f"color: {_OK}; font-weight: 600;")
        self.open_folder_button.setVisible(report.output_dir is not None)

    def show_message(self, key: str, **args) -> None:
        """A plain line under the run button (stopped, failed, nothing to run)."""

        self.status.setStyleSheet("")
        self.set_status(key, **args)

    # The matrix button ------------------------------------------------------------------------

    def set_matrix(
        self,
        state: str = MATRIX_EMPTY,
        selection: Selection | None = None,
        layout=None,
        pace: float = 0.0,
        frames_per_row: int = 0,
    ) -> None:
        self._matrix_args = {
            "state": state,
            "selection": selection,
            "layout": layout,
            "pace": pace,
            "frames_per_row": frames_per_row,
        }
        t = self._t
        title = t("home.matrix")
        if state == MATRIX_RUNNING:
            self._matrix_running(title)
            return
        if state == MATRIX_READY and selection is not None:
            lines = []
            facts = [t("home.m.frames", count=selection.frames)]
            if frames_per_row > 0:
                facts.append(t("home.m.per_row", count=frames_per_row))
            if selection.defects >= 0:
                facts.append(t("home.m.defects", count=selection.defects))
            lines.append((" · ".join(facts), ""))
            when = [_when(selection.at)]
            if selection.seconds > 0:
                when.append(t("home.m.took", elapsed=clock_text(selection.seconds)))
            lines.append((" · ".join(part for part in when if part), _MUTED_COLOR))
            if selection.problems:
                lines.append((t("home.m.problems", count=selection.problems), _WARN))
            self.matrix_button.show_card(state, 1.0, title, (t("home.m.ready"), _OK), lines)
            return
        if state == MATRIX_PARTIAL and selection is not None:
            title = t("home.m.title_share", percent=int(selection.share * 100))
            what = "home.m.failed" if selection.run == RUN_FAILED else "home.m.stopped"
            lines = [(t(what, stage=selection.stage) if selection.stage else t(f"{what}_plain"), _WARN)]
            if selection.at:
                lines.append((_when(selection.at), _MUTED_COLOR))
            lines.append((t("home.m.rest_on_open"), _MUTED_COLOR))
            self.matrix_button.show_card(state, selection.share, title, (t("home.m.partial"), _WARN), lines)
            return
        if state == MATRIX_NONE and layout is not None and layout.layers:
            frames = max(layer.mask_count for layer in layout.layers)
            facts = [t("home.m.frames", count=frames), t("home.m.layers", count=len(layout.layers))]
            if layout.is_pair:
                facts.append(t("home.m.pair"))
            lines = [(" · ".join(facts), "")]
            if pace > 0:
                lines.append((t("home.m.estimate", elapsed=clock_text(pace * frames)), _MUTED_COLOR))
            else:
                lines.append((t("home.m.compute_on_open"), _MUTED_COLOR))
            self.matrix_button.show_card(state, 0.0, title, (t("home.m.none"), _MUTED_COLOR), lines)
            return
        self.matrix_button.show_card(MATRIX_EMPTY, 0.0, title, ("", ""), [(t("home.m.empty"), _MUTED_COLOR)])

    def _matrix_running(self, title: str) -> None:
        t = self._t
        view = self._last_view
        if view is None:
            self.matrix_button.show_card(MATRIX_RUNNING, 0.0, t("home.m.title_share", percent=0), ("", ""), [(t("batch.running"), _WARN)])
            return
        stages = list(view.stages)
        current_index = next((index for index, stage in enumerate(stages) if stage.state == RUNNING), -1)
        if current_index < 0:
            current_index = max((index for index, stage in enumerate(stages) if stage.state == DONE), default=0)
        current = stages[current_index] if stages else None
        lines: list[tuple[str, str]] = []
        if current is not None:
            lines.append((t("home.m.stage", index=current_index + 1, count=len(stages), title=current.title), ""))
            facts = []
            if current.total:
                facts.append(t("home.m.frame", done=current.done, total=current.total))
            if current.rate > 0:
                facts.append(t("batch.rate", rate=f"{current.rate:.0f}" if current.rate >= 10 else f"{current.rate:.1f}"))
            if current.detail:
                facts.append(current.detail)
            if facts:
                lines.append((" · ".join(facts), _MUTED_COLOR))
        marks = []
        for stage in stages:
            if stage.key.startswith("cells:") or stage.key.startswith("pair:"):
                mark = "✓" if stage.state == DONE else ("▶" if stage.state == RUNNING else "○")
                name = t("home.m.pair_short") if stage.key.startswith("pair:") else stage.name
                marks.append(f"{name} {mark}")
        if marks:
            lines.append((t("home.m.layers_line", marks=" · ".join(marks)), _MUTED_COLOR))
        problems = self.problem_frames()
        if problems:
            lines.append((t("batch.problems_short", count=problems), _WARN))
        if view.paused:
            corner = (t("batch.paused"), _WARN)
        elif view.eta is not None:
            corner = (t("home.m.left", eta=clock_text(view.eta)), _WARN)
        else:
            corner = ("", "")
        self.matrix_button.show_card(
            MATRIX_RUNNING, view.share, t("home.m.title_share", percent=int(view.share * 100)), corner, lines
        )

    # Recent selections ------------------------------------------------------------------------

    def set_selections(self, selections: list[Selection], current: str = "") -> None:
        self._selections = list(selections)
        self._current = str(current or "")
        self.selections.blockSignals(True)
        self.selections.clear()
        current_row = -1
        for selection in self._selections:
            item = QListWidgetItem(f"{selection.name}\n{self._selection_details(selection)}")
            item.setData(SELECTION_ROLE, selection.snapshot)
            item.setToolTip(selection.snapshot)
            if not Path(selection.snapshot).exists():
                item.setForeground(Qt.GlobalColor.gray)
            self.selections.addItem(item)
            if current and _same(selection.snapshot, current):
                current_row = self.selections.count() - 1
        self.selections.setCurrentRow(current_row)
        self.selections.blockSignals(False)
        self.empty_hint.setVisible(not self._selections)
        self._sync_forget()

    def _selection_details(self, selection: Selection) -> str:
        t = self._t
        parts = []
        when = _when(selection.at)
        if when:
            parts.append(when)
        parts.append(selection.errors_format.upper())
        if not Path(selection.snapshot).exists():
            parts.append(t("home.s.missing"))
        elif selection.run == RUN_DONE:
            parts.append(t("home.s.done", files=selection.written) if selection.written else t("home.s.matrix"))
            if selection.problems:
                parts.append(f"⚠ {selection.problems}")
        elif selection.run == RUN_STOPPED:
            parts.append(t("home.s.stopped", percent=int(selection.share * 100)))
        elif selection.run == RUN_FAILED:
            parts.append(t("home.s.failed"))
        else:
            parts.append(t("home.s.new"))
        return " · ".join(parts)

    def _pick_item(self, item: QListWidgetItem | None) -> None:
        if item is not None and not self._running:
            self.selectionPicked.emit(str(item.data(SELECTION_ROLE) or ""))

    def _sync_forget(self) -> None:
        self.forget_button.setEnabled(self.selections.currentItem() is not None and not self._running)

    def _forget_current(self) -> None:
        item = self.selections.currentItem()
        if item is not None:
            self.forgetRequested.emit(str(item.data(SELECTION_ROLE) or ""))


def _same(a: str, b: str) -> bool:
    return str(a).replace("\\", "/").rstrip("/").lower() == str(b).replace("\\", "/").rstrip("/").lower()
