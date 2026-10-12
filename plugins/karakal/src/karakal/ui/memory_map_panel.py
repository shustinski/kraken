"""«Memory map» tab of the memory work mode: build the map on request, look at it, export it.

The panel only shows and asks; the controller (app/memory_map_controller.py) builds the map in
a thread, keeps it in the tab state and writes the files.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.memory_map import (
    CONFLICT,
    DEFECT,
    EMPTY,
    MODE_PAIR,
    ONE,
    OUTSIDE,
    PARTIAL,
    SEAM_BITS,
    SEAM_EMPTY,
    STATE_NAMES,
    ZERO,
    MemoryMap,
)

# Map colours (RGB). EMPTY is amber with both networks (a missing cell) and grey with one.
STATE_RGB = {
    OUTSIDE: (32, 36, 42),
    EMPTY: (239, 159, 39),
    ONE: (55, 138, 221),
    ZERO: (151, 196, 89),
    CONFLICT: (226, 75, 74),
    DEFECT: (212, 83, 126),
    PARTIAL: (128, 128, 128),
}
SINGLE_EMPTY_RGB = (96, 100, 106)
SERVICE_COLOR = QColor(255, 176, 64)
BLOCK_COLOR = QColor(200, 210, 222, 160)
FRAME_COLOR = QColor(255, 216, 74, 140)
SEAM_COLOR = QColor(255, 80, 80)
LEGEND_STATES = (ONE, ZERO, EMPTY, CONFLICT, DEFECT, PARTIAL)


class MemoryMapCanvas(QGraphicsView):
    """The field, one scene unit per site: wheel zooms under the mouse, drag pans."""

    hovered = pyqtSignal(int, int)
    activated = pyqtSignal(int, int)
    left = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor(*STATE_RGB[OUTSIDE]))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setMouseTracking(True)
        self._pixmap = QGraphicsPixmapItem()
        self._pixmap.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self.scene().addItem(self._pixmap)
        self._overlays: list = []
        self._marker: QGraphicsRectItem | None = None
        self._size = (0, 0)

    def set_image(self, image: QImage, *, fit: bool) -> None:
        self._pixmap.setPixmap(QPixmap.fromImage(image))
        self._size = (image.height(), image.width())
        self.scene().setSceneRect(QRectF(0, 0, image.width(), image.height()))
        if fit:
            self.fit()

    def fit(self) -> None:
        if self._size[1]:
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def clear_overlays(self) -> None:
        for item in self._overlays:
            self.scene().removeItem(item)
        self._overlays = []

    def add_rect(self, rect: QRectF, color: QColor, *, dashed: bool = False, width: float = 1.0, label: str = "") -> None:
        pen = QPen(color, width, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
        pen.setCosmetic(True)
        item = self.scene().addRect(rect, pen, QBrush(Qt.BrushStyle.NoBrush))
        item.setZValue(5)
        self._overlays.append(item)
        if label:
            text = QGraphicsSimpleTextItem(label)
            text.setBrush(QBrush(color))
            text.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            text.setPos(rect.topLeft() + QPointF(0.5, 0.5))
            text.setZValue(6)
            self.scene().addItem(text)
            self._overlays.append(text)

    def mark(self, rect: QRectF | None) -> None:
        """A red frame around something to look at (a seam, a block); None removes it."""

        if self._marker is not None:
            self.scene().removeItem(self._marker)
            self._marker = None
        if rect is None:
            return
        pen = QPen(SEAM_COLOR, 2.0)
        pen.setCosmetic(True)
        self._marker = self.scene().addRect(rect, pen, QBrush(Qt.BrushStyle.NoBrush))
        self._marker.setZValue(10)

    def show_rect(self, rect: QRectF) -> None:
        margin = max(rect.width(), rect.height(), 20.0)
        self.fitInView(rect.adjusted(-margin, -margin, margin, margin), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt name
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        event.accept()

    def _cell(self, event) -> tuple[int, int] | None:
        point = self.mapToScene(event.position().toPoint())
        row, col = int(np.floor(point.y())), int(np.floor(point.x()))
        if 0 <= row < self._size[0] and 0 <= col < self._size[1]:
            return row, col
        return None

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt name
        cell = self._cell(event)
        if cell is None:
            self.left.emit()
        else:
            self.hovered.emit(*cell)
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt name
        cell = self._cell(event)
        if cell is not None:
            self.activated.emit(*cell)
        event.accept()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt name
        self.left.emit()
        super().leaveEvent(event)


def map_image(mmap: MemoryMap, shown: set[int]) -> QImage:
    """RGB picture of the field: hidden states as background, service blocks dimmed."""

    table = np.zeros((max(STATE_NAMES) + 1, 3), np.uint8)
    background = np.array(STATE_RGB[OUTSIDE], np.uint8)
    for code, rgb in STATE_RGB.items():
        if code == EMPTY and mmap.mode != MODE_PAIR:
            rgb = SINGLE_EMPTY_RGB
        table[code] = rgb if (code == OUTSIDE or code in shown) else background
    rgb = table[mmap.field]
    service = [b.index for b in mmap.blocks if b.service]
    if service and mmap.block_index.size:
        dim = np.isin(mmap.block_index, service)
        rgb[dim] = (rgb[dim].astype(np.uint16) + background) // 2
    rgb = np.ascontiguousarray(rgb)
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()


class MemoryMapPanel(QWidget):
    buildRequested = pyqtSignal(int)
    cancelRequested = pyqtSignal()
    exportRequested = pyqtSignal()
    # A field site to open in the frame window (row, column).
    frameRequested = pyqtSignal(int, int)
    # The operator's role of a block: (block index, service).
    blockRoleChanged = pyqtSignal(int, bool)

    def __init__(self, translator: Callable[..., str], parent=None) -> None:
        super().__init__(parent)
        self._t = translator
        self._map: MemoryMap | None = None
        self._shown: set[int] = set(LEGEND_STATES)
        self._block_lines = True
        self._frame_lines = False
        self._filling_blocks = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 4, 0, 0)
        root.setSpacing(6)

        bar = QHBoxLayout()
        self.build_button = QPushButton(self._t("memory_map.build"), self)
        self.build_button.setToolTip(self._t("memory_map.build_tooltip"))
        self.build_button.clicked.connect(lambda: self.buildRequested.emit(int(self.per_row_spin.value())))
        bar.addWidget(self.build_button)
        self.stop_button = QPushButton(self._t("memory_map.stop"), self)
        self.stop_button.clicked.connect(self.cancelRequested.emit)
        self.stop_button.hide()
        bar.addWidget(self.stop_button)
        bar.addWidget(QLabel(self._t("memory_map.per_row"), self))
        self.per_row_spin = QSpinBox(self)
        self.per_row_spin.setRange(1, 100_000)
        self.per_row_spin.setToolTip(self._t("memory_map.per_row_tooltip"))
        bar.addWidget(self.per_row_spin)
        self.progress = QProgressBar(self)
        self.progress.setMaximumWidth(220)
        self.progress.hide()
        bar.addWidget(self.progress)
        self.status_label = QLabel("", self)
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(self.status_label, stretch=1)
        self.export_button = QPushButton(self._t("memory_map.export"), self)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.exportRequested.emit)
        bar.addWidget(self.export_button)
        root.addLayout(bar)
        self.sources_label = QLabel("", self)
        self.sources_label.setWordWrap(True)
        root.addWidget(self.sources_label)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        left = QWidget(splitter)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = MemoryMapCanvas(left)
        self.canvas.hovered.connect(self._on_hover)
        self.canvas.left.connect(lambda: self.hover_label.setText(self._t("memory_map.hover_hint")))
        self.canvas.activated.connect(self.frameRequested.emit)
        left_layout.addWidget(self.canvas, stretch=1)
        legend = QHBoxLayout()
        self.legend_checks: dict[object, QCheckBox] = {}
        for code in LEGEND_STATES:
            check = QCheckBox(self)
            check.setChecked(True)
            check.toggled.connect(lambda on, code=code: self._toggle_state(code, on))
            self.legend_checks[code] = check
            legend.addWidget(check)
        for key, default in (("blocks", True), ("frames", False)):
            check = QCheckBox(self._t(f"memory_map.show_{key}"), self)
            check.setChecked(default)
            check.toggled.connect(lambda on, key=key: self._toggle_lines(key, on))
            self.legend_checks[key] = check
            legend.addWidget(check)
        fit = QPushButton(self._t("memory_map.fit"), self)
        fit.clicked.connect(self.canvas.fit)
        legend.addStretch(1)
        legend.addWidget(fit)
        left_layout.addLayout(legend)
        self.hover_label = QLabel(self._t("memory_map.hover_hint"), self)
        left_layout.addWidget(self.hover_label)
        splitter.addWidget(left)

        # The side panel scrolls rather than squeezing its parts over each other in a low window.
        side_scroll = QScrollArea(splitter)
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setMinimumWidth(380)
        side_scroll.setMaximumWidth(520)
        side = QWidget(side_scroll)
        side_scroll.setWidget(side)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        summary = QGroupBox(self._t("memory_map.summary"), side)
        self.summary_form = QFormLayout(summary)
        side_layout.addWidget(summary)
        blocks = QGroupBox(self._t("memory_map.blocks"), side)
        blocks_layout = QVBoxLayout(blocks)
        self.blocks_table = QTableWidget(0, 4, blocks)
        self.blocks_table.setHorizontalHeaderLabels(
            [self._t("memory_map.block_no"), self._t("memory_map.block_place"), self._t("memory_map.block_sites"), self._t("memory_map.block_role")]
        )
        self.blocks_table.verticalHeader().hide()
        self.blocks_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.blocks_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.blocks_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.blocks_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.blocks_table.cellClicked.connect(lambda row, _col: self._show_block(row))
        self.blocks_table.setMinimumHeight(200)
        self.blocks_table.horizontalHeader().setStretchLastSection(True)
        blocks_layout.addWidget(self.blocks_table)
        self.blocks_table.setToolTip(self._t("memory_map.service_hint"))
        hint = QLabel(self._t("memory_map.service_short"), blocks)
        hint.setWordWrap(True)
        hint.setToolTip(self._t("memory_map.service_hint"))
        blocks_layout.addWidget(hint)
        side_layout.addWidget(blocks, stretch=2)
        seams = QGroupBox(self._t("memory_map.seams"), side)
        seams_layout = QVBoxLayout(seams)
        self.seams_label = QLabel("", seams)
        self.seams_label.setWordWrap(True)
        seams_layout.addWidget(self.seams_label)
        self.seams_list = QListWidget(seams)
        self.seams_list.currentRowChanged.connect(self._show_seam)
        self.seams_list.setMinimumHeight(140)
        seams_layout.addWidget(self.seams_list)
        side_layout.addWidget(seams, stretch=1)
        splitter.addWidget(side_scroll)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)
        self._update_legend_texts()

    # --- From the controller -------------------------------------------------------------------

    def set_sources(self, text: str, can_build: bool, reason: str = "") -> None:
        self.sources_label.setText(text)
        self.build_button.setEnabled(can_build)
        self.build_button.setToolTip(reason or self._t("memory_map.build_tooltip"))

    def set_per_row(self, value: int) -> None:
        self.per_row_spin.setValue(max(1, int(value)))

    def set_busy(self, stage: str, done: int, total: int) -> None:
        self.build_button.hide()
        self.stop_button.show()
        self.export_button.setEnabled(False)
        self.progress.show()
        self.progress.setMaximum(max(1, int(total)))
        self.progress.setValue(int(done))
        self.status_label.setText(self._t(f"memory_map.stage.{stage}", done=done, total=total))

    def set_idle(self, message: str = "") -> None:
        self.stop_button.hide()
        self.build_button.show()
        self.progress.hide()
        self.export_button.setEnabled(self._map is not None)
        self.status_label.setText(message)

    def set_stale(self, stale: bool) -> None:
        if stale and self._map is not None:
            self.status_label.setText(self._t("memory_map.stale"))

    def set_map(self, mmap: MemoryMap | None, *, fit: bool = True) -> None:
        self._map = mmap
        self.export_button.setEnabled(mmap is not None)
        self._update_legend_texts()
        if mmap is None:
            self.canvas.set_image(QImage(1, 1, QImage.Format.Format_RGB888), fit=True)
            self.canvas.clear_overlays()
            self.blocks_table.setRowCount(0)
            self.seams_list.clear()
            self.seams_label.setText("")
            self._fill_summary()
            return
        self._render(fit=fit)
        self._fill_summary()
        self._fill_blocks()
        self._fill_seams()

    # --- Drawing -------------------------------------------------------------------------------

    def _render(self, *, fit: bool = False) -> None:
        mmap = self._map
        if mmap is None:
            return
        self.canvas.set_image(map_image(mmap, self._shown), fit=fit)
        self.canvas.clear_overlays()
        if self._block_lines:
            for block in mmap.blocks:
                rect = QRectF(block.col, block.row, block.cols, block.rows)
                if block.service:
                    # No caption: small service blocks sit side by side and captions would overlap.
                    self.canvas.add_rect(rect, SERVICE_COLOR, dashed=True, width=1.5)
                else:
                    self.canvas.add_rect(rect, BLOCK_COLOR)
        if self._frame_lines:
            for frame in mmap.frames:
                x0 = (frame.x - mmap.origin[0]) / mmap.pitch.x + 0.5
                y0 = (frame.y - mmap.origin[1]) / mmap.pitch.y + 0.5
                self.canvas.add_rect(QRectF(x0, y0, frame.width / mmap.pitch.x, frame.height / mmap.pitch.y), FRAME_COLOR)

    def _toggle_state(self, code: int, on: bool) -> None:
        (self._shown.add if on else self._shown.discard)(code)
        self._render()

    def _toggle_lines(self, key: str, on: bool) -> None:
        if key == "blocks":
            self._block_lines = on
        else:
            self._frame_lines = on
        self._render()

    def _update_legend_texts(self) -> None:
        mode = self._map.mode if self._map is not None else MODE_PAIR
        single = mode != MODE_PAIR
        # States a map of these networks can have: with one network there are no zeros (or ones)
        # of the other kind and no conflicts.
        offered = {
            MODE_PAIR: set(LEGEND_STATES),
            "ones": {ONE, EMPTY, DEFECT, PARTIAL},
            "zeros": {ZERO, EMPTY, DEFECT, PARTIAL},
        }.get(mode, set(LEGEND_STATES))
        for code in LEGEND_STATES:
            check = self.legend_checks[code]
            key = f"memory_map.state.{STATE_NAMES[code]}" + ("_single" if single and code == EMPTY else "")
            rgb = SINGLE_EMPTY_RGB if single and code == EMPTY else STATE_RGB[code]
            check.setText(self._t(key))
            check.setStyleSheet(f"QCheckBox::indicator:checked {{ background-color: rgb{rgb}; border: 1px solid #aab4c0; }}")
            check.setVisible(code in offered)

    # --- Side panel ----------------------------------------------------------------------------

    def _fill_summary(self) -> None:
        while self.summary_form.rowCount():
            self.summary_form.removeRow(0)
        mmap = self._map
        if mmap is None:
            self.summary_form.addRow(QLabel(self._t("memory_map.none"), self))
            return
        counts = mmap.counts()
        rows, cols = mmap.shape
        lines = [
            (self._t("memory_map.networks"), self._t(f"memory_map.mode.{mmap.mode}")),
            (self._t("memory_map.field"), self._t("memory_map.field_value", rows=rows, cols=cols)),
            (self._t("memory_map.state.one"), f"{counts.get('one', 0):,}".replace(",", " ")),
        ]
        if mmap.mode == MODE_PAIR:
            lines.append((self._t("memory_map.state.zero"), f"{counts.get('zero', 0):,}".replace(",", " ")))
            lines.append((self._t("memory_map.state.empty"), f"{counts.get('empty', 0):,}".replace(",", " ")))
            lines.append((self._t("memory_map.state.conflict"), f"{counts.get('conflict', 0):,}".replace(",", " ")))
        else:
            if mmap.mode == "zeros":
                lines[2] = (self._t("memory_map.state.zero"), f"{counts.get('zero', 0):,}".replace(",", " "))
            lines.append((self._t("memory_map.state.empty_single"), f"{counts.get('empty', 0):,}".replace(",", " ")))
        lines.append((self._t("memory_map.state.defect"), f"{counts.get('defect', 0):,}".replace(",", " ")))
        lines.append((self._t("memory_map.disagreements"), str(mmap.disagreements)))
        for title, value in lines:
            label = QLabel(value, self)
            label.setWordWrap(True)
            self.summary_form.addRow(title, label)
        if mmap.mode != MODE_PAIR:
            note = QLabel(self._t("memory_map.single_note"), self)
            note.setWordWrap(True)
            self.summary_form.addRow(note)

    def _fill_blocks(self) -> None:
        mmap = self._map
        self._filling_blocks = True
        self.blocks_table.setRowCount(len(mmap.blocks) if mmap else 0)
        for row, block in enumerate(mmap.blocks if mmap else ()):
            self.blocks_table.setItem(row, 0, QTableWidgetItem(str(block.index)))
            self.blocks_table.setItem(
                row, 1, QTableWidgetItem(self._t("memory_map.block_place_value", row=block.row, col=block.col, rows=block.rows, cols=block.cols))
            )
            self.blocks_table.setItem(row, 2, QTableWidgetItem(f"{block.sites:,}".replace(",", " ")))
            combo = QComboBox(self.blocks_table)
            combo.addItem(self._t("memory_map.role.main"), False)
            combo.addItem(self._t("memory_map.role.service"), True)
            combo.setCurrentIndex(1 if block.service else 0)
            combo.currentIndexChanged.connect(
                lambda _i, combo=combo, index=block.index: self._role_changed(index, bool(combo.currentData()))
            )
            self.blocks_table.setCellWidget(row, 3, combo)
        self._filling_blocks = False

    def _role_changed(self, index: int, service: bool) -> None:
        if not self._filling_blocks:
            self.blockRoleChanged.emit(index, service)

    def _show_block(self, row: int) -> None:
        mmap = self._map
        if mmap is None or not 0 <= row < len(mmap.blocks):
            return
        block = mmap.blocks[row]
        rect = QRectF(block.col, block.row, block.cols, block.rows)
        self.canvas.mark(rect)
        self.canvas.show_rect(rect)

    def _fill_seams(self) -> None:
        mmap = self._map
        self.seams_list.clear()
        counts = mmap.seam_counts()
        self.seams_label.setText(
            self._t(
                "memory_map.seams_value",
                bits=counts.get(SEAM_BITS, 0),
                empty=counts.get(SEAM_EMPTY, 0),
                bad=sum(v for k, v in counts.items() if k not in (SEAM_BITS, SEAM_EMPTY)),
                estimated=sum(1 for f in mmap.frames if f.method != SEAM_BITS),
            )
        )
        for seam in mmap.seams:
            if seam.status in (SEAM_BITS, SEAM_EMPTY):
                continue
            a, b = mmap.frames[seam.a], mmap.frames[seam.b]
            item = QListWidgetItem(
                self._t(
                    "memory_map.seam_item",
                    a=a.frame_id,
                    b=b.frame_id,
                    kind=self._t(f"memory_map.seam_kind.{seam.kind}"),
                    status=self._t(f"memory_map.seam_status.{seam.status}"),
                    agreement=int(round(100 * seam.agreement)),
                )
            )
            item.setData(Qt.ItemDataRole.UserRole, (seam.a, seam.b))
            item.setToolTip(item.text())
            self.seams_list.addItem(item)

    def _show_seam(self, row: int) -> None:
        mmap = self._map
        item = self.seams_list.item(row) if row >= 0 else None
        if mmap is None or item is None:
            self.canvas.mark(None)
            return
        a, b = item.data(Qt.ItemDataRole.UserRole)
        fa, fb = mmap.frames[a], mmap.frames[b]
        x0 = (min(fa.x, fb.x) - mmap.origin[0]) / mmap.pitch.x
        y0 = (min(fa.y, fb.y) - mmap.origin[1]) / mmap.pitch.y
        x1 = (max(fa.x + fa.width, fb.x + fb.width) - mmap.origin[0]) / mmap.pitch.x
        y1 = (max(fa.y + fa.height, fb.y + fb.height) - mmap.origin[1]) / mmap.pitch.y
        rect = QRectF(x0, y0, x1 - x0, y1 - y0)
        self.canvas.mark(rect)
        self.canvas.show_rect(rect)

    def _on_hover(self, row: int, col: int) -> None:
        mmap = self._map
        if mmap is None:
            return
        state = int(mmap.field[row, col])
        block = int(mmap.block_index[row, col]) if mmap.block_index.size else -1
        point = mmap.frame_point(row, col)
        name = STATE_NAMES[state]
        key = f"memory_map.state.{name}" + ("_single" if state == EMPTY and mmap.mode != MODE_PAIR else "")
        parts = [self._t("memory_map.hover_site", row=row, col=col), self._t(key) if state != OUTSIDE else self._t("memory_map.state.outside")]
        if block >= 0:
            service = mmap.blocks[block].service
            parts.append(self._t("memory_map.hover_block", index=block) + (f" ({self._t('memory_map.service_mark')})" if service else ""))
        if point is not None:
            parts.append(self._t("memory_map.hover_frame", frame=mmap.frames[point[0]].frame_id))
        self.hover_label.setText(" · ".join(parts))
