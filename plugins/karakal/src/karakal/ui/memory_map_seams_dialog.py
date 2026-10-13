"""«Seam diagram» window of the memory map: how the frames were sewn, for the operator to see why.

The frames as they were taken, row by row, one square each, coloured by island (frames sewn
together by bits); lines between frames show the seams that did not sew. Under it, histograms of
the seams: how far each offset is from the usual one, how many bits agreed, how many sites were
compared. A click shows the frame on the map, a double click opens it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PyQt6.QtCore import QPoint, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from ..core.memory_map import (
    PLACED_ALONE,
    PLACED_BITS,
    SEAM_AMBIGUOUS,
    SEAM_BITS,
    SEAM_EMPTY,
    SEAM_FAILED,
    SEAM_INCONSISTENT,
    MemoryMap,
    MemoryMapConfig,
    seam_statistics,
)

# Islands: distinct colours that read on the dark background (they repeat after ten islands).
ISLAND_RGB = (
    (78, 154, 232),
    (242, 142, 43),
    (89, 181, 79),
    (237, 201, 72),
    (186, 122, 201),
    (255, 120, 140),
    (118, 203, 198),
    (196, 156, 120),
    (160, 200, 90),
    (200, 200, 210),
)
NO_ISLAND_SITES = QColor(64, 70, 78)
NO_SITES = QColor(30, 34, 40)
SEAM_COLORS = {
    SEAM_EMPTY: QColor(255, 170, 60),
    SEAM_FAILED: QColor(240, 70, 70),
    SEAM_AMBIGUOUS: QColor(240, 70, 70),
    SEAM_INCONSISTENT: QColor(230, 60, 220),
}
ALONE_COLOR = QColor(240, 70, 70)
# A frame sewn off the grid of the memory's blocks (self-check).
OFF_GRID_COLOR = QColor(255, 230, 40)
SELECTED_COLOR = QColor(255, 255, 255)
BAR_COLOR = QColor(78, 154, 232)
MARK_COLOR = QColor(240, 70, 70)
MEDIAN_COLOR = QColor(237, 201, 72)


SERVICE_ISLAND = QColor(150, 156, 166)


def island_colors(mmap: MemoryMap | None) -> dict[int, QColor]:
    """Islands with main blocks take the distinct colours in turn; islands of service blocks only
    (strays of the periphery) are grey, so that the arrays do not run out of colours."""

    colors: dict[int, QColor] = {}
    turn = 0
    for island in mmap.islands if mmap is not None else ():
        if any(not mmap.blocks[b].service for b in island.blocks):
            colors[island.index] = QColor(*ISLAND_RGB[turn % len(ISLAND_RGB)])
            turn += 1
        else:
            colors[island.index] = SERVICE_ISLAND
    return colors


class FrameGrid(QWidget):
    """The frames in the order they were taken, one square each."""

    frameSelected = pyqtSignal(int)
    frameOpened = pyqtSignal(int)

    def __init__(self, translator: Callable[..., str], parent=None) -> None:
        super().__init__(parent)
        self._t = translator
        self._map: MemoryMap | None = None
        self._per_row = 1
        self._cell = 14
        self._selected = -1
        self._seams: dict[int, list] = {}
        self._colors: dict = {}
        self._off_grid: set[int] = set()
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_map(self, mmap: MemoryMap | None) -> None:
        self._map = mmap
        self._colors = island_colors(mmap)
        self._off_grid = set(mmap.misaligned) if mmap is not None else set()
        self._per_row = max(1, int(mmap.per_row)) if mmap is not None else 1
        self._seams = {}
        for seam in mmap.seams if mmap is not None else ():
            self._seams.setdefault(seam.a, []).append(seam)
            self._seams.setdefault(seam.b, []).append(seam)
        self._selected = -1
        self._resize()

    def fit_width(self, width: int) -> None:
        """Squares as big as the width allows (6 to 22 px)."""

        cell = int(np.clip((width - 2) // max(1, self._per_row), 6, 22))
        if cell != self._cell:
            self._cell = cell
            self._resize()

    def select(self, index: int) -> None:
        self._selected = int(index)
        self.update()

    def _rows(self) -> int:
        count = len(self._map.frames) if self._map is not None else 0
        return -(-count // self._per_row)

    def _resize(self) -> None:
        self.setFixedSize(self._per_row * self._cell + 1, max(1, self._rows()) * self._cell + 1)
        self.update()

    def _rect(self, index: int) -> QRectF:
        row, col = divmod(index, self._per_row)
        return QRectF(col * self._cell, row * self._cell, self._cell - 1, self._cell - 1)

    def index_at(self, point: QPoint) -> int:
        if self._map is None:
            return -1
        col, row = point.x() // self._cell, point.y() // self._cell
        index = row * self._per_row + col
        if 0 <= col < self._per_row and 0 <= index < len(self._map.frames):
            return int(index)
        return -1

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt name
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        mmap = self._map
        if mmap is None:
            painter.end()
            return
        hatch = QBrush(QColor(0, 0, 0, 150), Qt.BrushStyle.BDiagPattern)
        for index, frame in enumerate(mmap.frames):
            rect = self._rect(index)
            if frame.island >= 0:
                painter.fillRect(rect, self._colors.get(frame.island, SERVICE_ISLAND))
                if frame.method != PLACED_BITS:
                    painter.fillRect(rect, hatch)
            else:
                painter.fillRect(rect, NO_ISLAND_SITES if frame.sites else NO_SITES)
            if frame.method == PLACED_ALONE and frame.island >= 0:
                pen = QPen(ALONE_COLOR, 1.5)
                painter.setPen(pen)
                painter.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))
            if index in self._off_grid:
                painter.setPen(QPen(OFF_GRID_COLOR, 2.0))
                painter.drawRect(rect.adjusted(1.0, 1.0, -1.0, -1.0))
        # Seams that did not sew, between frames that have memory: on the edge the frames share.
        width = max(2.0, self._cell / 5.0)
        for seam in mmap.seams:
            if seam.status == SEAM_BITS:
                continue
            a, b = mmap.frames[seam.a], mmap.frames[seam.b]
            if not (a.sites and b.sites) or (a.island < 0 and b.island < 0):
                continue
            pen = QPen(SEAM_COLORS.get(seam.status, MARK_COLOR), width)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            painter.setPen(pen)
            rect = self._rect(seam.a)
            if seam.kind == "right":
                x = rect.right() + 0.5
                painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom() + 1))
            else:
                y = rect.bottom() + 0.5
                painter.drawLine(int(rect.left()), int(y), int(rect.right() + 1), int(y))
        if 0 <= self._selected < len(mmap.frames):
            painter.setPen(QPen(SELECTED_COLOR, 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self._rect(self._selected).adjusted(-1, -1, 1, 1))
        painter.end()

    def frame_text(self, index: int) -> str:
        mmap = self._map
        frame = mmap.frames[index]
        lines = [frame.frame_id]
        if frame.island >= 0:
            island = next((item for item in mmap.islands if item.index == frame.island), None)
            text = self._t("memory_map.grid.island", index=frame.island)
            if island is not None:
                text += " — " + self._t(
                    f"memory_map.placement.{island.placement}",
                    across=f"{island.uncertainty[0]:.2f}",
                    down=f"{island.uncertainty[1]:.2f}",
                )
            lines.append(text)
            if index in self._off_grid:
                lines.append(self._t("memory_map.grid.off_grid"))
        else:
            lines.append(self._t("memory_map.grid.no_island" if frame.sites else "memory_map.grid.no_sites"))
        lines.append(self._t(f"memory_map.placed.{frame.method}"))
        lines.append(self._t("memory_map.grid.sites", count=frame.sites))
        for seam in self._seams.get(index, ()):
            other = mmap.frames[seam.b if seam.a == index else seam.a]
            side = {
                ("right", True): "right",
                ("right", False): "left",
                ("down", True): "down",
                ("down", False): "up",
            }[(seam.kind, seam.a == index)]
            text = self._t(
                "memory_map.grid.seam",
                side=self._t(f"memory_map.grid.side.{side}"),
                frame=other.frame_id,
                status=self._t(f"memory_map.grid.status.{seam.status}"),
            )
            if seam.compared:
                text += " " + self._t("memory_map.grid.seam_numbers", agreement=int(round(100 * seam.agreement)), compared=seam.compared)
            lines.append(text)
        return "\n".join(lines)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt name
        index = self.index_at(event.position().toPoint())
        if index >= 0:
            QToolTip.showText(event.globalPosition().toPoint(), self.frame_text(index), self)
        else:
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt name
        index = self.index_at(event.position().toPoint())
        if index >= 0 and event.button() == Qt.MouseButton.LeftButton:
            self.select(index)
            self.frameSelected.emit(index)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt name
        index = self.index_at(event.position().toPoint())
        if index >= 0:
            self.frameOpened.emit(index)
        event.accept()


class ValueHistogram(QWidget):
    """Values in bins, with dashed limits to read them against and the median in yellow."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(220, 112)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._values = np.zeros(0)
        self._span = (0.0, 1.0)
        self._marks: tuple[float, ...] = ()
        self._digits = 1
        self._empty = ""

    def set_values(
        self, values: np.ndarray, span: tuple[float, float], *, marks: tuple[float, ...] = (), digits: int = 1, empty: str = ""
    ) -> None:
        self._values = np.asarray(values, np.float64)
        self._span = (float(span[0]), float(span[1]) if span[1] > span[0] else float(span[0]) + 1.0)
        self._marks = tuple(marks)
        self._digits = digits
        self._empty = empty
        self.update()

    def counts(self, bins: int = 30) -> np.ndarray:
        clipped = np.clip(self._values, self._span[0], self._span[1])
        return np.histogram(clipped, bins=bins, range=self._span)[0]

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt name
        painter = QPainter(self)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.base())
        text_height = self.fontMetrics().height() + 2
        area = QRectF(self.rect().adjusted(6, 6, -6, -(text_height + 4)))
        if self._values.size == 0:
            painter.setPen(palette.color(palette.ColorRole.PlaceholderText))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, self._empty)
            painter.end()
            return
        counts = self.counts()
        lo, hi = self._span
        step = area.width() / len(counts)
        peak = max(1, int(counts.max()))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(BAR_COLOR)
        for offset, count in enumerate(counts):
            if count:
                height = max(1.0, area.height() * count / peak)
                painter.drawRect(QRectF(area.left() + offset * step + 0.5, area.bottom() - height, max(1.0, step - 1.0), height))

        def x_of(value: float) -> float:
            return area.left() + (min(max(value, lo), hi) - lo) / (hi - lo) * area.width()

        painter.setPen(QPen(MARK_COLOR, 1.2, Qt.PenStyle.DashLine))
        for value in self._marks:
            painter.drawLine(int(x_of(value)), int(area.top()), int(x_of(value)), int(area.bottom()))
        painter.setPen(QPen(MEDIAN_COLOR, 1.2))
        median = float(np.median(self._values))
        painter.drawLine(int(x_of(median)), int(area.top()), int(x_of(median)), int(area.bottom()))
        painter.setPen(palette.color(palette.ColorRole.PlaceholderText))
        fmt = f"{{:.{self._digits}f}}"
        painter.drawText(QRectF(area.left(), area.bottom() + 2, 70, text_height), Qt.AlignmentFlag.AlignLeft, fmt.format(lo))
        painter.drawText(QRectF(area.right() - 70, area.bottom() + 2, 70, text_height), Qt.AlignmentFlag.AlignRight, fmt.format(hi))
        painter.drawText(
            QRectF(area.left(), area.bottom() + 2, area.width(), text_height),
            Qt.AlignmentFlag.AlignHCenter,
            f"n = {self._values.size}, ~{fmt.format(median)}",
        )
        painter.end()


class MemoryMapSeamsDialog(QDialog):
    """The seam diagram and histograms; it stays open next to the map."""

    frameSelected = pyqtSignal(int)
    frameOpened = pyqtSignal(int)

    def __init__(self, translator: Callable[..., str], mmap: MemoryMap | None, parent=None) -> None:
        super().__init__(parent)
        self._t = translator
        self.setWindowTitle(self._t("memory_map.seams_window"))
        self.setModal(False)
        self.resize(1180, 760)
        layout = QVBoxLayout(self)

        self.summary_label = QLabel("", self)
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        legend = QLabel(self._t("memory_map.grid.legend"), self)
        legend.setWordWrap(True)
        legend.setStyleSheet("color: #8a97a5;")
        layout.addWidget(legend)

        self.grid = FrameGrid(self._t, self)
        self.grid.frameSelected.connect(self.frameSelected.emit)
        self.grid.frameOpened.connect(self.frameOpened.emit)
        self.grid_scroll = QScrollArea(self)
        self.grid_scroll.setWidget(self.grid)
        self.grid_scroll.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.grid_scroll, stretch=1)

        charts = QGroupBox(self._t("memory_map.hist.title"), self)
        charts_layout = QGridLayout(charts)
        self.histograms: dict[str, ValueHistogram] = {}
        for position, key in enumerate(("right_dx", "right_dy", "agreement", "down_dx", "down_dy", "compared")):
            box = QVBoxLayout()
            title = QLabel(self._t(f"memory_map.hist.{key}"), charts)
            title.setWordWrap(True)
            box.addWidget(title)
            histogram = ValueHistogram(charts)
            box.addWidget(histogram)
            self.histograms[key] = histogram
            charts_layout.addLayout(box, position // 3, position % 3)
        note = QLabel(self._t("memory_map.hist.note"), charts)
        note.setWordWrap(True)
        note.setStyleSheet("color: #8a97a5;")
        charts_layout.addWidget(note, 2, 0, 1, 3)
        layout.addWidget(charts)
        self.set_map(mmap)

    def set_map(self, mmap: MemoryMap | None) -> None:
        self._map = mmap
        self.grid.set_map(mmap)
        self.grid.fit_width(self.grid_scroll.viewport().width())
        if mmap is None:
            self.summary_label.setText(self._t("memory_map.none"))
            for histogram in self.histograms.values():
                histogram.set_values(np.zeros(0), (0.0, 1.0))
            return
        counts = mmap.seam_counts()
        # Frames with memory on the map, by how they were placed; frames whose cells make no block
        # (strays of the periphery) apart.
        placed = {method: sum(1 for f in mmap.frames if f.method == method and f.island >= 0) for method in ("bits", "estimated", "alone")}
        self.summary_label.setText(
            self._t(
                "memory_map.seams_summary",
                islands=len(mmap.islands),
                main=sum(1 for island in mmap.islands if any(not mmap.blocks[b].service for b in island.blocks)),
                bits=counts.get(SEAM_BITS, 0),
                empty=counts.get(SEAM_EMPTY, 0),
                bad=sum(v for k, v in counts.items() if k not in (SEAM_BITS, SEAM_EMPTY)),
                by_bits=placed["bits"],
                estimated=placed["estimated"],
                alone=placed["alone"],
                stray=sum(1 for f in mmap.frames if f.sites and f.island < 0),
            )
        )
        cfg = MemoryMapConfig()
        stats = seam_statistics(mmap)
        empty = self._t("memory_map.hist.empty")
        for key in ("right_dx", "right_dy", "down_dx", "down_dy"):
            values = stats[key]
            reach = max(1.0, float(np.abs(values).max()) * 1.05) if values.size else 1.0
            # Beyond half a site a guess by the usual offset would put the cells a site off.
            self.histograms[key].set_values(values, (-reach, reach), marks=(-0.5, 0.5), digits=2, empty=empty)
        agreement = stats["agreement"]
        low = min(50.0, float(agreement.min())) if agreement.size else 50.0
        self.histograms["agreement"].set_values(agreement, (low, 100.0), marks=(100.0 * cfg.min_agreement,), digits=0, empty=empty)
        compared = stats["compared"]
        top = float(compared.max()) * 1.05 if compared.size else 1.0
        self.histograms["compared"].set_values(compared, (0.0, max(top, 2.0 * cfg.min_compared)), marks=(float(cfg.min_compared),), digits=0, empty=empty)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt name
        super().resizeEvent(event)
        self.grid.fit_width(self.grid_scroll.viewport().width())
