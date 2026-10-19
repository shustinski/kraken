"""qt: viewport tile item."""
from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QPainter, QPixmap
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QStyleOptionGraphicsItem,
    QWidget,
)

from kraken_core.frame_matrix.qt_parts.frame_rect import FrameRect

from kraken_core.frame_matrix.qt_parts.matrix_lod import MatrixLod

@dataclass(frozen=True, slots=True)
class FrameSelection:
    """Compact selection represented by rectangles, not millions of IDs."""

    rectangles: tuple[FrameRect, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rectangles", tuple(self.rectangles))

    @classmethod
    def single(cls, x: int, y: int) -> FrameSelection:
        return cls((FrameRect(x, y, x, y),))

    @property
    def is_empty(self) -> bool:
        return not self.rectangles

    def contains(self, x: int, y: int) -> bool:
        return any(rectangle.contains(x, y) for rectangle in self.rectangles)

    def coordinates(self, *, maximum: int | None = None) -> Iterator[tuple[int, int]]:
        """Iterate unique coordinates, optionally protecting callers from expansion."""
        seen: set[tuple[int, int]] = set()
        for rectangle in self.rectangles:
            for coordinate in rectangle.coordinates():
                if coordinate in seen:
                    continue
                seen.add(coordinate)
                if maximum is not None and len(seen) > maximum:
                    raise ValueError(f"selection contains more than {maximum} frames")
                yield coordinate

@dataclass(frozen=True, slots=True)
class FrameCellData:
    """Presentation data for one materialized layer/frame state."""

    x: int
    y: int
    status: str = "empty"
    performer_color: str | None = None
    performer_initials: str = ""
    label: str = ""
    tooltip: str = ""
    thumbnail: QPixmap | None = field(default=None, compare=False, repr=False)
    payload: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.x) < 1 or int(self.y) < 1:
            raise ValueError("frame coordinates are one-based and must be positive")
        object.__setattr__(self, "x", int(self.x))
        object.__setattr__(self, "y", int(self.y))
        object.__setattr__(self, "status", str(self.status or "empty"))

@dataclass(frozen=True, slots=True)
class FrameContext:
    """Value emitted for a frame context menu request."""

    x: int
    y: int
    cell: FrameCellData
    selection: FrameSelection

class _ViewportTileItem(QGraphicsItem):
    """One visible tile; it paints either an aggregate or its individual cells."""

    def __init__(self, owner: FrameMatrixView, tile_x: int, tile_y: int, tile_cells: int) -> None:
        super().__init__()
        self.owner = owner
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.tile_cells = tile_cells
        pitch = owner.CELL_PITCH
        left = tile_x * tile_cells * pitch
        top = tile_y * tile_cells * pitch
        columns = min(tile_cells, owner.matrix_width - tile_x * tile_cells)
        rows = min(tile_cells, owner.matrix_height - tile_y * tile_cells)
        self._bounds = QRectF(left, top, columns * pitch, rows * pitch)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def boundingRect(self) -> QRectF:  # type: ignore[override]
        return self._bounds

    def paint(  # type: ignore[override]
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        del option, widget
        lod = self.owner.lod_level()
        if lod is MatrixLod.OVERVIEW or self.tile_cells > self.owner.BASE_TILE_CELLS:
            self.owner._paint_overview_tile(painter, self)
            return
        self.owner._paint_tile_cells(painter, self, lod)

if TYPE_CHECKING:
    from kraken_core.frame_matrix.qt import FrameMatrixView
