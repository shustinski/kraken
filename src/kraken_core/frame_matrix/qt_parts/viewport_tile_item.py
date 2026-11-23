"""qt: viewport tile item."""

from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
import weakref
from PyQt6 import sip
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
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
        self._owner = weakref.ref(owner)
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.tile_cells = tile_cells
        pitch = owner.CELL_PITCH
        left = tile_x * tile_cells * pitch
        top = tile_y * tile_cells * pitch
        columns = min(tile_cells, owner.matrix_width - tile_x * tile_cells)
        rows = min(tile_cells, owner.matrix_height - tile_y * tile_cells)
        self._bounds = QRectF(left, top, columns * pitch, rows * pitch)
        self._zoom_snapshot: QPixmap | None = None
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption, True)

    def boundingRect(self) -> QRectF:  # type: ignore[override]
        return self._bounds

    def paint(  # type: ignore[override]
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        del widget
        owner = self._owner()
        if owner is None or sip.isdeleted(owner):
            return
        snapshot = self._zoom_snapshot
        if (
            snapshot is not None
            and not snapshot.isNull()
            and owner._zoom_animating
            and not owner._capturing_zoom_frame
        ):
            painter.drawPixmap(self._bounds, snapshot, QRectF(snapshot.rect()))
            return
        lod = owner.lod_level()
        if lod is MatrixLod.OVERVIEW or self.tile_cells > owner.BASE_TILE_CELLS:
            owner._paint_overview_tile(painter, self)
            return
        owner._paint_tile_cells(painter, self, lod, option.exposedRect)


@dataclass(frozen=True, slots=True)
class PreparedSelectionPaint:
    """Stored fill and frame for one selected cell. Ready to draw."""

    fill: QRectF
    border: QRectF


class _SelectionOverlayItem(QGraphicsItem):
    """Selection strokes above every tile, so a later cell cannot cover them."""

    def __init__(self, owner: FrameMatrixView) -> None:
        super().__init__()
        self._owner = weakref.ref(owner)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption, True)
        self.setZValue(1)

    def boundingRect(self) -> QRectF:  # type: ignore[override]
        owner = self._owner()
        if owner is None or sip.isdeleted(owner):
            return QRectF()
        return QRectF(0.0, 0.0, owner.matrix_width * owner.CELL_PITCH, owner.matrix_height * owner.CELL_PITCH)

    def paint(  # type: ignore[override]
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        del widget
        owner = self._owner()
        if owner is None or sip.isdeleted(owner):
            return
        paints = owner._selection_paints
        if not paints:
            return
        lod = owner.lod_level()
        if lod is MatrixLod.OVERVIEW:
            return
        from kraken_core.theme import scene_color

        exposed = option.exposedRect
        pen = QPen(QColor(scene_color("selection")), 3.0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        for item in paints:
            if exposed.intersects(item.border):
                painter.drawRect(item.border)


if TYPE_CHECKING:
    from kraken_core.frame_matrix.qt import FrameMatrixView
