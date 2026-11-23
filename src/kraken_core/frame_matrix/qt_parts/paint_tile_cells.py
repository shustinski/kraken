"""qt: paint tile cells."""

from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Mapping
import math
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from kraken_core.frame_matrix.models import (
    MatrixItem,
)

from kraken_core.frame_matrix.qt_parts.module_state import _LOGGER

from kraken_core.frame_matrix.qt_parts.matrix_lod import MatrixLod

from kraken_core.frame_matrix.qt_parts.viewport_tile_item import FrameCellData


class PaintTileCellsMixin:
    def _paint_tile_cells(self, painter: QPainter, tile: _ViewportTileItem, lod: MatrixLod, exposed: QRectF) -> None:
        start_column = max(tile.tile_x * tile.tile_cells, int(math.floor(exposed.left() / self.CELL_PITCH)))
        start_row = max(tile.tile_y * tile.tile_cells, int(math.floor(exposed.top() / self.CELL_PITCH)))
        end_column = min(
            self.matrix_width, (tile.tile_x + 1) * tile.tile_cells, int(math.ceil(exposed.right() / self.CELL_PITCH))
        )
        end_row = min(
            self.matrix_height, (tile.tile_y + 1) * tile.tile_cells, int(math.ceil(exposed.bottom() / self.CELL_PITCH))
        )
        from kraken_core.theme import scene_color

        base_pen = QPen(QColor("#475569"), 1.0)
        base_pen.setCosmetic(True)
        cell_border_width = 3.0
        selection_color = QColor(scene_color("selection"))
        selection_fill = QColor(selection_color)
        selection_fill.setAlphaF(0.2)
        selection_pen = QPen(selection_color, cell_border_width)
        selection_pen.setCosmetic(True)
        for row in range(start_row, end_row):
            y = self._y_for_row(row)
            for column in range(start_column, end_column):
                x = column + 1
                cell = self._cells.get((x, y), FrameCellData(x, y))
                rect = QRectF(
                    column * self.CELL_PITCH,
                    row * self.CELL_PITCH,
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )
                painter.fillRect(rect, self._cell_visual_color(cell, self._fill_mode))
        mosaic = self._paint_loaded_mosaic(painter, tile)
        for row in range(start_row, end_row):
            y = self._y_for_row(row)
            for column in range(start_column, end_column):
                x = column + 1
                cell = self._cells.get((x, y), FrameCellData(x, y))
                payload = cell.payload if isinstance(cell.payload, Mapping) else {}
                rect = QRectF(
                    column * self.CELL_PITCH,
                    row * self.CELL_PITCH,
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )
                if (
                    not mosaic
                    and self._fill_mode == "thumbnail"
                    and lod in {MatrixLod.CELLS, MatrixLod.DETAILS}
                    and cell.thumbnail is not None
                    and not cell.thumbnail.isNull()
                ):
                    painter.save()
                    painter.setOpacity(0.92)
                    painter.drawPixmap(rect.toRect(), cell.thumbnail)
                    painter.restore()

                selected = self._selection.contains(x, y)
                if selected:
                    painter.fillRect(rect, selection_fill)
                border_color = self._cell_visual_color(cell, self._border_mode)
                semantic_pen = QPen(border_color if border_color.isValid() else base_pen.color(), cell_border_width)
                semantic_pen.setCosmetic(True)
                painter.setPen(semantic_pen)
                painter.drawRect(rect)
                if selected:
                    painter.setPen(selection_pen)
                    painter.drawRect(rect)

                if lod is MatrixLod.DETAILS:
                    painter.setPen(QColor(scene_color("label")))
                    frame_number = (y - 1) * self.matrix_width + x
                    coordinate_text = cell.label or f"{frame_number:04d}"
                    painter.drawText(
                        rect.adjusted(4.0, 3.0, -4.0, -3.0),
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                        coordinate_text,
                    )
                    if cell.performer_initials:
                        painter.drawText(
                            rect.adjusted(4.0, 3.0, -4.0, -3.0),
                            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                            cell.performer_initials,
                        )
                renderers = getattr(self, "_renderers", ())
                if renderers:
                    matrix_item = MatrixItem(
                        key=str(payload.get("key") or f"{x}:{y}"),
                        x=x,
                        y=y,
                        status=cell.status,
                        label=cell.label,
                        tooltip=cell.tooltip,
                        metadata=dict(payload),
                    )
                    context = {
                        "painter": painter,
                        "rect": rect,
                        "lod": lod.value,
                        "selected": selected,
                        "semantic_lod": (
                            self.semantic_lod().value if callable(getattr(self, "semantic_lod", None)) else lod.value
                        ),
                    }
                    for renderer in renderers:
                        try:
                            renderer.render(matrix_item, context)
                        except Exception:
                            _LOGGER.exception(
                                "Matrix renderer %r failed for cell (%s, %s)",
                                renderer,
                                x,
                                y,
                            )
                            continue

    def _paint_loaded_mosaic(self, painter: QPainter, tile: _ViewportTileItem) -> bool:
        """Draw the exact tile, or the previous level until that tile arrives."""

        pieces = self._mosaic_pieces(tile)
        if not pieces:
            return False
        for pixmap, source, dest in pieces:
            painter.drawPixmap(dest, pixmap, source)
        return True

    def _mosaic_pieces(self, tile: _ViewportTileItem) -> tuple[tuple[QPixmap, QRectF, QRectF], ...]:
        images = getattr(self, "_mosaic_images", None)
        revision = getattr(self, "_mosaic_revision", None)
        if not images or revision is None:
            return ()
        orientation = self.orientation().value
        exact = self._stored_mosaic(tile.tile_x, tile.tile_y, tile.tile_cells, revision, orientation)
        if exact is not None:
            target = tile.boundingRect()
            return ((exact, QRectF(0, 0, exact.width(), exact.height()), target),)
        coarser = self._coarser_mosaic_piece(tile, revision, orientation)
        if coarser is not None:
            return (coarser,)
        return self._finer_mosaic_pieces(tile, revision, orientation)

    def _stored_mosaic(
        self,
        tile_x: int,
        tile_y: int,
        cells: int,
        revision: object,
        orientation: str,
    ) -> QPixmap | None:
        images = self._mosaic_images
        found = images.get((tile_x, tile_y, cells, revision, orientation))
        if found is not None and not found.isNull():
            return found
        for key, pixmap in images.items():
            if (
                len(key) >= 5
                and key[0] == tile_x
                and key[1] == tile_y
                and key[2] == cells
                and key[4] == orientation
                and pixmap is not None
                and not pixmap.isNull()
            ):
                return pixmap
        return None

    def _coarser_mosaic_piece(
        self,
        tile: _ViewportTileItem,
        revision: object,
        orientation: str,
    ) -> tuple[QPixmap, QRectF, QRectF] | None:
        cells = tile.tile_cells * 2
        limit = max(self.matrix_width, self.matrix_height, cells)
        while cells <= limit:
            parent_x = (tile.tile_x * tile.tile_cells) // cells
            parent_y = (tile.tile_y * tile.tile_cells) // cells
            pixmap = self._stored_mosaic(parent_x, parent_y, cells, revision, orientation)
            if pixmap is not None:
                source = _source_in_parent(tile, cells, pixmap, self.matrix_width, self.matrix_height)
                return pixmap, source, tile.boundingRect()
            cells *= 2
        return None

    def _finer_mosaic_pieces(
        self,
        tile: _ViewportTileItem,
        revision: object,
        orientation: str,
    ) -> tuple[tuple[QPixmap, QRectF, QRectF], ...]:
        finer = tile.tile_cells // 2
        target = tile.boundingRect()
        while finer >= self.BASE_TILE_CELLS:
            scale = tile.tile_cells // finer
            pieces: list[tuple[QPixmap, QRectF, QRectF]] = []
            for step_y in range(scale):
                for step_x in range(scale):
                    child_x = tile.tile_x * scale + step_x
                    child_y = tile.tile_y * scale + step_y
                    pixmap = self._stored_mosaic(child_x, child_y, finer, revision, orientation)
                    if pixmap is None:
                        continue
                    pieces.append(
                        (
                            pixmap,
                            QRectF(0, 0, pixmap.width(), pixmap.height()),
                            _child_destination(tile, child_x, child_y, finer, target, self.matrix_width, self.matrix_height),
                        )
                    )
            if pieces:
                return tuple(pieces)
            finer //= 2
        return ()


def _source_in_parent(
    tile: _ViewportTileItem,
    parent_cells: int,
    pixmap: QPixmap,
    matrix_width: int,
    matrix_height: int,
) -> QRectF:
    parent_x = (tile.tile_x * tile.tile_cells) // parent_cells
    parent_y = (tile.tile_y * tile.tile_cells) // parent_cells
    origin_column = parent_x * parent_cells
    origin_row = parent_y * parent_cells
    parent_columns = min(parent_cells, matrix_width - origin_column)
    parent_rows = min(parent_cells, matrix_height - origin_row)
    column = tile.tile_x * tile.tile_cells
    row = tile.tile_y * tile.tile_cells
    columns = min(tile.tile_cells, matrix_width - column)
    rows = min(tile.tile_cells, matrix_height - row)
    return QRectF(
        (column - origin_column) / parent_columns * pixmap.width(),
        (row - origin_row) / parent_rows * pixmap.height(),
        columns / parent_columns * pixmap.width(),
        rows / parent_rows * pixmap.height(),
    )


def _child_destination(
    tile: _ViewportTileItem,
    child_x: int,
    child_y: int,
    child_cells: int,
    target: QRectF,
    matrix_width: int,
    matrix_height: int,
) -> QRectF:
    origin_column = tile.tile_x * tile.tile_cells
    origin_row = tile.tile_y * tile.tile_cells
    columns = min(tile.tile_cells, matrix_width - origin_column)
    rows = min(tile.tile_cells, matrix_height - origin_row)
    column = child_x * child_cells
    row = child_y * child_cells
    child_columns = min(child_cells, matrix_width - column)
    child_rows = min(child_cells, matrix_height - row)
    return QRectF(
        target.left() + (column - origin_column) / columns * target.width(),
        target.top() + (row - origin_row) / rows * target.height(),
        child_columns / columns * target.width(),
        child_rows / rows * target.height(),
    )


if TYPE_CHECKING:
    from kraken_core.frame_matrix.qt_parts.viewport_tile_item import _ViewportTileItem
