"""qt: paint tile cells."""
from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Mapping
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from kraken_core.frame_matrix.models import (
    MatrixItem,
)

from kraken_core.frame_matrix.qt_parts.module_state import _LOGGER

from kraken_core.frame_matrix.qt_parts.matrix_lod import MatrixLod

from kraken_core.frame_matrix.qt_parts.viewport_tile_item import FrameCellData


class PaintTileCellsMixin:
    def _paint_tile_cells(self, painter: QPainter, tile: _ViewportTileItem, lod: MatrixLod) -> None:
        start_column = tile.tile_x * tile.tile_cells
        start_row = tile.tile_y * tile.tile_cells
        end_column = min(self.matrix_width, start_column + tile.tile_cells)
        end_row = min(self.matrix_height, start_row + tile.tile_cells)
        base_pen = QPen(QColor("#475569"), 1.0)
        base_pen.setCosmetic(True)
        selection_pen = QPen(QColor("#f8fafc"), 2.2)
        selection_pen.setCosmetic(True)
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
                fill_color = self._cell_visual_color(cell, self._fill_mode)
                painter.fillRect(rect, fill_color)
                if (
                    self._fill_mode == "thumbnail"
                    and lod in {MatrixLod.CELLS, MatrixLod.DETAILS}
                    and cell.thumbnail is not None
                    and not cell.thumbnail.isNull()
                ):
                    painter.save()
                    painter.setOpacity(0.92)
                    painter.drawPixmap(rect.toRect(), cell.thumbnail)
                    painter.restore()

                border_color = self._cell_visual_color(cell, self._border_mode)
                semantic_pen = QPen(border_color if border_color.isValid() else base_pen.color(), 3.0)
                semantic_pen.setCosmetic(True)
                painter.setPen(semantic_pen)
                painter.drawRect(rect)
                if self._selection.contains(x, y):
                    painter.setPen(selection_pen)
                    painter.drawRect(rect.adjusted(-1.5, -1.5, 1.5, 1.5))

                if lod is MatrixLod.DETAILS:
                    painter.setPen(QColor("#f8fafc"))
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
                        "selected": self._selection.contains(x, y),
                        "semantic_lod": (
                            self.semantic_lod().value
                            if callable(getattr(self, "semantic_lod", None))
                            else lod.value
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

if TYPE_CHECKING:
    from kraken_core.frame_matrix.qt_parts.viewport_tile_item import _ViewportTileItem
