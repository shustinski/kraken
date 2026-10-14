"""frame_matrix_view: layout item geometry."""
from __future__ import annotations
from pathlib import Path
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QBrush, QColor, QPixmap
from PyQt6.QtWidgets import (
    QListWidgetItem,
)
from contour.ui.item_status_painting import FRAME_STATUS_ROLE

from contour.ui.frame_matrix_view_parts.module_state import (
    _LOD_LABEL_MIN_ZOOM,
    _STATUS_MARKER_COLORS,
)

from contour.ui.frame_matrix_view_parts.trailing_frame_number import _trailing_frame_number


class LayoutItemGeometryMixin:
    def _layout_item_geometry(
        self,
        index: int,
        item: QListWidgetItem,
        group: tuple,
        *,
        region: tuple[int, int],
    ) -> None:
        background_item, status_item, selection_item, pixmap_item, text_item = group
        hidden = item.isHidden()
        in_region = self._index_in_render_region(index, region)
        for graphics_item in group:
            graphics_item.setVisible(not hidden and in_region)
        if hidden or not in_region:
            pixmap_item.setPixmap(QPixmap())
            return
        columns = max(1, self._columns)
        cell_w = max(1, self._grid_size.width())
        cell_h = max(1, self._grid_size.height())
        step_x = self._cell_step_x()
        step_y = self._cell_step_y()
        col = index % columns
        row = index // columns
        x = col * step_x
        y = row * step_y
        selected = index == self._current_row
        background_item.setRect(QRectF(x + 0.5, y + 0.5, max(1, cell_w - 1), max(1, cell_h - 1)))
        background_item.setBrush(QBrush(QColor("#111827")))
        status_color = _STATUS_MARKER_COLORS.get(str(item.data(FRAME_STATUS_ROLE) or ""))
        if status_color is not None:
            marker_size = max(4, min(8, cell_w // 8, cell_h // 6))
            status_item.setRect(QRectF(x + 2, y + 2, marker_size, marker_size))
            status_item.setBrush(QBrush(status_color))
            status_item.setData(1, True)
            status_item.show()
        else:
            status_item.setData(1, False)
            status_item.hide()
        selection_item.setRect(QRectF(x + 1, y + 1, max(1, cell_w - 2), max(1, cell_h - 2)))
        selection_item.setVisible(selected)
        pixmap = self._pixmap_for_item_lod(item)
        pixmap_item.setPixmap(pixmap)
        self._position_pixmap_item(pixmap_item, pixmap, x, y)
        text = item.text()
        if not text:
            tip = item.toolTip()
            text = Path(str(tip)).stem if tip else ""
        text = _trailing_frame_number(text)
        text_item.setText(text)
        text_item.setVisible(bool(text) and self._matrix_zoom >= _LOD_LABEL_MIN_ZOOM)
        text_item.setPos(x + 3, y + max(0, cell_h - 15))
