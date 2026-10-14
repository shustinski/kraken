"""frame_matrix_view: cover crop pixmap cell."""
from __future__ import annotations
from PyQt6.QtCore import QRect, QSize
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QGraphicsPixmapItem,
    QListWidgetItem,
)
from ...graphics.viewport_navigation import (
    MAX_ZOOM_FACTOR,
    MIN_ZOOM_FACTOR,
)
from contour.ui.large_dataset import clamp_thumbnail_source_size

from contour.ui.frame_matrix_view_parts.module_state import (
    _FRAME_ID_ROLE,
    _THUMBNAIL_LOD_LEVELS,
)


class CoverCropPixmapCellMixin:
    def _frame_id_for_item(self, item: QListWidgetItem) -> int:
        value = item.data(_FRAME_ID_ROLE)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
        try:
            return self._items.index(item)
        except ValueError:
            return -1

    def _navigator_lod(self) -> int:
        if not self._navigator_lods:
            return 0
        if len(self._navigator_lods) == 1:
            return self._navigator_lods[0]
        zoom = max(MIN_ZOOM_FACTOR, min(MAX_ZOOM_FACTOR, float(self._matrix_zoom)))
        if zoom >= 1.0:
            return self._navigator_lods[0]
        if zoom >= 0.35 or len(self._navigator_lods) == 2:
            return self._navigator_lods[min(1, len(self._navigator_lods) - 1)]
        return self._navigator_lods[-1]

    def _thumbnail_lod(self) -> float:
        effective = max(MIN_ZOOM_FACTOR, min(MAX_ZOOM_FACTOR, self._matrix_zoom))
        for level in _THUMBNAIL_LOD_LEVELS:
            if effective <= level:
                return level
        return _THUMBNAIL_LOD_LEVELS[-1]

    def _cover_crop_pixmap_to_cell_aspect(self, pixmap: QPixmap) -> QPixmap:
        if pixmap.isNull():
            return pixmap
        target_aspect = max(1, self._grid_size.width()) / float(max(1, self._grid_size.height()))
        source_w = max(1, pixmap.width())
        source_h = max(1, pixmap.height())
        source_aspect = source_w / float(source_h)
        if abs(source_aspect - target_aspect) <= 1e-6:
            return pixmap
        if source_aspect > target_aspect:
            crop_w = max(1, round(source_h * target_aspect))
            crop_x = max(0, (source_w - crop_w) // 2)
            return pixmap.copy(QRect(crop_x, 0, crop_w, source_h))
        crop_h = max(1, round(source_w / target_aspect))
        crop_y = max(0, (source_h - crop_h) // 2)
        return pixmap.copy(QRect(0, crop_y, source_w, crop_h))

    def _pixmap_scene_scale(self, pixmap: QPixmap) -> float:
        if pixmap.isNull():
            return 1.0
        return max(
            max(1, self._grid_size.width()) / float(max(1, pixmap.width())),
            max(1, self._grid_size.height()) / float(max(1, pixmap.height())),
        )

    def _position_pixmap_item(self, pixmap_item: QGraphicsPixmapItem, pixmap: QPixmap, x: int, y: int) -> None:
        pixmap_item.setScale(self._pixmap_scene_scale(pixmap))
        display_w = pixmap.width() * pixmap_item.scale()
        display_h = pixmap.height() * pixmap_item.scale()
        pixmap_item.setPos(
            x + max(0, (self._grid_size.width() - display_w) / 2.0),
            y + max(0, (self._grid_size.height() - display_h) / 2.0),
        )

    def thumbnailSourceSize(self) -> QSize:
        lod = self._thumbnail_lod()
        width, height = clamp_thumbnail_source_size(
            round(self._icon_size.width() * lod),
            round(self._icon_size.height() * lod),
        )
        return QSize(width, height)

    def _emit_thumbnail_lod_if_needed(self) -> None:
        source_size = self.thumbnailSourceSize()
        if source_size == self._emitted_thumbnail_source_size:
            return
        self._emitted_thumbnail_source_size = QSize(source_size)
        self.thumbnailLodChanged.emit(source_size.width(), source_size.height())

    def _cell_step_x(self) -> int:
        cell_w = max(1, self._grid_size.width())
        return max(1, cell_w - min(self._overlap_pixels_x, max(0, cell_w - 1)))

    def _cell_step_y(self) -> int:
        cell_h = max(1, self._grid_size.height())
        return max(1, cell_h - min(self._overlap_pixels_y, max(0, cell_h - 1)))
