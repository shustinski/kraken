"""editor_scene: image pixmap."""
from __future__ import annotations
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import (
    QPainterPath,
    QPixmap,
)
from ...adapters.qt.image_conversion import cv_to_qimage


class ImagePixmapMixin:
    def set_pending_path_width(self, width: float, cosmetic: bool | None = None) -> None:
        self._pending_brush_width = max(1.0, float(width))
        pen = self._pending_path_item.pen()
        pen.setWidthF(max(1.0, float(width)))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if cosmetic is not None:
            pen.setCosmetic(bool(cosmetic))
        self._pending_path_item.setPen(pen)

    def set_image(self, image) -> None:
        if image is None:
            self.set_image_pixmap(QPixmap())
            return
        pixmap = QPixmap.fromImage(cv_to_qimage(image))
        self.set_image_pixmap(pixmap)

    def set_image_pixmap(self, pixmap: QPixmap) -> None:
        if pixmap is None or pixmap.isNull():
            self._image_item.setPixmap(QPixmap())
            self._image_rect = QRectF(0, 0, 1, 1)
            self._main_frame_item.setPath(QPainterPath())
            self._main_frame_item.hide()
            self._pending_neighbor_frames = None
            self.clear_neighbor_frames()
            self.set_debug_candidates([])
            self.set_metal_overlays({}, {})
            self._update_scene_rect()
            return
        existing = self._image_item.pixmap()
        if (
            not existing.isNull()
            and existing.size() == pixmap.size()
            and round(self._image_rect.width()) == pixmap.width()
            and round(self._image_rect.height()) == pixmap.height()
            and existing.cacheKey() == pixmap.cacheKey()
        ):
            return
        self._image_item.setPixmap(pixmap)
        self._image_rect = QRectF(pixmap.rect())
        self._update_main_frame()
        self._update_scene_rect()
        if self._pending_neighbor_frames:
            self._render_pending_neighbor_frames()

    def main_image_rect(self) -> QRectF:
        return QRectF(self._image_rect)

    def main_image_pixmap(self) -> QPixmap:
        return self._image_item.pixmap()

    def navigation_base_rect(self) -> QRectF:
        rect = QRectF(self._image_rect)
        if self._neighbor_grid_bounds is not None:
            rect = rect.united(self._neighbor_grid_bounds)
        return rect

    def clear_neighbor_frames(self) -> None:
        for item in self._neighbor_frame_items:
            self.removeItem(item)
        self._neighbor_frame_items.clear()
        self._neighbor_frame_paths.clear()
        self._neighbor_frame_bounds.clear()
        self._neighbor_grid_bounds = None
        self._main_frame_item.hide()
        self._update_scene_rect()
        self.update(self.sceneRect())

    def set_neighbor_frames(
        self,
        frames: list[tuple],
        opacity: float,
        overlap_pixels: int = 0,
        show_main_frame: bool = True,
    ) -> None:
        self._pending_neighbor_frames = list(frames)
        self._pending_neighbor_opacity = float(opacity)
        self._pending_neighbor_overlap_pixels = int(overlap_pixels)
        self._pending_neighbor_show_main_frame = bool(show_main_frame)
        self._render_pending_neighbor_frames()
