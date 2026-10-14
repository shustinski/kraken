"""editor_scene: render pending neighbor frames."""
from __future__ import annotations
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import (
    QImage,
    QPixmap,
    QTransform,
)
from PyQt6.QtWidgets import (
    QGraphicsPixmapItem,
)
from ...adapters.qt.image_conversion import cv_to_qimage
from contour.domain import PolygonData
from contour.graphics_items import (
    _hole_display_hidden,
)


class RenderPendingNeighborFramesMixin:
    def _render_pending_neighbor_frames(self) -> None:
        frames = self._pending_neighbor_frames
        if frames is None:
            return
        self.clear_neighbor_frames()
        self._main_frame_item.setVisible(bool(self._pending_neighbor_show_main_frame))
        if not frames or self._image_rect.width() <= 1.0 or self._image_rect.height() <= 1.0:
            return
        opacity = self._pending_neighbor_opacity
        overlap_pixels = self._pending_neighbor_overlap_pixels
        main_width = float(self._image_rect.width())
        main_height = float(self._image_rect.height())
        overlap = max(0.0, min(float(overlap_pixels), min(main_width, main_height) - 1.0))
        step_x = max(1.0, main_width - overlap)
        step_y = max(1.0, main_height - overlap)
        bounds = QRectF(self._image_rect)
        clamped_opacity = max(0.05, min(1.0, float(opacity)))
        neighbor_bounds: list[tuple[str, QRectF]] = []
        for frame in frames:
            column_offset, row_offset, image, image_path = frame[:4]
            polygons = frame[4] if len(frame) > 4 else []
            source_size = frame[5] if len(frame) > 5 else None
            if column_offset == 0 and row_offset == 0:
                continue
            pixmap = QPixmap.fromImage(image if isinstance(image, QImage) else cv_to_qimage(image))
            if pixmap.isNull():
                continue
            item = QGraphicsPixmapItem(pixmap)
            item.setZValue(-20)
            item.setOpacity(clamped_opacity)
            item.setToolTip(str(image_path))
            scale_x = main_width / max(1, pixmap.width())
            scale_y = main_height / max(1, pixmap.height())
            item.setTransform(QTransform.fromScale(scale_x, scale_y))
            item.setPos(float(column_offset) * step_x, float(row_offset) * step_y)
            frame_rect = QRectF(item.pos().x(), item.pos().y(), main_width, main_height)
            neighbor_bounds.append((str(image_path), frame_rect))
            self.addItem(item)
            self._neighbor_frame_items.append(item)
            self._neighbor_frame_paths[item] = str(image_path)
            bounds = bounds.united(QRectF(item.pos().x(), item.pos().y(), main_width, main_height))
            if polygons:
                source_width, source_height = source_size or (pixmap.width(), pixmap.height())
                vector_transform = QTransform.fromScale(
                    main_width / max(1, int(source_width)),
                    main_height / max(1, int(source_height)),
                )
                holes_by_parent: dict[int, list[PolygonData]] = {}
                for polygon in polygons:
                    if polygon.is_hole and polygon.parent_id is not None:
                        holes_by_parent.setdefault(int(polygon.parent_id), []).append(polygon)
                for polygon in polygons:
                    if polygon.is_hole and _hole_display_hidden(
                        polygon,
                        {loaded_polygon.id: loaded_polygon for loaded_polygon in polygons},
                    ):
                        continue
                    cutouts = [] if polygon.is_hole else holes_by_parent.get(int(polygon.id), [])
                    path_item = self._neighbor_vector_item(
                        polygon,
                        str(image_path),
                        clamped_opacity,
                        cutouts,
                    )
                    path_item.setZValue(-18.9 if polygon.is_hole else -19)
                    path_item.setTransform(vector_transform)
                    path_item.setPos(item.pos())
                    self.addItem(path_item)
                    self._neighbor_frame_items.append(path_item)
                    self._neighbor_frame_paths[path_item] = str(image_path)
        self._neighbor_frame_bounds = neighbor_bounds
        self._neighbor_grid_bounds = bounds if self._neighbor_frame_items else None
        self._update_scene_rect()
        self.update(self.sceneRect())
