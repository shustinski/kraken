"""editor_scene: begin zoom vector render."""
from __future__ import annotations
from math import ceil
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QGraphicsPixmapItem,
)
from ...application.processing import normalize_via_display_mode
from contour.graphics_items import (
    ZoomContactBatchItem,
)
from contour.graphics.tool_mode_logic import (
    is_via_polygon as _is_via_polygon,
)

from contour.graphics.editor_scene_parts.module_state import (
    _ZOOM_BATCH_TILE_SIZE,
    _ZOOM_RASTER_MAX_DIMENSION,
)

from contour.graphics.editor_scene_parts.gradient_arrows_path import _zoom_quantized_rgba


class BeginZoomVectorRenderMixin:
    def begin_zoom_vector_render_mode(self, *, minimum_contacts: int) -> bool:
        if self._zoom_contact_composite_items:
            return True
        if not self._polygon_overlays_visible or self._display_settings.show_labels:
            return False
        visible_contacts = [
            (polygon_id, self._polygon_items[polygon_id])
            for polygon_id, polygon in self._polygons.items()
            if (
                _is_via_polygon(polygon)
                and polygon_id in self._polygon_items
                and self._polygon_items[polygon_id].isVisible()
            )
        ]
        if len(visible_contacts) < max(1, int(minimum_contacts)):
            return False

        rectangle_mode = (
            normalize_via_display_mode(self._display_settings.via_display_mode)
            == "rectangle"
        )
        grouped_contacts: dict[
            tuple[bool, int, int, float, int, float, float, int, int],
            tuple[list[QPointF], QPen, QBrush],
        ] = {}
        zoom_colors: dict[int, QColor] = {}
        quantized_rgba: dict[int, int] = {}
        batched_ids: set[int] = set()
        for polygon_id, item in visible_contacts:
            bounds = item.path().boundingRect()
            if bounds.isEmpty():
                continue
            width = max(0.1, float(bounds.width()))
            height = max(0.1, float(bounds.height()))
            center = item.mapToScene(bounds.center())
            pen = item.pen()
            brush = item.brush()
            raw_pen_rgba = int(pen.color().rgba())
            pen_rgba = quantized_rgba.get(raw_pen_rgba)
            if pen_rgba is None:
                pen_rgba = _zoom_quantized_rgba(raw_pen_rgba)
                quantized_rgba[raw_pen_rgba] = pen_rgba
            raw_brush_rgba = int(brush.color().rgba())
            brush_rgba = quantized_rgba.get(raw_brush_rgba)
            if brush_rgba is None:
                brush_rgba = _zoom_quantized_rgba(raw_brush_rgba)
                quantized_rgba[raw_brush_rgba] = brush_rgba
            key = (
                polygon_id in self._selected_polygon_ids,
                pen_rgba,
                brush_rgba,
                float(pen.widthF()),
                int(brush.style().value),
                round(width, 3),
                round(height, 3),
                int(center.x() // _ZOOM_BATCH_TILE_SIZE),
                int(center.y() // _ZOOM_BATCH_TILE_SIZE),
            )
            group = grouped_contacts.get(key)
            if group is None:
                pen_color = zoom_colors.get(pen_rgba)
                if pen_color is None:
                    pen_color = QColor.fromRgba(pen_rgba)
                    zoom_colors[pen_rgba] = pen_color
                brush_color = zoom_colors.get(brush_rgba)
                if brush_color is None:
                    brush_color = QColor.fromRgba(brush_rgba)
                    zoom_colors[brush_rgba] = brush_color
                zoom_pen = QPen(pen)
                zoom_pen.setColor(pen_color)
                zoom_brush = QBrush(brush)
                zoom_brush.setColor(brush_color)
                centers: list[QPointF] = []
                grouped_contacts[key] = (centers, zoom_pen, zoom_brush)
            else:
                centers = group[0]
            centers.append(center)
            batched_ids.add(polygon_id)

        if not grouped_contacts:
            return False

        batches: list[ZoomContactBatchItem] = []
        for key, (centers, pen, brush) in grouped_contacts.items():
            batch = ZoomContactBatchItem(
                centers=centers,
                width=key[5],
                height=key[6],
                pen=pen,
                brush=brush,
                rectangles=rectangle_mode,
            )
            batch.setZValue(3.1 if key[0] else 3.0)
            batches.append(batch)

        batches.sort(key=lambda item: item.zValue())
        raster_bounds = QRectF()
        for batch in batches:
            raster_bounds = (
                batch.boundingRect()
                if raster_bounds.isNull()
                else raster_bounds.united(batch.boundingRect())
            )
        raster_scale = min(
            1.0,
            _ZOOM_RASTER_MAX_DIMENSION / max(1.0, float(raster_bounds.width())),
            _ZOOM_RASTER_MAX_DIMENSION / max(1.0, float(raster_bounds.height())),
        )
        raster_width = max(1, ceil(float(raster_bounds.width()) * raster_scale))
        raster_height = max(1, ceil(float(raster_bounds.height()) * raster_scale))
        raster_image = QImage(
            raster_width,
            raster_height,
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        raster_image.fill(Qt.GlobalColor.transparent)
        raster_painter = QPainter(raster_image)
        raster_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        raster_painter.scale(raster_scale, raster_scale)
        raster_painter.translate(-raster_bounds.left(), -raster_bounds.top())
        for batch in batches:
            batch.paint(raster_painter, None)
        raster_painter.end()

        raster_item = QGraphicsPixmapItem(QPixmap.fromImage(raster_image))
        raster_item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        raster_item.setPos(raster_bounds.topLeft())
        raster_item.setScale(1.0 / raster_scale)
        raster_item.setZValue(3.0)
        self.addItem(raster_item)
        self._zoom_contact_composite_items = [raster_item]
        self._zoom_hidden_contact_ids = batched_ids
        for polygon_id in self._zoom_hidden_contact_ids:
            self._polygon_items[polygon_id].setVisible(False)
        return True
