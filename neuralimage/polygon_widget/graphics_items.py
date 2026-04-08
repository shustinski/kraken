from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPen, QPolygonF
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsPolygonItem, QGraphicsSimpleTextItem

from .models import DisplaySettings, PolygonData


class VertexHandleItem(QGraphicsEllipseItem):
    def __init__(self, polygon_id: int, vertex_index: int, parent: QGraphicsPolygonItem | None = None) -> None:
        super().__init__(parent)
        self.polygon_id = polygon_id
        self.vertex_index = vertex_index
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setZValue(6)

    def update_geometry(self, point: tuple[float, float], size: float, color: QColor) -> None:
        radius = size / 2.0
        self.setRect(QRectF(point[0] - radius, point[1] - radius, size, size))
        self.setBrush(QBrush(color))
        self.setPen(QPen(color, 1.0))

class EditablePolygonItem(QGraphicsPolygonItem):
    def __init__(self, polygon: PolygonData, display_settings: DisplaySettings) -> None:
        super().__init__()
        self.polygon_id = polygon.id
        self._polygon = polygon.clone()
        self._label_item = QGraphicsSimpleTextItem(str(polygon.id), self)
        self._handles: list[VertexHandleItem] = []
        self.setZValue(3)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.update_from_polygon(self._polygon, display_settings, selected=False)

    def update_from_polygon(self, polygon: PolygonData, display_settings: DisplaySettings, selected: bool) -> None:
        self._polygon = polygon.clone()
        qpolygon = QPolygonF([QPointF(x_coord, y_coord) for x_coord, y_coord in self._polygon.points])
        self.setPolygon(qpolygon)

        color_name = display_settings.selected_color if selected else (
            display_settings.hole_color if polygon.is_hole else display_settings.external_color
        )
        outline = QColor(color_name)
        fill = QColor(color_name)
        fill.setAlphaF(max(0.0, min(1.0, display_settings.fill_opacity)))

        self.setPen(QPen(outline, max(1.0, display_settings.line_width)))
        self.setBrush(QBrush(fill))
        self._label_item.setText(str(polygon.id))
        self._label_item.setBrush(QBrush(outline))
        self._label_item.setVisible(display_settings.show_labels)

        bbox = self.boundingRect()
        self._label_item.setPos(bbox.left(), bbox.top() - 16.0)

        handle_color = QColor(display_settings.vertex_color)
        show_handles = display_settings.show_vertices and selected
        target_handle_count = len(self._polygon.points) if show_handles else 0
        while len(self._handles) < target_handle_count:
            self._handles.append(VertexHandleItem(self.polygon_id, len(self._handles), self))
        while len(self._handles) > target_handle_count:
            handle = self._handles.pop()
            if handle.scene() is not None:
                handle.scene().removeItem(handle)
            handle.setParentItem(None)

        if show_handles:
            for index, point in enumerate(self._polygon.points):
                handle = self._handles[index]
                handle.vertex_index = index
                handle.update_geometry(point, display_settings.vertex_size, handle_color)
                handle.setVisible(True)

    @property
    def polygon(self) -> PolygonData:
        return self._polygon.clone()
