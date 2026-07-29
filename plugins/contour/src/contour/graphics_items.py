from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsSimpleTextItem

from .application.processing import DisplaySettings, normalize_via_display_mode
from .domain import PolygonData


class ZoomContactBatchItem(QGraphicsItem):
    def __init__(
        self,
        *,
        centers: list[QPointF],
        width: float,
        height: float,
        pen: QPen,
        brush: QBrush,
        rectangles: bool,
    ) -> None:
        super().__init__()
        self._width = max(0.1, float(width))
        self._height = max(0.1, float(height))
        self._pen = QPen(pen)
        self._brush = QBrush(brush)
        self._rectangles = bool(rectangles)
        half_width = self._width / 2.0
        half_height = self._height / 2.0
        self._rects = [
            QRectF(
                center.x() - half_width,
                center.y() - half_height,
                self._width,
                self._height,
            )
            for center in centers
        ]
        self._centers = QPolygonF(centers)
        self._scale_y = self._height / self._width
        self._paint_centers = (
            QPolygonF(
                [
                    QPointF(point.x(), point.y() / self._scale_y)
                    for point in self._centers
                ]
            )
            if abs(self._scale_y - 1.0) > 1e-6
            else self._centers
        )
        if self._rects:
            bounds = QRectF(self._rects[0])
            for rect in self._rects[1:]:
                bounds = bounds.united(rect)
            margin = max(1.0, self._pen.widthF())
            self._bounds = bounds.adjusted(-margin, -margin, margin, margin)
        else:
            self._bounds = QRectF()
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def boundingRect(self) -> QRectF:
        return QRectF(self._bounds)

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        if self._rectangles:
            painter.setPen(self._pen)
            painter.setBrush(self._brush)
            painter.drawRects(self._rects)
            return

        painter.save()
        if abs(self._scale_y - 1.0) > 1e-6:
            painter.scale(1.0, self._scale_y)
        marker_color = QColor(self._pen.color())
        marker_color.setAlpha(max(96, self._brush.color().alpha()))
        marker_pen = QPen(
            marker_color,
            self._width,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
        )
        painter.setPen(marker_pen)
        painter.drawPoints(self._paint_centers)
        painter.restore()


class VertexHandleItem(QGraphicsEllipseItem):
    def __init__(self, polygon_id: int, vertex_index: int, parent: QGraphicsItem | None = None) -> None:
        super().__init__(parent)
        self.polygon_id = polygon_id
        self.vertex_index = vertex_index
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setZValue(6)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)

    def update_geometry(self, point: tuple[float, float], size: float, color: QColor) -> None:
        radius = size / 2.0
        self.setPos(QPointF(point[0], point[1]))
        self.setRect(QRectF(-radius, -radius, size, size))
        self.setBrush(QBrush(color))
        pen = QPen(color, 1.0)
        pen.setCosmetic(True)
        self.setPen(pen)


class EditablePolygonItem(QGraphicsPathItem):
    def __init__(self, polygon: PolygonData, display_settings: DisplaySettings) -> None:
        super().__init__()
        self.polygon_id = polygon.id
        self._polygon = polygon
        self._label_item = QGraphicsSimpleTextItem(str(polygon.id), self)
        self._label_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self._handles: list[VertexHandleItem] = []
        self.setZValue(3)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.update_from_polygon(polygon, display_settings, selected=False)

    def update_from_polygon(
        self,
        polygon: PolygonData,
        display_settings: DisplaySettings,
        selected: bool,
        cutout_polygons: list[PolygonData] | None = None,
        custom_color: str | None = None,
        *,
        conductor_hover_highlight: bool = False,
        preview_vertices: bool = False,
    ) -> None:
        self._polygon = polygon
        path = QPainterPath()
        path.addPath(_display_path_for_polygon(self._polygon, display_settings))
        for cutout in cutout_polygons or []:
            path.addPath(_display_path_for_polygon(cutout, display_settings))
        path.setFillRule(Qt.FillRule.OddEvenFill)
        self.setPath(path)

        self._update_appearance(
            display_settings,
            selected=selected,
            custom_color=custom_color,
            conductor_hover_highlight=conductor_hover_highlight,
        )

        self._label_item.setText(str(polygon.id))
        self._label_item.setVisible(display_settings.show_labels)

        bbox = self.boundingRect()
        self._label_item.setPos(bbox.left(), bbox.top() - 16.0)

        handle_color = QColor(display_settings.vertex_color)
        show_handles = (
            display_settings.show_vertices
            and (selected or preview_vertices)
            and not _is_ellipse_display_polygon(self._polygon)
        )
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

    def update_selection_appearance(
        self,
        display_settings: DisplaySettings,
        *,
        selected: bool,
        custom_color: str | None = None,
    ) -> None:
        self._update_appearance(
            display_settings,
            selected=selected,
            custom_color=custom_color,
            conductor_hover_highlight=False,
        )

    def _update_appearance(
        self,
        display_settings: DisplaySettings,
        *,
        selected: bool,
        custom_color: str | None,
        conductor_hover_highlight: bool,
    ) -> None:
        polygon = self._polygon
        cat = str(getattr(polygon, "category", "") or "")
        if selected and cat == "via":
            color_name = display_settings.via_selection_color
        elif selected:
            color_name = display_settings.selected_color
        elif conductor_hover_highlight:
            color_name = display_settings.conductor_hover_highlight_color
        elif cat == "metal_wide_gradient":
            color_name = "#2563EB"
        elif custom_color:
            color_name = custom_color
        elif polygon.is_hole:
            color_name = display_settings.hole_color
        else:
            color_name = display_settings.external_color
        outline = QColor(color_name)
        fill = QColor(color_name)
        if polygon.is_hole:
            fill.setAlpha(0)
        else:
            fill.setAlphaF(max(0.0, min(1.0, display_settings.fill_opacity)))

        pen = QPen(outline, max(1.0, display_settings.line_width))
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(fill))
        self._label_item.setBrush(QBrush(outline))

    @property
    def polygon(self) -> PolygonData:
        return self._polygon.clone()


def _closed_polygon_path(points: list[tuple[float, float]]) -> QPainterPath:
    path = QPainterPath()
    if not points:
        return path
    first_x, first_y = points[0]
    path.moveTo(first_x, first_y)
    for x_coord, y_coord in points[1:]:
        path.lineTo(x_coord, y_coord)
    if len(points) > 2:
        path.closeSubpath()
    return path


def _is_ellipse_display_polygon(polygon: PolygonData) -> bool:
    return polygon.shape_hint == "box" or polygon.category == "via"


def _ellipse_path_from_points(points: list[tuple[float, float]]) -> QPainterPath:
    path = QPainterPath()
    if not points:
        return path
    x_values = [float(point[0]) for point in points]
    y_values = [float(point[1]) for point in points]
    left = min(x_values)
    top = min(y_values)
    right = max(x_values)
    bottom = max(y_values)
    if right <= left or bottom <= top:
        return _closed_polygon_path(points)
    path.addEllipse(QRectF(left, top, right - left, bottom - top))
    return path


def _display_path_for_polygon(polygon: PolygonData, display_settings: DisplaySettings | None = None) -> QPainterPath:
    if _is_ellipse_display_polygon(polygon) and (
        display_settings is None or normalize_via_display_mode(display_settings.via_display_mode) == "circle"
    ):
        return _ellipse_path_from_points(polygon.points)
    return _closed_polygon_path(polygon.points)
