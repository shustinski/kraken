from __future__ import annotations

from collections.abc import Iterator

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsSimpleTextItem

from .application.processing import DisplaySettings, normalize_via_display_mode
from .domain import PolygonData

INVALID_POLYGON_DESCRIPTION_COLOR = "#DC2626"
MOVE_TARGET_VERTEX_HIGHLIGHT_COLOR = "#FBBF24"
MOVE_TARGET_EDGE_HIGHLIGHT_COLOR = "#38BDF8"


def _uses_cif_paint_display(polygon: PolygonData) -> bool:
    from .serializers import _polygon_uses_authored_cif_paint_ring

    return _polygon_uses_authored_cif_paint_ring(polygon)


def _hole_display_hidden(polygon: PolygonData, polygons_by_id: dict[int, PolygonData]) -> bool:
    if not polygon.is_hole or polygon.parent_id is None:
        return False
    parent = polygons_by_id.get(polygon.parent_id)
    if parent is None:
        return False
    return _uses_cif_paint_display(parent)


def _vector_display_path_for_polygon(
    polygon: PolygonData,
    display_settings: DisplaySettings | None = None,
    cutout_polygons: list[PolygonData] | None = None,
) -> QPainterPath:
    """Build the visible outline path for vector display."""

    if _uses_cif_paint_display(polygon):
        path = _closed_polygon_path(polygon.cif_paint_ring)
        path.setFillRule(Qt.FillRule.WindingFill)
        return path
    path = QPainterPath()
    path.addPath(_display_path_for_polygon(polygon, display_settings))
    for cutout in cutout_polygons or []:
        path.addPath(_cutout_path_for_polygon(cutout, display_settings, outer=polygon))
    path.setFillRule(Qt.FillRule.WindingFill)
    return path


class VectorPolygonDisplayItem(QGraphicsPathItem):
    """Read-only polygon item with KLayout-compatible CIF fill semantics."""

    def __init__(
        self,
        polygon: PolygonData,
        display_settings: DisplaySettings,
        *,
        cutout_polygons: list[PolygonData] | None = None,
        opacity: float = 1.0,
        color_name: str | None = None,
        tooltip: str = "",
    ) -> None:
        super().__init__()
        self._polygon = polygon
        self.setOpacity(float(opacity))
        self.setToolTip(tooltip)
        self.apply_polygon_display(
            polygon,
            display_settings,
            cutout_polygons=cutout_polygons,
            color_name=color_name,
        )

    def apply_polygon_display(
        self,
        polygon: PolygonData,
        display_settings: DisplaySettings,
        *,
        cutout_polygons: list[PolygonData] | None = None,
        color_name: str | None = None,
    ) -> None:
        self._polygon = polygon
        outline_path = _vector_display_path_for_polygon(polygon, display_settings, cutout_polygons)
        self.setPath(outline_path)
        if color_name is None:
            color_name = display_settings.hole_color if polygon.is_hole else display_settings.external_color
        if polygon.description_is_invalid():
            color_name = INVALID_POLYGON_DESCRIPTION_COLOR
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
    def __init__(
        self,
        polygon: PolygonData,
        display_settings: DisplaySettings,
        *,
        custom_color: str | None = None,
        paint: bool = True,
    ) -> None:
        super().__init__()
        self.polygon_id = polygon.id
        self._polygon = polygon
        self._label_item = QGraphicsSimpleTextItem(str(polygon.id), self)
        self._label_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self._handles: list[VertexHandleItem] = []
        self._edge_highlight_item = QGraphicsPathItem(self)
        self._edge_highlight_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._edge_highlight_item.setZValue(5)
        self.setZValue(3)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        if paint:
            self.update_from_polygon(
                polygon,
                display_settings,
                selected=False,
                custom_color=custom_color,
            )
        else:
            self.bind_polygon_data(polygon)

    def bind_polygon_data(self, polygon: PolygonData) -> None:
        """Attach polygon identity without rebuilding path/appearance (bulk load)."""

        self.polygon_id = polygon.id
        self._polygon = polygon
        for handle in self._handles:
            handle.setVisible(False)

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
        highlight_vertex_index: int | None = None,
        highlight_edge_index: int | None = None,
        needs_repair: bool | None = None,
    ) -> None:
        self.polygon_id = polygon.id
        self._polygon = polygon
        outline_path = _vector_display_path_for_polygon(self._polygon, display_settings, cutout_polygons)
        self.setPath(outline_path)
        self._hit_path = _hit_path_for_polygon(self._polygon, cutout_polygons=cutout_polygons)

        self._update_appearance(
            display_settings,
            selected=selected,
            custom_color=custom_color,
            conductor_hover_highlight=conductor_hover_highlight,
            needs_repair=needs_repair,
        )

        self._label_item.setText(str(polygon.id))
        self._label_item.setVisible(display_settings.show_labels)

        bbox = self.boundingRect()
        self._label_item.setPos(bbox.left(), bbox.top() - 16.0)

        handle_color = QColor(display_settings.vertex_color)
        show_handles = not _is_ellipse_display_polygon(self._polygon) and (
            preview_vertices or (selected and display_settings.show_vertices)
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
            highlight_size = max(display_settings.vertex_size + 2.0, display_settings.vertex_size * 1.35)
            highlight_color = QColor(MOVE_TARGET_VERTEX_HIGHLIGHT_COLOR)
            for index, point in enumerate(self._polygon.points):
                handle = self._handles[index]
                handle.polygon_id = self.polygon_id
                handle.vertex_index = index
                if highlight_vertex_index is not None and index == highlight_vertex_index:
                    handle.update_geometry(point, highlight_size, highlight_color)
                else:
                    handle.update_geometry(point, display_settings.vertex_size, handle_color)
                handle.setVisible(True)

        self._update_edge_highlight(
            highlight_edge_index,
            conductor_hover_highlight=conductor_hover_highlight,
            display_settings=display_settings,
        )

    def _update_edge_highlight(
        self,
        highlight_edge_index: int | None,
        *,
        conductor_hover_highlight: bool = False,
        display_settings: DisplaySettings | None = None,
    ) -> None:
        if highlight_edge_index is not None and len(self._polygon.points) >= 2:
            points = self._polygon.points
            start = points[highlight_edge_index]
            end = points[(highlight_edge_index + 1) % len(points)]
            path = QPainterPath()
            path.moveTo(start[0], start[1])
            path.lineTo(end[0], end[1])
            self._edge_highlight_item.setPath(path)
            edge_pen = QPen(QColor(MOVE_TARGET_EDGE_HIGHLIGHT_COLOR), 4.0)
            edge_pen.setCosmetic(True)
            edge_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            self._edge_highlight_item.setPen(edge_pen)
            self._edge_highlight_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            self._edge_highlight_item.setVisible(True)
            return
        if conductor_hover_highlight and display_settings is not None:
            hover_path = self.path()
            if hover_path.isEmpty():
                self._edge_highlight_item.setPath(QPainterPath())
                return
            self._edge_highlight_item.setPath(hover_path)
            hover_color = QColor(display_settings.conductor_hover_highlight_color)
            hover_pen = QPen(
                hover_color,
                max(3.0, float(display_settings.line_width) * 2.0),
            )
            hover_pen.setCosmetic(True)
            hover_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            hover_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            self._edge_highlight_item.setPen(hover_pen)
            self._edge_highlight_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            self._edge_highlight_item.setVisible(True)
            return
        self._edge_highlight_item.setPath(QPainterPath())

    def update_selection_appearance(
        self,
        display_settings: DisplaySettings,
        *,
        selected: bool,
        custom_color: str | None = None,
        needs_repair: bool | None = None,
    ) -> None:
        self._update_appearance(
            display_settings,
            selected=selected,
            custom_color=custom_color,
            conductor_hover_highlight=False,
            needs_repair=needs_repair,
        )

    def _update_appearance(
        self,
        display_settings: DisplaySettings,
        *,
        selected: bool,
        custom_color: str | None,
        conductor_hover_highlight: bool,
        needs_repair: bool | None = None,
    ) -> None:
        polygon = self._polygon
        cat = str(getattr(polygon, "category", "") or "")
        mark_repair = bool(needs_repair) if needs_repair is not None else polygon.description_is_invalid()
        if selected and cat == "via":
            color_name = display_settings.via_selection_color
        elif selected:
            color_name = display_settings.selected_color
        elif conductor_hover_highlight:
            color_name = display_settings.conductor_hover_highlight_color
        elif mark_repair:
            color_name = INVALID_POLYGON_DESCRIPTION_COLOR
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
        if conductor_hover_highlight:
            pen.setWidthF(max(3.0, float(display_settings.line_width) * 2.0))
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(fill))
        self._label_item.setBrush(QBrush(outline))

    @property
    def polygon(self) -> PolygonData:
        return self._polygon.clone()

    def shape(self) -> QPainterPath:
        hit_path = getattr(self, "_hit_path", None)
        if hit_path is not None and not hit_path.isEmpty():
            return hit_path
        return super().shape()

def _iter_shapely_polygon_parts(geom: object) -> Iterator[object]:
    from shapely.geometry import Polygon as ShapelyPolygon

    if getattr(geom, "is_empty", True):
        return
    geom_type = getattr(geom, "geom_type", "")
    if geom_type == "Polygon":
        yield geom
    elif geom_type == "MultiPolygon":
        for part in geom.geoms:
            yield part
    elif geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from _iter_shapely_polygon_parts(part)


def _ring_to_path(coords: object) -> QPainterPath:
    ring_points = list(coords)
    if len(ring_points) < 2:
        return QPainterPath()
    path = QPainterPath()
    start_x, start_y = float(ring_points[0][0]), float(ring_points[0][1])
    path.moveTo(start_x, start_y)
    for x_coord, y_coord in ring_points[1:]:
        path.lineTo(float(x_coord), float(y_coord))
    path.closeSubpath()
    return path


def _shapely_polygon_parts_to_path(geom: object) -> QPainterPath:
    path = QPainterPath()
    for poly in _iter_shapely_polygon_parts(geom):
        exterior_coords = list(poly.exterior.coords)
        path.addPath(_ring_to_path(exterior_coords))
        exterior_area = _ring_signed_area(exterior_coords)
        for interior in poly.interiors:
            interior_coords = list(interior.coords)
            if exterior_area != 0.0 and _ring_signed_area(interior_coords) * exterior_area > 0.0:
                interior_coords = list(reversed(interior_coords))
            path.addPath(_ring_to_path(interior_coords))
    path.setFillRule(Qt.FillRule.WindingFill)
    return path


def _hit_path_for_polygon(
    polygon: PolygonData,
    *,
    cutout_polygons: list[PolygonData] | None = None,
) -> QPainterPath:
    """Build a pick path that matches editable metal, not phantom winding-fill overlap."""

    if _is_ellipse_display_polygon(polygon):
        return _ellipse_path_from_points(polygon.points)

    if cutout_polygons:
        path = QPainterPath()
        path.addPath(_display_path_for_polygon(polygon))
        for cutout in cutout_polygons:
            path.addPath(_cutout_path_for_polygon(cutout, outer=polygon))
        path.setFillRule(Qt.FillRule.WindingFill)
        return path

    shell = list(polygon.points)
    if len(shell) < 3:
        return _closed_polygon_path(shell)

    from shapely import make_valid
    from shapely.geometry import Polygon as ShapelyPolygon

    try:
        geom = make_valid(ShapelyPolygon(shell))
    except Exception:
        return _closed_polygon_path(shell)
    if geom.is_empty:
        return QPainterPath()
    return _shapely_polygon_parts_to_path(geom)


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


def _ring_signed_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    count = len(points)
    for index, (x_coord, y_coord) in enumerate(points):
        next_x, next_y = points[(index + 1) % count]
        area += float(x_coord) * float(next_y) - float(next_x) * float(y_coord)
    return area


def _cutout_path_for_polygon(
    polygon: PolygonData,
    display_settings: DisplaySettings | None = None,
    *,
    outer: PolygonData | None = None,
) -> QPainterPath:
    """Build a hole subpath that stays a hole under winding-fill hit tests.

    ``QGraphicsPathItem.shape()`` uses winding fill. CIF holes often share the
    outer ring's winding, so they must be reversed; shapely interiors are
    already opposite and must be left as-is.
    """

    if _is_ellipse_display_polygon(polygon) and (
        display_settings is None or normalize_via_display_mode(display_settings.via_display_mode) == "circle"
    ):
        return _ellipse_path_from_points(polygon.points)
    points = list(polygon.points)
    if (
        outer is not None
        and len(points) >= 3
        and len(outer.points) >= 3
        and _ring_signed_area(outer.points) * _ring_signed_area(points) > 0.0
    ):
        points = list(reversed(points))
    return _closed_polygon_path(points)


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
