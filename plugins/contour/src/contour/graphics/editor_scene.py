from __future__ import annotations

from math import ceil, floor, hypot

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QTransform,
    QUndoStack,
)
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)
from shapely import make_valid, unary_union
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box as shapely_box

from ..adapters.qt.image_conversion import cv_to_qimage
from ..application.polygon_antialiasing import antialias_polygons
from ..application.processing import DisplaySettings, normalize_via_display_mode
from ..application.vector_geometry_postprocess import (
    VectorGeometrySettings,
    collapse_redundant_vertices_in_polygons,
    dissolve_self_intersecting_polygons,
    polygons_needing_repair,
    postprocess_after_editor_mutation,
    postprocess_changed_polygon_edit,
    repair_invalid_polygon_descriptions,
    resolve_focus_id_after_geometry_pass,
    union_after_removing_polygon_ids,
)
from ..commands import (
    AddPolygonCommand,
    AddPolygonsCommand,
    AddVertexCommand,
    DeletePolygonCommand,
    ReplacePolygonSetCommand,
)
from ..domain import PolygonData, compute_polygon_metrics, integer_point, integer_points
from ..graphics_items import (
    EditablePolygonItem,
    INVALID_POLYGON_DESCRIPTION_COLOR,
    VertexHandleItem,
    ZoomContactBatchItem,
    _cutout_path_for_polygon,
    _display_path_for_polygon,
)
from ..i18n import active_language, tr
from .brush_vector import (
    QUAD_SEGS_BRUSH_DEFAULT,
    apply_boolean,
    bbox_intersects_geom_bounds,
    extract_polygonal_union,
    polygon_equivalent_preserved,
    polygon_footprint_geom,
    polygon_paint_footprint_geom,
    region_geometry,
    shapely_to_polygon_data_list,
    simplify_polygonal_geometry,
    tool_geometry,
)
from ..domain.polygon_ring import collapse_redundant_polyline_vertices, is_valid_closed_polygon_ring
from .geometry import (
    _bbox_from_points,
    _bboxes_intersect,
    _centered_rect,
    _distance_to_segment,
    _measurement_label_position,
    _polygon_data_rect,
    _smallest_containing_polygon,
    _stable_object_color,
    is_valid_open_polyline_last_edge,
    resolve_conductor_hover_target_id,
)
from .polygon_creation import (
    POLYGON_COMMIT_INVALID_RING,
    POLYGON_COMMIT_TOO_FEW_VERTICES,
    POLYGON_COMMIT_TOO_SMALL_AREA,
    polygon_commit_acceptability,
)
from .tool_mode_logic import (
    available_editor_tools,
    can_add_polygon,
    can_add_polygon_set,
    can_add_via,
    is_recognized_via,
)
from .tool_mode_logic import (
    is_via_polygon as _is_via_polygon,
)
from .tools import EditorTool

_ZOOM_COLOR_QUANTIZATION_STEP = 17
_ZOOM_BATCH_TILE_SIZE = 256.0
_ZOOM_RASTER_MAX_DIMENSION = 4096
_CONTACT_PASTE_SPATIAL_CELL_SIZE = 64.0
_OUTER_POLYGON_ITEM_Z = 3.0
_HOLE_POLYGON_ITEM_Z = 3.2
_OUTER_PICK_Z_SPAN = 0.19
_PICK_CYCLE_DISTANCE = 4.0
_ZOOM_QUANTIZED_CHANNELS = tuple(
    min(255, max(0, round(channel / _ZOOM_COLOR_QUANTIZATION_STEP) * _ZOOM_COLOR_QUANTIZATION_STEP))
    for channel in range(256)
)


def _gradient_arrows_path(arrows: list[tuple[float, float, float, float]]) -> QPainterPath:
    path = QPainterPath()
    for origin_x, origin_y, delta_x, delta_y in arrows:
        length = hypot(delta_x, delta_y)
        if length <= 1e-9:
            continue
        ux, uy = delta_x / length, delta_y / length
        nx, ny = -uy, ux
        tip_length = 0.32 * length
        tip_half = 0.18 * length
        end_x = origin_x + delta_x
        end_y = origin_y + delta_y
        shaft_x = end_x - ux * tip_length
        shaft_y = end_y - uy * tip_length
        path.moveTo(origin_x, origin_y)
        path.lineTo(shaft_x, shaft_y)
        path.moveTo(end_x, end_y)
        path.lineTo(end_x - ux * tip_length + nx * tip_half, end_y - uy * tip_length + ny * tip_half)
        path.lineTo(end_x - ux * tip_length - nx * tip_half, end_y - uy * tip_length - ny * tip_half)
        path.closeSubpath()
    return path


def _zoom_quantized_rgba(rgba: int) -> int:
    value = int(rgba)
    return (
        (value & 0xFF000000)
        | (_ZOOM_QUANTIZED_CHANNELS[(value >> 16) & 0xFF] << 16)
        | (_ZOOM_QUANTIZED_CHANNELS[(value >> 8) & 0xFF] << 8)
        | _ZOOM_QUANTIZED_CHANNELS[value & 0xFF]
    )


class _ContactSpatialIndex:
    def __init__(self, minimum_distance: float) -> None:
        self._minimum_distance = max(0.0, float(minimum_distance))
        self._minimum_distance_sq = self._minimum_distance * self._minimum_distance
        self._entries: list[tuple[float, float, QRectF]] = []
        self._cell_entries: dict[tuple[int, int], list[int]] = {}

    @staticmethod
    def _cells_for_rect(rect: QRectF):
        normalized = rect.normalized()
        cell_size = _CONTACT_PASTE_SPATIAL_CELL_SIZE
        left = floor(float(normalized.left()) / cell_size)
        right = floor(float(normalized.right()) / cell_size)
        top = floor(float(normalized.top()) / cell_size)
        bottom = floor(float(normalized.bottom()) / cell_size)
        for cell_y in range(top, bottom + 1):
            for cell_x in range(left, right + 1):
                yield cell_x, cell_y

    def add(self, center_x: float, center_y: float, rect: QRectF) -> None:
        entry_index = len(self._entries)
        stored_rect = QRectF(rect)
        self._entries.append((float(center_x), float(center_y), stored_rect))
        for cell in self._cells_for_rect(stored_rect):
            self._cell_entries.setdefault(cell, []).append(entry_index)

    def conflicts(self, center_x: float, center_y: float, rect: QRectF) -> bool:
        search_rect = QRectF(rect)
        if self._minimum_distance > 0.0:
            radius = self._minimum_distance
            search_rect = search_rect.united(
                QRectF(
                    float(center_x) - radius,
                    float(center_y) - radius,
                    radius * 2.0,
                    radius * 2.0,
                )
            )
        visited: set[int] = set()
        for cell in self._cells_for_rect(search_rect):
            for entry_index in self._cell_entries.get(cell, ()):
                if entry_index in visited:
                    continue
                visited.add(entry_index)
                existing_x, existing_y, existing_rect = self._entries[entry_index]
                if rect.intersects(existing_rect):
                    return True
                if (
                    self._minimum_distance_sq > 0.0
                    and (
                        (float(center_x) - existing_x) ** 2
                        + (float(center_y) - existing_y) ** 2
                    )
                    < self._minimum_distance_sq
                ):
                    return True
        return False


class PolygonEditorScene(QGraphicsScene):
    polygonsChanged = pyqtSignal()
    activePolygonChanged = pyqtSignal(object)
    logRequested = pyqtSignal(str)

    def __init__(self, parent: QGraphicsView | None = None) -> None:
        super().__init__(parent)
        self.undo_stack = QUndoStack(self)
        self._ui_language = active_language()
        self._display_settings = DisplaySettings()
        self._polygons: dict[int, PolygonData] = {}
        self._polygon_items: dict[int, EditablePolygonItem] = {}
        self._recycled_polygon_items: list[EditablePolygonItem] = []
        self._recycled_polygon_cleanup_timer = QTimer(self)
        self._recycled_polygon_cleanup_timer.setInterval(0)
        self._recycled_polygon_cleanup_timer.timeout.connect(self._drain_recycled_polygon_items)
        self._hole_children_by_parent: dict[int, list[PolygonData]] = {}
        self._polygon_child_ids_by_parent: dict[int, set[int]] = {}
        self._selected_polygon_id: int | None = None
        self._selected_polygon_ids: set[int] = set()
        self._next_polygon_id = 1
        self._polygon_overlays_visible = True
        self._gradient_overlay_user_visible = True
        self._polygon_category_visible: dict[str, bool] = {}
        self._zoom_contact_composite_items: list[QGraphicsItem] = []
        self._zoom_hidden_contact_ids: set[int] = set()
        self._protect_recognized_vias = False
        self._minimum_contact_distance = 0.0
        self._outer_pick_z_rank: dict[int, int] = {}
        self._pick_cycle_pos: QPointF | None = None
        self._pick_cycle_ids: list[int] = []
        self._pick_cycle_index = 0

        self._image_item = QGraphicsPixmapItem()
        self._image_item.setZValue(0)
        self.addItem(self._image_item)
        self._image_rect = QRectF(0, 0, 1, 1)
        self._neighbor_frame_items: list[QGraphicsItem] = []
        self._neighbor_frame_paths: dict[QGraphicsItem, str] = {}
        self._neighbor_frame_bounds: list[tuple[str, QRectF]] = []
        self._neighbor_grid_bounds: QRectF | None = None
        self._pending_neighbor_frames: list[tuple] | None = None
        self._pending_neighbor_opacity: float = 0.0
        self._pending_neighbor_overlap_pixels: int = 0
        self._pending_neighbor_show_main_frame: bool = True
        self._debug_candidate_items: list[QGraphicsPathItem | QGraphicsSimpleTextItem] = []
        self._recycled_debug_candidate_items: list[QGraphicsPathItem] = []
        self._recycled_debug_cleanup_timer = QTimer(self)
        self._recycled_debug_cleanup_timer.setInterval(0)
        self._recycled_debug_cleanup_timer.timeout.connect(self._drain_recycled_debug_candidate_items)
        self._metal_overlay_items: list[QGraphicsPathItem] = []
        self._extra_layer_items: list[QGraphicsPixmapItem] = []
        self._gradient_overlay_item = QGraphicsPixmapItem()
        self._gradient_overlay_item.setZValue(0.9)
        self._gradient_overlay_item.setOpacity(0.45)
        self.addItem(self._gradient_overlay_item)
        self._gradient_overlay_item.hide()
        self._gradient_arrows_item = QGraphicsPathItem()
        self._gradient_arrows_item.setZValue(0.95)
        self._gradient_arrows_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        arrows_pen = QPen(QColor("#F8FAFC"), 1.0)
        arrows_pen.setCosmetic(True)
        arrows_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        arrows_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        self._gradient_arrows_item.setPen(arrows_pen)
        self._gradient_arrows_item.setBrush(QBrush(QColor("#F8FAFC")))
        self.addItem(self._gradient_arrows_item)
        self._gradient_arrows_item.hide()
        self._gradient_arrows_has_content = False
        self._random_object_colors_enabled = False
        self._object_colors: dict[int, str] = {}
        self._hover_conductor_polygon_id: int | None = None
        self._vertex_preview_polygon_id: int | None = None
        self._show_all_editable_vertices = False
        self._delete_area_highlight_ids: set[int] = set()
        self._vector_geometry_settings = VectorGeometrySettings()
        self._polygons_needing_repair: dict[int, list[str]] = {}

        self._main_frame_item = QGraphicsPathItem()
        self._main_frame_item.setZValue(2)
        main_frame_pen = QPen(QColor("#FACC15"), 2.0)
        main_frame_pen.setCosmetic(True)
        self._main_frame_item.setPen(main_frame_pen)
        self._main_frame_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.addItem(self._main_frame_item)
        self._main_frame_item.hide()

        self._pending_points: list[tuple[float, float]] = []
        self._pending_cursor: tuple[float, float] | None = None
        self._pending_polyline_for_brush = False
        self._pending_brush_width = 1.5
        self._pending_path_item = QGraphicsPathItem()
        self._pending_path_item.setZValue(10)
        pending_pen = QPen(QColor("#F7B801"), 1.5, Qt.PenStyle.DashLine)
        pending_pen.setCosmetic(True)
        pending_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pending_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        self._pending_path_item.setPen(pending_pen)
        self._pending_path_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.addItem(self._pending_path_item)
        self._preview_rect_item = QGraphicsPathItem()
        self._preview_rect_item.setZValue(11)
        preview_pen = QPen(QColor("#38BDF8"), 1.5, Qt.PenStyle.DashLine)
        preview_pen.setCosmetic(True)
        self._preview_rect_item.setPen(preview_pen)
        preview_brush = QColor("#38BDF8")
        preview_brush.setAlpha(48)
        self._preview_rect_item.setBrush(QBrush(preview_brush))
        self.addItem(self._preview_rect_item)
        self._via_cursor_item = QGraphicsPathItem()
        self._via_cursor_item.setZValue(12)
        via_cursor_pen = QPen(QColor("#A78BFA"), 1.5, Qt.PenStyle.DashLine)
        via_cursor_pen.setCosmetic(True)
        self._via_cursor_item.setPen(via_cursor_pen)
        via_cursor_brush = QColor("#A78BFA")
        via_cursor_brush.setAlpha(42)
        self._via_cursor_item.setBrush(QBrush(via_cursor_brush))
        self.addItem(self._via_cursor_item)
        self._via_cursor_item.hide()
        self._brush_cursor_item = QGraphicsEllipseItem()
        self._brush_cursor_item.setZValue(12)
        brush_cursor_pen = QPen(QColor("#4ADE80"), 1.5, Qt.PenStyle.DashLine)
        brush_cursor_pen.setCosmetic(True)
        self._brush_cursor_item.setPen(brush_cursor_pen)
        self._brush_cursor_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.addItem(self._brush_cursor_item)
        self._brush_cursor_item.hide()
        self._measurement_item = QGraphicsPathItem()
        self._measurement_item.setZValue(13)
        measurement_pen = QPen(QColor("#F59E0B"), 2.0, Qt.PenStyle.DashLine)
        measurement_pen.setCosmetic(True)
        self._measurement_item.setPen(measurement_pen)
        self._measurement_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.addItem(self._measurement_item)
        self._measurement_start_marker = QGraphicsEllipseItem()
        self._measurement_start_marker.setZValue(14)
        self._measurement_start_marker.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self._measurement_start_marker.setBrush(QBrush(QColor("#F59E0B")))
        marker_pen = QPen(QColor("#F8FAFC"), 1.0)
        marker_pen.setCosmetic(True)
        self._measurement_start_marker.setPen(marker_pen)
        self.addItem(self._measurement_start_marker)
        self._measurement_start_marker.hide()
        self._measurement_end_marker = QGraphicsEllipseItem()
        self._measurement_end_marker.setZValue(14)
        self._measurement_end_marker.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self._measurement_end_marker.setBrush(QBrush(QColor("#F59E0B")))
        self._measurement_end_marker.setPen(marker_pen)
        self.addItem(self._measurement_end_marker)
        self._measurement_end_marker.hide()
        self._measurement_label_item = QGraphicsSimpleTextItem()
        self._measurement_label_item.setZValue(15)
        self._measurement_label_item.setBrush(QBrush(QColor("#F8FAFC")))
        self._measurement_label_item.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.addItem(self._measurement_label_item)
        self._measurement_label_item.hide()
        self.setSceneRect(QRectF(0, 0, 1, 1))

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
            and int(round(self._image_rect.width())) == pixmap.width()
            and int(round(self._image_rect.height())) == pixmap.height()
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

    def neighbor_frame_path_at(self, scene_pos: QPointF) -> str | None:
        x_coord = float(scene_pos.x())
        y_coord = float(scene_pos.y())
        if self._neighbor_frame_bounds:
            main_rect = QRectF(self._image_rect)
            candidates: list[tuple[str, QRectF]] = []
            for image_path, frame_rect in self._neighbor_frame_bounds:
                if frame_rect.contains(x_coord, y_coord):
                    candidates.append((image_path, frame_rect))
            if not candidates:
                return None
            if len(candidates) == 1:
                return candidates[0][0]

            def _distance_to_center(item: tuple[str, QRectF]) -> float:
                center = item[1].center()
                return hypot(x_coord - center.x(), y_coord - center.y())

            if main_rect.contains(x_coord, y_coord):
                return min(candidates, key=_distance_to_center)[0]
            return min(candidates, key=_distance_to_center)[0]

        for item in self.items(scene_pos):
            if item in self._neighbor_frame_paths:
                return self._neighbor_frame_paths[item]
        return None

    def _neighbor_vector_item(
        self,
        polygon: PolygonData,
        image_path: str,
        opacity: float,
        cutout_polygons: list[PolygonData] | None = None,
    ) -> QGraphicsPathItem:
        path = QPainterPath()
        path.addPath(_display_path_for_polygon(polygon, self._display_settings))
        for cutout in cutout_polygons or []:
            path.addPath(_cutout_path_for_polygon(cutout, self._display_settings, outer=polygon))
        path.setFillRule(Qt.FillRule.WindingFill)
        path_item = QGraphicsPathItem(path)
        path_item.setZValue(-19)
        path_item.setOpacity(opacity)
        color_name = self._display_settings.hole_color if polygon.is_hole else self._display_settings.external_color
        if polygon.description_is_invalid():
            color_name = INVALID_POLYGON_DESCRIPTION_COLOR
        outline = QColor(color_name)
        fill = QColor(color_name)
        if polygon.is_hole:
            fill.setAlpha(0)
        else:
            fill.setAlphaF(max(0.0, min(1.0, self._display_settings.fill_opacity)))
        pen = QPen(outline, max(1.0, self._display_settings.line_width))
        pen.setCosmetic(True)
        path_item.setPen(pen)
        path_item.setBrush(QBrush(fill))
        path_item.setToolTip(image_path)
        return path_item

    def set_debug_candidates(self, candidates: list[object]) -> None:
        self._recycled_debug_cleanup_timer.stop()
        for item in self._debug_candidate_items:
            item.setVisible(False)
            if isinstance(item, QGraphicsPathItem):
                self._recycled_debug_candidate_items.append(item)
            else:
                self.removeItem(item)
        self._debug_candidate_items.clear()
        for candidate in candidates:
            if bool(getattr(candidate, "accepted", False)):
                continue
            bbox = getattr(candidate, "bbox", (0, 0, 0, 0))
            if not isinstance(bbox, (tuple, list)) or len(bbox) < 4:
                continue
            x_coord, y_coord, width, height = (float(value) for value in bbox[:4])
            if width <= 0.0 or height <= 0.0:
                continue

            reason = str(getattr(candidate, "reason", "") or "")
            score = max(0.0, min(100.0, float(getattr(candidate, "score", 0.0) or 0.0)))
            below_threshold = "below_threshold" in reason
            color = QColor("#F59E0B" if below_threshold else "#EF4444")
            rect = QRectF(x_coord, y_coord, width, height)
            path = QPainterPath()
            path.addEllipse(rect)
            if self._recycled_debug_candidate_items:
                path_item = self._recycled_debug_candidate_items.pop()
                path_item.setPath(path)
            else:
                path_item = QGraphicsPathItem(path)
                self.addItem(path_item)
            path_item.setZValue(4.5)
            pen = QPen(color, 2.0, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            path_item.setPen(pen)
            fill = QColor(color)
            fill.setAlpha(36)
            path_item.setBrush(QBrush(fill))
            status = "Ниже порога" if below_threshold else "Отклонён"
            if self._ui_language != "ru":
                status = "Below threshold" if below_threshold else "Rejected"
            path_item.setToolTip(f"{status}\n{reason or '-'}\nScore: {score:.1f}")
            path_item.setVisible(True)
            self._debug_candidate_items.append(path_item)
        if self._recycled_debug_candidate_items:
            self._recycled_debug_cleanup_timer.start()

    def _drain_recycled_debug_candidate_items(self) -> None:
        batch_size = min(32, len(self._recycled_debug_candidate_items))
        for _index in range(batch_size):
            self.removeItem(self._recycled_debug_candidate_items.pop())
        if not self._recycled_debug_candidate_items:
            self._recycled_debug_cleanup_timer.stop()

    def set_metal_overlays(
        self,
        layers: dict[str, list[PolygonData]],
        visibility: dict[str, bool],
    ) -> None:
        for item in self._metal_overlay_items:
            self.removeItem(item)
        self._metal_overlay_items.clear()
        layer_styles: list[tuple[str, str, bool]] = [
            ("rejected", "#EF4444", False),
            ("suspicious", "#EAB308", False),
            ("border", "#3B82F6", False),
            ("wide_pairs_suspicious", "#EAB308", True),
            ("wide_pairs_rejected", "#DC2626", True),
        ]
        z = 2.2
        _ru = self._ui_language == "ru"
        _layer_tip = {
            "rejected": "Отклонён" if _ru else "Rejected",
            "suspicious": "Сомнительный" if _ru else "Suspicious",
            "border": "У границы кадра" if _ru else "Border touch",
            "wide_pairs_suspicious": "Широкий проводник (сомнительно)" if _ru else "Wide trace (suspicious)",
            "wide_pairs_rejected": "Широкий проводник (отклонён)" if _ru else "Wide trace (rejected)",
        }
        for key, color_hex, dashed in layer_styles:
            if not visibility.get(key, False):
                continue
            for poly in layers.get(key) or []:
                path_item = QGraphicsPathItem(_display_path_for_polygon(poly, self._display_settings))
                path_item.setZValue(z)
                c = QColor(color_hex)
                pen = QPen(c, 1.75)
                pen.setCosmetic(True)
                if dashed:
                    pen.setStyle(Qt.PenStyle.DashLine)
                path_item.setPen(pen)
                path_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                reason = str(getattr(poly, "reject_reason", "") or "")
                cap = _layer_tip.get(key, key)
                tip_lines = [cap]
                if reason.strip():
                    tip_lines.append(reason)
                path_item.setToolTip("\n".join(tip_lines))
                path_item.setData(int(Qt.ItemDataRole.UserRole), key)
                path_item.setData(int(Qt.ItemDataRole.UserRole) + 1, reason)
                self.addItem(path_item)
                self._metal_overlay_items.append(path_item)

    def metal_overlay_pick(self, scene_pos: QPointF) -> tuple[str, str] | None:
        """Return ``(layer_key, reject_reason)`` for the topmost metal overlay under ``scene_pos``."""

        for item in self.items(scene_pos):
            if item in self._metal_overlay_items:
                layer = item.data(int(Qt.ItemDataRole.UserRole))
                if layer is None:
                    continue
                reason = item.data(int(Qt.ItemDataRole.UserRole) + 1)
                return (str(layer), str(reason or ""))
        return None

    def set_gradient_overlay(self, image, opacity: float = 0.45) -> None:
        self.clear_gradient_field_arrows()
        if image is None:
            self._gradient_overlay_item.setPixmap(QPixmap())
            self._gradient_overlay_item.hide()
            return
        pixmap = QPixmap.fromImage(cv_to_qimage(image))
        if pixmap.isNull():
            self._gradient_overlay_item.setPixmap(QPixmap())
            self._gradient_overlay_item.hide()
            return
        self._gradient_overlay_item.setPixmap(pixmap)
        self._gradient_overlay_item.setOpacity(max(0.0, min(1.0, float(opacity))))
        self._gradient_overlay_item.setPos(0.0, 0.0)
        self._sync_gradient_overlay_visibility()

    def clear_gradient_overlay(self) -> None:
        self._gradient_overlay_item.setPixmap(QPixmap())
        self._gradient_overlay_item.hide()
        self.clear_gradient_field_arrows()

    def set_gradient_field_arrows(self, arrows: list[tuple[float, float, float, float]]) -> None:
        path = _gradient_arrows_path(arrows)
        self._gradient_arrows_item.setPath(path)
        self._gradient_arrows_has_content = bool(arrows) and not path.isEmpty()
        self._sync_gradient_overlay_visibility()

    def clear_gradient_field_arrows(self) -> None:
        if self._gradient_arrows_has_content or not self._gradient_arrows_item.path().isEmpty():
            self._gradient_arrows_item.setPath(QPainterPath())
        self._gradient_arrows_has_content = False
        self._gradient_arrows_item.hide()

    def set_gradient_overlay_opacity(self, opacity: float) -> None:
        self._gradient_overlay_item.setOpacity(max(0.0, min(1.0, float(opacity))))

    def set_gradient_overlay_visible(self, visible: bool) -> None:
        self._gradient_overlay_user_visible = bool(visible)
        self._sync_gradient_overlay_visibility()

    def gradient_overlay_user_visible(self) -> bool:
        return self._gradient_overlay_user_visible

    def _sync_gradient_overlay_visibility(self) -> None:
        has_content = not self._gradient_overlay_item.pixmap().isNull()
        self._gradient_overlay_item.setVisible(self._gradient_overlay_user_visible and has_content)
        self._gradient_arrows_item.setVisible(self._gradient_overlay_user_visible and self._gradient_arrows_has_content)

    def _update_main_frame(self) -> None:
        path = QPainterPath()
        path.addRect(self._image_rect)
        self._main_frame_item.setPath(path)

    def _update_scene_rect(self) -> None:
        rect = QRectF(self._image_rect)
        if self._neighbor_grid_bounds is not None:
            rect = rect.united(self._neighbor_grid_bounds)
        for item in self._neighbor_frame_items:
            rect = rect.united(item.sceneBoundingRect())
        self.setSceneRect(rect)

    def set_ui_language(self, language: str | None) -> None:
        self._ui_language = active_language(language)

    def warn_invalid_polygon_geometry(self) -> None:
        self.logRequested.emit(tr("polygon_invalid_geometry_log", language=self._ui_language))

    def _log_polygon_commit_rejection(self, reason: str | None) -> None:
        if reason == POLYGON_COMMIT_TOO_SMALL_AREA:
            self.logRequested.emit(tr("polygon_too_small_area_commit_log", language=self._ui_language))
            return
        if reason == POLYGON_COMMIT_TOO_FEW_VERTICES:
            self.logRequested.emit(tr("polygon_need_min_vertices_finish_log", language=self._ui_language))

    def set_display_settings(self, settings: DisplaySettings) -> None:
        self._display_settings = settings
        self._refresh_all_items()

    def set_random_object_colors_enabled(self, enabled: bool) -> None:
        self._random_object_colors_enabled = bool(enabled)
        self._refresh_all_items()

    def set_extra_layers(self, layers: list[dict[str, object]]) -> None:
        for item in self._extra_layer_items:
            self.removeItem(item)
        self._extra_layer_items.clear()
        for index, layer in enumerate(layers):
            if not bool(layer.get("visible", True)):
                continue
            pixmap = layer.get("pixmap")
            if not isinstance(pixmap, QPixmap) or pixmap.isNull():
                continue
            dx = float(layer.get("dx", 0.0) or 0.0)
            dy = float(layer.get("dy", 0.0) or 0.0)
            try:
                opacity = float(layer.get("opacity", 1.0))
            except (TypeError, ValueError):
                opacity = 1.0
            opacity = max(0.0, min(1.0, opacity))
            item = QGraphicsPixmapItem(pixmap)
            item.setZValue(0.8 + index * 0.001)
            item.setOpacity(opacity)
            item.setPos(dx, dy)
            item.setToolTip(str(layer.get("name", "")))
            self.addItem(item)
            self._extra_layer_items.append(item)

    def set_polygon_overlays_visible(self, visible: bool) -> None:
        self._polygon_overlays_visible = bool(visible)
        for polygon_id, item in self._polygon_items.items():
            poly = self._polygons[polygon_id]
            cat = str(getattr(poly, "category", "") or "")
            vis = self._polygon_category_visible.get(cat, True)
            item.setVisible(
                self._polygon_overlays_visible
                and vis
                and polygon_id not in self._zoom_hidden_contact_ids
            )
        zoom_contacts_visible = (
            self._polygon_overlays_visible
            and self._polygon_category_visible.get("via", True)
        )
        for item in self._zoom_contact_composite_items:
            item.setVisible(zoom_contacts_visible)
        self._pending_path_item.setVisible(self._polygon_overlays_visible)
        self._preview_rect_item.setVisible(self._polygon_overlays_visible)

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

    def end_zoom_vector_render_mode(self) -> None:
        if not self._zoom_contact_composite_items and not self._zoom_hidden_contact_ids:
            return
        for polygon_id in self._zoom_hidden_contact_ids:
            polygon = self._polygons.get(polygon_id)
            item = self._polygon_items.get(polygon_id)
            if polygon is None or item is None:
                continue
            category = str(getattr(polygon, "category", "") or "")
            item.setVisible(
                self._polygon_overlays_visible
                and self._polygon_category_visible.get(category, True)
            )
        self._zoom_hidden_contact_ids.clear()
        for composite in self._zoom_contact_composite_items:
            self.removeItem(composite)
        self._zoom_contact_composite_items.clear()

    def polygon_overlays_visible(self) -> bool:
        return self._polygon_overlays_visible

    def get_polygons(self) -> list[PolygonData]:
        return [self._polygons[polygon_id].clone() for polygon_id in sorted(self._polygons)]

    def contact_count(self) -> int:
        return sum(1 for polygon in self._polygons.values() if _is_via_polygon(polygon))

    def selected_contact_count(self) -> int:
        return sum(
            1
            for polygon_id in self._selected_polygon_ids
            if (
                polygon_id in self._polygons
                and _is_via_polygon(self._polygons[polygon_id])
            )
        )

    def selected_object_counts(self) -> tuple[int, int]:
        selected_count = sum(
            1
            for polygon_id in self._selected_polygon_ids
            if polygon_id in self._polygons
        )
        contact_count = self.selected_contact_count()
        return contact_count, selected_count - contact_count

    def set_protect_recognized_vias(self, enabled: bool) -> None:
        self._protect_recognized_vias = bool(enabled)

    def set_minimum_contact_distance(self, distance: float) -> None:
        self._minimum_contact_distance = max(0.0, float(distance))

    def can_add_polygon(self) -> bool:
        return can_add_polygon(self._polygons.values())

    def can_add_via(self) -> bool:
        return can_add_via(self._polygons.values())

    def can_add_polygon_set(self, polygons: list[PolygonData]) -> bool:
        return can_add_polygon_set(self._polygons.values(), polygons)

    def available_editor_tools(self) -> frozenset[EditorTool]:
        return available_editor_tools(self._polygons.values())

    def polygon_is_deletable(self, polygon: PolygonData | None) -> bool:
        if polygon is None:
            return False
        return not (self._protect_recognized_vias and is_recognized_via(polygon))

    def set_vector_geometry_settings(self, settings: VectorGeometrySettings | None) -> None:
        self._vector_geometry_settings = settings if settings is not None else VectorGeometrySettings()
        if self._polygon_items:
            self._refresh_all_items()
        else:
            self._polygons_needing_repair = {}

    def _rebuild_polygons_needing_repair_cache(self) -> None:
        self._polygons_needing_repair = polygons_needing_repair(
            list(self._polygons.values()),
            self._vector_geometry_settings,
        )

    def apply_polygons_needing_repair(
        self,
        reasons: dict[int, list[str]] | None,
        *,
        refresh_items: bool = True,
    ) -> None:
        """Install a precomputed repair map without rescanning geometry."""

        previous_ids = set(self._polygons_needing_repair)
        if reasons is None:
            self._polygons_needing_repair = {}
        else:
            self._polygons_needing_repair = {
                int(polygon_id): list(codes) for polygon_id, codes in reasons.items()
            }
        if not refresh_items or not self._polygon_items:
            return
        changed_ids = previous_ids | set(self._polygons_needing_repair)
        editable_vertex_ids = self._editable_vertex_polygon_ids()
        for polygon_id in changed_ids:
            item = self._polygon_items.get(polygon_id)
            if item is not None:
                self._refresh_polygon_item(polygon_id, item, editable_vertex_ids=editable_vertex_ids)

    def polygon_needs_repair(self, polygon_id: int) -> bool:
        return polygon_id in self._polygons_needing_repair

    def polygons_needing_repair_map(self) -> dict[int, list[str]]:
        return {polygon_id: list(reasons) for polygon_id, reasons in self._polygons_needing_repair.items()}

    def _selection_reference_polygon(self) -> PolygonData | None:
        if self._selected_polygon_id is None:
            return None
        polygon = self._polygons.get(self._selected_polygon_id)
        return polygon.clone() if polygon is not None else None

    def _restore_selection_fallback_if_needed(self, reference: PolygonData | None) -> None:
        if reference is None or self._selected_polygon_id is not None:
            return
        for polygon in self._polygons.values():
            if polygon_equivalent_preserved(polygon, [reference]):
                self.select_polygon(polygon.id)
                return

    def _maybe_push_vector_postprocess(self, undo_text: str) -> None:
        selection_fallback = self._selection_reference_polygon()
        before = self.get_polygons()
        needs_full_cleanup = float(self._vector_geometry_settings.min_hole_area_to_remove_px2) > 0.0
        if needs_full_cleanup:
            final, changed = postprocess_after_editor_mutation(
                before,
                self._vector_geometry_settings,
                frame_width_height=None,
                include_merge=False,
            )
        else:
            final, accepted, changed = postprocess_changed_polygon_edit(
                before,
                self._vector_geometry_settings,
                polygon_id=self._selected_polygon_id,
            )
            if not accepted:
                self.undo_stack.undo()
                self._restore_selection_fallback_if_needed(selection_fallback)
                self.warn_invalid_polygon_geometry()
                return
            if not changed:
                final, changed = postprocess_after_editor_mutation(
                    before,
                    self._vector_geometry_settings,
                    frame_width_height=None,
                    include_merge=False,
                )
        if changed:
            self.undo_stack.push(ReplacePolygonSetCommand(self, before, final, undo_text))
            self._restore_selection_fallback_if_needed(selection_fallback)

    def _bulk_restore_polygons(
        self,
        polygons: list[PolygonData],
        *,
        emit_signal: bool = True,
        selection_fallback: PolygonData | None = None,
    ) -> None:
        self.end_zoom_vector_render_mode()
        prev_primary = self._selected_polygon_id
        prev_selected_ids = set(self._selected_polygon_ids)
        self._recycle_polygon_items()
        self._polygons.clear()
        self._hole_children_by_parent.clear()
        self._polygon_child_ids_by_parent.clear()
        self._hover_conductor_polygon_id = None
        self._vertex_preview_polygon_id = None
        self._selected_polygon_id = None
        self._selected_polygon_ids.clear()
        self._next_polygon_id = 1
        for polygon in polygons:
            self._add_polygon_internal(polygon, emit_signal=False, refresh=False, paint=False)
        self._start_recycled_polygon_cleanup()
        if polygons:
            self._next_polygon_id = max(polygon.id for polygon in polygons) + 1
            new_ids = {polygon.id for polygon in polygons}
            preserved_sorted = sorted(prev_selected_ids & new_ids)
            if preserved_sorted:
                self._selected_polygon_ids = set(preserved_sorted)
                self._selected_polygon_id = (
                    prev_primary if prev_primary in preserved_sorted else preserved_sorted[0]
                )
            elif selection_fallback is not None:
                for polygon in polygons:
                    if polygon_equivalent_preserved(polygon, [selection_fallback]):
                        self._selected_polygon_id = polygon.id
                        self._selected_polygon_ids = {polygon.id}
                        break
                else:
                    self._selected_polygon_id = None
                    self._selected_polygon_ids.clear()
            else:
                self._selected_polygon_id = None
                self._selected_polygon_ids.clear()
        self._refresh_all_items()
        if emit_signal:
            self.polygonsChanged.emit()
            self.activePolygonChanged.emit(self._selected_polygon_id)

    def set_polygons(
        self,
        polygons: list[PolygonData],
        *,
        emit_signal: bool = True,
        repair_reasons: dict[int, list[str]] | None = None,
        scan_repair: bool = True,
    ) -> None:
        self.end_zoom_vector_render_mode()
        self.undo_stack.clear()
        self._recycle_polygon_items()
        self._polygons.clear()
        self._hole_children_by_parent.clear()
        self._polygon_child_ids_by_parent.clear()
        self._hover_conductor_polygon_id = None
        self._vertex_preview_polygon_id = None
        self._selected_polygon_id = None
        self._selected_polygon_ids.clear()
        self._next_polygon_id = 1
        if repair_reasons is not None:
            self._polygons_needing_repair = {
                int(polygon_id): list(codes) for polygon_id, codes in repair_reasons.items()
            }
        elif not scan_repair:
            self._polygons_needing_repair = {}
        will_refresh = bool(polygons) and (
            any(polygon.is_hole for polygon in polygons)
            or not all(_is_via_polygon(polygon) for polygon in polygons)
        )
        for polygon in polygons:
            self._add_polygon_internal(
                polygon,
                emit_signal=False,
                refresh=False,
                paint=not will_refresh,
            )
        self._start_recycled_polygon_cleanup()
        if polygons:
            self._next_polygon_id = max(polygon.id for polygon in polygons) + 1
        if will_refresh:
            self._refresh_all_items(rebuild_repair_cache=scan_repair and repair_reasons is None)
        elif scan_repair and repair_reasons is None and polygons:
            self._rebuild_polygons_needing_repair_cache()
        if emit_signal:
            self.polygonsChanged.emit()
            self.activePolygonChanged.emit(self._selected_polygon_id)

    def sync_conductor_hover_highlight(self, scene_pos: QPointF) -> None:
        if not self._polygon_overlays_visible:
            self._set_hover_conductor_polygon_id(None)
            return
        underneath = self.polygon_at(scene_pos)
        target_id = resolve_conductor_hover_target_id(self._polygons, underneath)
        self._set_hover_conductor_polygon_id(target_id)

    def clear_conductor_hover_highlight(self) -> None:
        self._set_hover_conductor_polygon_id(None)

    def set_show_all_editable_vertices(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._show_all_editable_vertices:
            return
        self._show_all_editable_vertices = enabled
        if enabled:
            self._vertex_preview_polygon_id = None
        self._refresh_all_items()

    def sync_vertex_preview(self, scene_pos: QPointF) -> None:
        if self._show_all_editable_vertices or not self._polygon_overlays_visible:
            self._set_vertex_preview_polygon_id(None)
            return
        self._set_vertex_preview_polygon_id(self.polygon_at(scene_pos))

    def clear_vertex_preview(self) -> None:
        self._set_vertex_preview_polygon_id(None)

    def _set_hover_conductor_polygon_id(self, conductor_id: int | None) -> None:
        if conductor_id is not None and conductor_id not in self._polygons:
            conductor_id = None
        if conductor_id == self._hover_conductor_polygon_id:
            return
        previous_id = self._hover_conductor_polygon_id
        self._hover_conductor_polygon_id = conductor_id
        self._refresh_polygon_items_by_id(previous_id, conductor_id)

    def _set_vertex_preview_polygon_id(self, polygon_id: int | None) -> None:
        if polygon_id is not None and polygon_id not in self._polygons:
            polygon_id = None
        if polygon_id == self._vertex_preview_polygon_id:
            return
        previous_ids = self._editable_vertex_polygon_ids(extra_ids={self._vertex_preview_polygon_id})
        self._vertex_preview_polygon_id = polygon_id
        next_ids = self._editable_vertex_polygon_ids()
        self._refresh_polygon_items_by_id(*(previous_ids | next_ids))

    def selected_polygon_id(self) -> int | None:
        return self._selected_polygon_id

    def select_polygon(self, polygon_id: int | None, *, additive: bool = False) -> None:
        if polygon_id is not None and polygon_id not in self._polygons:
            polygon_id = None
        refresh_ids = set(self._selected_polygon_ids)
        if self._selected_polygon_ids or self._vertex_preview_polygon_id is not None:
            refresh_ids.update(self._editable_vertex_polygon_ids())
        if polygon_id is None:
            if not additive:
                self._selected_polygon_ids.clear()
            self._selected_polygon_id = None
        elif additive:
            if polygon_id in self._selected_polygon_ids:
                self._selected_polygon_ids.remove(polygon_id)
                self._selected_polygon_id = next(iter(sorted(self._selected_polygon_ids)), None)
            else:
                self._selected_polygon_ids.add(polygon_id)
                self._selected_polygon_id = polygon_id
        else:
            self._selected_polygon_ids = {polygon_id}
            self._selected_polygon_id = polygon_id
        editable_vertex_ids = self._editable_vertex_polygon_ids()
        refresh_ids.update(self._selected_polygon_ids)
        refresh_ids.update(editable_vertex_ids)
        for refresh_id in refresh_ids:
            item = self._polygon_items.get(refresh_id)
            if item is not None:
                self._refresh_polygon_selection_item(
                    refresh_id,
                    item,
                    editable_vertex_ids=editable_vertex_ids,
                )
        self.activePolygonChanged.emit(self._selected_polygon_id)

    def select_polygons(self, polygon_ids: list[int]) -> None:
        refresh_ids = set(self._selected_polygon_ids)
        if self._selected_polygon_ids or self._vertex_preview_polygon_id is not None:
            refresh_ids.update(self._editable_vertex_polygon_ids())
        selected_ids = {polygon_id for polygon_id in polygon_ids if polygon_id in self._polygons}
        self._selected_polygon_ids = selected_ids
        self._selected_polygon_id = min(selected_ids) if selected_ids else None
        editable_vertex_ids = self._editable_vertex_polygon_ids()
        refresh_ids.update(selected_ids)
        refresh_ids.update(editable_vertex_ids)
        for refresh_id in refresh_ids:
            item = self._polygon_items.get(refresh_id)
            if item is not None:
                self._refresh_polygon_selection_item(
                    refresh_id,
                    item,
                    editable_vertex_ids=editable_vertex_ids,
                )
        self.activePolygonChanged.emit(self._selected_polygon_id)

    def select_polygons_in_rect(self, rect: QRectF, *, additive: bool = False) -> None:
        normalized = rect.normalized()
        if normalized.width() <= 0.0 or normalized.height() <= 0.0:
            if not additive:
                self.select_polygon(None)
            return
        selected_ids = {
            polygon_id
            for polygon_id, polygon in self._polygons.items()
            if _polygon_data_rect(polygon).intersects(normalized)
        }
        if additive:
            selected_ids.update(self._selected_polygon_ids)
        self.select_polygons(sorted(selected_ids))

    def polygon_snapshot(self, polygon_id: int | None) -> PolygonData | None:
        if polygon_id is None or polygon_id not in self._polygons:
            return None
        return self._polygons[polygon_id].clone()

    def polygon_is_contact(self, polygon_id: int | None) -> bool:
        return (
            polygon_id is not None
            and polygon_id in self._polygons
            and _is_via_polygon(self._polygons[polygon_id])
        )

    def _polygon_id_from_item(self, item: QGraphicsItem | None) -> int | None:
        if isinstance(item, VertexHandleItem):
            return item.polygon_id
        if isinstance(item, EditablePolygonItem):
            return item.polygon_id
        if item is not None:
            parent = item.parentItem()
            if isinstance(parent, EditablePolygonItem):
                return parent.polygon_id
        return None

    def _reset_pick_cycle(self) -> None:
        self._pick_cycle_pos = None
        self._pick_cycle_ids = []
        self._pick_cycle_index = 0

    def polygons_at(self, scene_pos: QPointF) -> list[int]:
        ordered: list[int] = []
        seen: set[int] = set()
        for item in self.items(scene_pos):
            polygon_id = self._polygon_id_from_item(item)
            if polygon_id is None or polygon_id in seen:
                continue
            seen.add(polygon_id)
            ordered.append(polygon_id)
        return ordered

    def polygon_at(self, scene_pos: QPointF, *, cycle: bool = False) -> int | None:
        hits = self.polygons_at(scene_pos)
        if not hits:
            self._reset_pick_cycle()
            return None
        if not cycle or len(hits) == 1:
            self._reset_pick_cycle()
            return hits[0]

        same_spot = (
            self._pick_cycle_pos is not None
            and hypot(scene_pos.x() - self._pick_cycle_pos.x(), scene_pos.y() - self._pick_cycle_pos.y())
            <= _PICK_CYCLE_DISTANCE
            and hits == self._pick_cycle_ids
        )
        if same_spot:
            self._pick_cycle_index = (self._pick_cycle_index + 1) % len(hits)
        else:
            self._pick_cycle_pos = QPointF(scene_pos)
            self._pick_cycle_ids = list(hits)
            self._pick_cycle_index = 0
        return hits[self._pick_cycle_index]

    def polygon_at_nearest_edge(self, scene_pos: QPointF, tolerance: float) -> int | None:
        hit = self.polygon_at(scene_pos)
        if hit is not None:
            return hit
        best_id: int | None = None
        best_distance = float(tolerance)
        target_x = float(scene_pos.x())
        target_y = float(scene_pos.y())
        for polygon_id, polygon in self._polygons.items():
            if len(polygon.points) < 2:
                continue
            for index, start in enumerate(polygon.points):
                end = polygon.points[(index + 1) % len(polygon.points)]
                distance = _distance_to_segment((target_x, target_y), start, end)
                if distance < best_distance:
                    best_distance = distance
                    best_id = polygon_id
        return best_id

    def vertex_at(self, scene_pos: QPointF, tolerance: float) -> tuple[int, int] | None:
        candidate_ids = []
        if self._selected_polygon_id is not None:
            candidate_ids.append(self._selected_polygon_id)
        candidate_ids.extend(
            polygon_id for polygon_id in sorted(self._selected_polygon_ids) if polygon_id != self._selected_polygon_id
        )
        candidate_ids.extend(
            polygon_id for polygon_id in sorted(self._polygons) if polygon_id not in self._selected_polygon_ids
        )
        for polygon_id in candidate_ids:
            polygon = self._polygons[polygon_id]
            for index, (x_coord, y_coord) in enumerate(polygon.points):
                if hypot(scene_pos.x() - x_coord, scene_pos.y() - y_coord) <= tolerance:
                    return polygon_id, index
        return None

    def nearest_vertex_in_polygon(self, polygon_id: int, scene_pos: QPointF) -> tuple[int, int] | None:
        polygon = self._polygons.get(polygon_id)
        if polygon is None or not polygon.points:
            return None
        best_index: int | None = None
        best_distance = float("inf")
        sx, sy = scene_pos.x(), scene_pos.y()
        for index, (x_coord, y_coord) in enumerate(polygon.points):
            distance = hypot(sx - x_coord, sy - y_coord)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is None:
            return None
        return polygon_id, best_index

    def nearest_vertex(self, scene_pos: QPointF) -> tuple[int, int] | None:
        candidate_ids: list[int] = []
        if self._selected_polygon_id is not None:
            candidate_ids.append(self._selected_polygon_id)
        candidate_ids.extend(
            polygon_id for polygon_id in sorted(self._selected_polygon_ids) if polygon_id != self._selected_polygon_id
        )
        candidate_ids.extend(
            polygon_id for polygon_id in sorted(self._polygons) if polygon_id not in self._selected_polygon_ids
        )
        best_hit: tuple[int, int] | None = None
        best_distance = float("inf")
        sx, sy = scene_pos.x(), scene_pos.y()
        for polygon_id in candidate_ids:
            polygon = self._polygons.get(polygon_id)
            if polygon is None:
                continue
            for index, (x_coord, y_coord) in enumerate(polygon.points):
                distance = hypot(sx - x_coord, sy - y_coord)
                if distance < best_distance:
                    best_distance = distance
                    best_hit = (polygon_id, index)
        return best_hit

    def delete_polygon_at(self, scene_pos: QPointF) -> bool:
        polygon_id = self.polygon_at(scene_pos)
        if polygon_id is None:
            return False
        return self.delete_polygon(polygon_id)

    def delete_via_at(self, scene_pos: QPointF) -> bool:
        polygon_id = self.polygon_at(scene_pos)
        polygon = self._polygons.get(polygon_id) if polygon_id is not None else None
        if polygon is None or not _is_via_polygon(polygon) or not self.polygon_is_deletable(polygon):
            return False
        self.delete_polygon(polygon_id)
        return True

    def delete_polygon(self, polygon_id: int | None = None) -> bool:
        target_ids = [polygon_id] if polygon_id is not None else sorted(self._selected_polygon_ids)
        if not target_ids and self._selected_polygon_id is not None:
            target_ids = [self._selected_polygon_id]
        target_ids = [
            target_id
            for target_id in target_ids
            if self.polygon_is_deletable(self._polygons.get(target_id))
        ]
        delete_ids = self._delete_polygon_ids_with_descendants(target_ids)
        delete_ids = [
            target_id
            for target_id in delete_ids
            if self.polygon_is_deletable(self._polygons.get(target_id))
        ]
        target_polygons = [self._polygons[target_id].clone() for target_id in delete_ids if target_id in self._polygons]
        if not target_polygons:
            return False
        before = self.get_polygons()
        remaining = [polygon.clone() for polygon in before if polygon.id not in set(delete_ids)]
        remaining = union_after_removing_polygon_ids(remaining, set(delete_ids))
        self.undo_stack.push(ReplacePolygonSetCommand(self, before, remaining, "Delete polygon"))
        return True

    def _delete_polygon_ids_with_descendants(self, target_ids: list[int | None]) -> list[int]:
        delete_ids: set[int] = set()
        pending = [
            target_id
            for target_id in target_ids
            if target_id in self._polygons and not self._polygons[target_id].is_hole
        ]
        while pending:
            current_id = pending.pop()
            if current_id is None or current_id in delete_ids or current_id not in self._polygons:
                continue
            delete_ids.add(current_id)
            pending.extend(
                child_id
                for child_id in self._polygon_child_ids_by_parent.get(current_id, ())
                if child_id not in delete_ids
            )
        for target_id in target_ids:
            if target_id in self._polygons and self._polygons[target_id].is_hole:
                delete_ids.add(target_id)
        return sorted(delete_ids)

    def selected_polygons(self) -> list[PolygonData]:
        return [
            self._polygons[polygon_id].clone()
            for polygon_id in sorted(self._selected_polygon_ids)
            if polygon_id in self._polygons
        ]

    def selected_deletable_polygons(self) -> list[PolygonData]:
        return [polygon for polygon in self.selected_polygons() if self.polygon_is_deletable(polygon)]

    def antialias_selected_polygons(self, grade: int) -> bool:
        if not self.can_add_polygon():
            return False
        target_ids = {polygon_id for polygon_id in self._selected_polygon_ids if polygon_id in self._polygons}
        if not target_ids and self._selected_polygon_id is not None and self._selected_polygon_id in self._polygons:
            target_ids = {self._selected_polygon_id}
        if not target_ids:
            return False
        before = self.get_polygons()
        after, changed = antialias_polygons(before, grade, only_ids=target_ids)
        if not changed:
            return False
        self.undo_stack.push(ReplacePolygonSetCommand(self, before, after, "Antialias polygons"))
        self.select_polygons(sorted(target_ids))
        return True

    def repair_invalid_polygon_descriptions(self) -> bool:
        before = self.get_polygons()
        settings = self._vector_geometry_settings
        has_keyhole = any(
            (not polygon.is_hole) and polygon.description_invalid_reason() == "repeated_vertex"
            for polygon in before
        )
        if not polygons_needing_repair(before, settings) and not has_keyhole:
            return False
        after = repair_invalid_polygon_descriptions(before, settings)
        before_signature = [
            (polygon.id, polygon.is_hole, polygon.parent_id, tuple(polygon.points)) for polygon in before
        ]
        after_signature = [
            (polygon.id, polygon.is_hole, polygon.parent_id, tuple(polygon.points)) for polygon in after
        ]
        if after_signature == before_signature:
            return False
        self.undo_stack.push(
            ReplacePolygonSetCommand(
                self,
                before,
                after,
                tr("repair_invalid_polygons_undo", language=self._ui_language),
            )
        )
        return True

    def antialias_polygon(self, polygon_id: int | None, grade: int) -> bool:
        if polygon_id is None or polygon_id not in self._polygons:
            return False
        before = self.get_polygons()
        after, changed = antialias_polygons(before, grade, only_ids={polygon_id})
        if not changed:
            self.select_polygon(polygon_id)
            return False
        self.undo_stack.push(ReplacePolygonSetCommand(self, before, after, "Antialias polygon"))
        self.select_polygon(polygon_id)
        self._set_vertex_preview_polygon_id(polygon_id)
        return True

    def antialias_polygons_in_rect(self, rect: QRectF, grade: int) -> bool:
        if not self.can_add_polygon():
            return False
        normalized = rect.normalized()
        if normalized.width() <= 0.0 or normalized.height() <= 0.0:
            return False
        target_ids = {
            polygon_id
            for polygon_id, polygon in self._polygons.items()
            if _polygon_data_rect(polygon).intersects(normalized)
        }
        if not target_ids:
            return False
        before = self.get_polygons()
        after, changed = antialias_polygons(before, grade, only_ids=target_ids)
        if not changed:
            self.select_polygons(sorted(target_ids))
            return False
        self.undo_stack.push(ReplacePolygonSetCommand(self, before, after, "Antialias polygons"))
        self.select_polygons(sorted(target_ids))
        self._set_vertex_preview_polygon_id(None)
        return True

    def add_cloned_polygons_at(
        self,
        polygons: list[PolygonData],
        source_anchor: QPointF,
        target_anchor: QPointF,
    ) -> list[int]:
        if not polygons or not self.can_add_polygon_set(polygons):
            return []
        dx = target_anchor.x() - source_anchor.x()
        dy = target_anchor.y() - source_anchor.y()
        shifted_polygons: list[PolygonData] = []
        for polygon in polygons:
            shifted = polygon.clone()
            shifted.points = integer_points(
                [(float(x) + dx, float(y) + dy) for x, y in polygon.points]
            )
            shifted.area, shifted.perimeter, shifted.bbox = compute_polygon_metrics(
                shifted.points
            )
            shifted_polygons.append(shifted)
        clipped_polygons = self._clip_pasted_polygons_to_inset(shifted_polygons, inset=3.0)
        if not clipped_polygons:
            return []
        if not all(_is_via_polygon(polygon) for polygon in clipped_polygons):
            return self._merge_pasted_polygons(clipped_polygons)
        return self._add_pasted_contacts(clipped_polygons)

    def _add_pasted_contacts(self, polygons: list[PolygonData]) -> list[int]:
        new_polygons: list[PolygonData] = []
        spatial_index = _ContactSpatialIndex(self._minimum_contact_distance)
        for existing in self._polygons.values():
            if not _is_via_polygon(existing):
                continue
            existing_rect = _polygon_data_rect(existing)
            existing_center = existing_rect.center()
            spatial_index.add(
                existing_center.x(),
                existing_center.y(),
                existing_rect,
            )
        for polygon in polygons:
            candidate_rect = _polygon_data_rect(polygon)
            candidate_center = candidate_rect.center()
            center_x = candidate_center.x()
            center_y = candidate_center.y()
            if spatial_index.conflicts(
                center_x,
                center_y,
                candidate_rect,
            ):
                continue
            new_id = self._next_polygon_id
            self._next_polygon_id += 1
            new_polygon = polygon.clone()
            new_polygon.id = new_id
            new_polygon.parent_id = None
            new_polygons.append(new_polygon)
            spatial_index.add(center_x, center_y, candidate_rect)
        if not new_polygons:
            return []
        self.undo_stack.push(
            AddPolygonsCommand(
                self,
                new_polygons,
                "Paste polygons",
                select_after_redo=True,
            )
        )
        return [polygon.id for polygon in new_polygons]

    def _clip_pasted_polygons_to_inset(
        self,
        polygons: list[PolygonData],
        *,
        inset: float,
    ) -> list[PolygonData]:
        image_rect = QRectF(self._image_rect).normalized()
        if image_rect.width() <= 1.0 or image_rect.height() <= 1.0:
            return [polygon.clone() for polygon in polygons]
        clip_rect = image_rect.adjusted(inset, inset, -inset, -inset)
        if clip_rect.width() <= 0.0 or clip_rect.height() <= 0.0:
            return []
        polygon_rects = [
            (polygon, _polygon_data_rect(polygon))
            for polygon in polygons
        ]
        if all(clip_rect.contains(polygon_rect) for _polygon, polygon_rect in polygon_rects):
            return [polygon.clone() for polygon in polygons]
        clip_geom = shapely_box(
            float(clip_rect.left()),
            float(clip_rect.top()),
            float(clip_rect.right()),
            float(clip_rect.bottom()),
        )
        if all(_is_via_polygon(polygon) for polygon in polygons):
            clipped_contacts: list[PolygonData] = []
            for polygon, polygon_rect in polygon_rects:
                if not clip_rect.intersects(polygon_rect):
                    continue
                if clip_rect.contains(polygon_rect):
                    clipped_contacts.append(polygon.clone())
                    continue
                try:
                    polygon_geom = tool_geometry(
                        polygon.points,
                        None,
                        quad_segs=QUAD_SEGS_BRUSH_DEFAULT,
                    )
                    clipped = shapely_to_polygon_data_list(
                        unary_union(make_valid(polygon_geom.intersection(clip_geom)))
                    )
                except Exception:
                    clipped = []
                for candidate in clipped:
                    candidate.category = polygon.category
                    candidate.shape_hint = polygon.shape_hint
                    clipped_contacts.append(candidate)
            return clipped_contacts
        try:
            polygon_map = {polygon.id: polygon.clone() for polygon in polygons}
            pasted_region = region_geometry(polygon_map, list(polygon_map))
            clipped = shapely_to_polygon_data_list(
                unary_union(make_valid(pasted_region.intersection(clip_geom)))
            )
        except Exception:
            return []
        reference = next((polygon for polygon in polygons if not polygon.is_hole), polygons[0])
        for candidate in clipped:
            candidate.category = reference.category
            candidate.shape_hint = reference.shape_hint
        return clipped

    def _merge_pasted_polygons(self, pasted_polygons: list[PolygonData]) -> list[int]:
        try:
            pasted_map = {polygon.id: polygon.clone() for polygon in pasted_polygons}
            pasted_region = region_geometry(pasted_map, list(pasted_map))
        except Exception:
            return []
        overlapping_root_ids: list[int] = []
        for polygon_id, polygon in self._polygons.items():
            if polygon.is_hole:
                continue
            try:
                existing_region = region_geometry(
                    self._polygons,
                    self._polygon_family_ids(polygon_id),
                )
            except ValueError:
                continue
            if existing_region.intersects(pasted_region):
                overlapping_root_ids.append(polygon_id)
        if not overlapping_root_ids:
            return self._add_pasted_polygon_set(pasted_polygons)
        replaced_ids = self._render_polygon_ids(overlapping_root_ids)
        try:
            if replaced_ids:
                existing_region = region_geometry(self._polygons, replaced_ids)
                merged_region = unary_union(
                    make_valid(existing_region.union(pasted_region))
                )
            else:
                merged_region = unary_union(make_valid(pasted_region))
            rebuilt = shapely_to_polygon_data_list(merged_region)
        except Exception:
            return []
        if not rebuilt:
            return []
        reference = (
            self._polygons[overlapping_root_ids[0]]
            if overlapping_root_ids
            else next(
                (polygon for polygon in pasted_polygons if not polygon.is_hole),
                pasted_polygons[0],
            )
        )
        for candidate in rebuilt:
            candidate.category = reference.category
            candidate.shape_hint = reference.shape_hint
        rebuilt = self._assign_polygon_ids(rebuilt, replaced_ids)
        before = self.get_polygons()
        replaced_id_set = set(replaced_ids)
        after = [
            polygon.clone()
            for polygon in before
            if polygon.id not in replaced_id_set
        ]
        after.extend(polygon.clone() for polygon in rebuilt)
        after.sort(key=lambda polygon: polygon.id)
        self.undo_stack.push(
            ReplacePolygonSetCommand(self, before, after, "Paste and merge polygons")
        )
        selected_ids = [polygon.id for polygon in rebuilt]
        self.select_polygons(selected_ids)
        return selected_ids

    def _add_pasted_polygon_set(
        self,
        polygons: list[PolygonData],
    ) -> list[int]:
        id_map: dict[int, int] = {}
        new_polygons: list[PolygonData] = []
        for polygon in polygons:
            new_id = self._next_polygon_id
            self._next_polygon_id += 1
            id_map[polygon.id] = new_id
            cloned = polygon.clone()
            cloned.id = new_id
            new_polygons.append(cloned)
        for polygon in new_polygons:
            polygon.parent_id = (
                None
                if polygon.parent_id is None
                else id_map.get(polygon.parent_id)
            )
        self.undo_stack.push(
            AddPolygonsCommand(
                self,
                new_polygons,
                "Paste polygons",
                select_after_redo=True,
            )
        )
        return [polygon.id for polygon in new_polygons]

    def add_vertex_at(self, polygon_id: int, scene_pos: QPointF) -> bool:
        if not self.can_add_polygon() or polygon_id not in self._polygons:
            return False
        insert_index = self._nearest_segment_insert_index(polygon_id, scene_pos)
        new_point = integer_point((scene_pos.x(), scene_pos.y()))
        points = self.polygon_points(polygon_id)
        insert_at = max(0, min(len(points), insert_index))
        trial = list(points)
        trial.insert(insert_at, new_point)
        collapsed = collapse_redundant_polyline_vertices(trial, closed=True, min_vertices=3)
        if len(collapsed) <= len(points):
            return False
        if not is_valid_closed_polygon_ring(trial):
            self.warn_invalid_polygon_geometry()
            return False
        self.undo_stack.push(AddVertexCommand(self, polygon_id, insert_index, new_point))
        processed, accepted, _changed = postprocess_changed_polygon_edit(
            self.get_polygons(),
            self._vector_geometry_settings,
            polygon_id=polygon_id,
        )
        if not accepted:
            self.undo_stack.undo()
            self.warn_invalid_polygon_geometry()
            return False
        if _changed:
            before = self.get_polygons()
            self.undo_stack.push(ReplacePolygonSetCommand(self, before, processed, "Vector geometry cleanup"))
        self.select_polygon(polygon_id)
        return True

    def delete_vertex_at(self, scene_pos: QPointF, tolerance: float) -> bool:
        if not self.can_add_polygon():
            return False
        hit = self.vertex_at(scene_pos, tolerance)
        if hit is None:
            return False
        polygon_id, vertex_index = hit
        polygon = self._polygons[polygon_id]
        if len(polygon.points) <= 3:
            self.logRequested.emit(tr("polygon_min_vertices_log", language=self._ui_language))
            return False
        before = self.get_polygons()
        after_delete = [item.clone() for item in before]
        target = next((item for item in after_delete if item.id == polygon_id), None)
        if target is None or vertex_index < 0 or vertex_index >= len(target.points):
            return False
        points = list(target.points)
        points.pop(vertex_index)
        target.points = integer_points(points)
        area, perimeter, bbox = compute_polygon_metrics(target.points)
        target.area = float(area)
        target.perimeter = float(perimeter)
        target.bbox = bbox
        after_delete = collapse_redundant_vertices_in_polygons(after_delete)
        after_delete = dissolve_self_intersecting_polygons(after_delete)
        after_delete = collapse_redundant_vertices_in_polygons(after_delete)
        self.undo_stack.push(ReplacePolygonSetCommand(self, before, after_delete, "Delete vertex"))
        focus_id = resolve_focus_id_after_geometry_pass(before, polygon_id, after_delete)
        self.select_polygon(focus_id)
        return True

    def polygon_points(self, polygon_id: int) -> list[tuple[float, float]]:
        polygon = self._polygons.get(polygon_id)
        if polygon is None:
            return []
        return integer_points(polygon.points)

    def has_polygon(self, polygon_id: int | None) -> bool:
        return polygon_id is not None and polygon_id in self._polygons

    def preview_vertex_move(self, polygon_id: int, vertex_index: int, point: QPointF) -> None:
        self._set_vertex_internal(polygon_id, vertex_index, integer_point((point.x(), point.y())), emit_signal=False)

    def preview_polygon_move(self, polygon_id: int, points: list[tuple[float, float]]) -> None:
        self._replace_polygon_points_internal(polygon_id, points, emit_signal=False)

    def start_pending_polygon(self, *, for_brush: bool = False) -> None:
        self._pending_points.clear()
        self._pending_cursor = None
        self._pending_polyline_for_brush = bool(for_brush)
        self._update_pending_path()

    def append_brush_vertex(self, scene_pos: QPointF, brush_diameter: float) -> None:
        del brush_diameter
        nx, ny = integer_point((scene_pos.x(), scene_pos.y()))
        # Screen-pixel spacing is enforced in the view; keep only exact duplicates here.
        if self._pending_points and hypot(nx - self._pending_points[-1][0], ny - self._pending_points[-1][1]) < 1e-6:
            return
        self._pending_points.append((nx, ny))
        self._update_pending_path()

    def replace_pending_points(self, points: list[tuple[float, float]]) -> None:
        self._pending_points = integer_points(points)
        self._pending_cursor = None
        self._update_pending_path()

    def append_pending_point(self, scene_pos: QPointF) -> None:
        point = integer_point((scene_pos.x(), scene_pos.y()))
        if (
            self._pending_points
            and hypot(point[0] - self._pending_points[-1][0], point[1] - self._pending_points[-1][1]) < 1.0
        ):
            return
        trial = [*self._pending_points, point]
        if not self._pending_polyline_for_brush and not is_valid_open_polyline_last_edge(trial):
            self.warn_invalid_polygon_geometry()
            return
        self._pending_points.append(point)
        self._pending_points = collapse_redundant_polyline_vertices(
            self._pending_points,
            closed=False,
            min_vertices=2,
        )
        self._update_pending_path()

    def update_pending_cursor(self, scene_pos: QPointF) -> None:
        if not self._pending_points:
            return
        self._pending_cursor = integer_point((scene_pos.x(), scene_pos.y()))
        self._update_pending_path()

    def cancel_pending_polygon(self) -> None:
        self._pending_points.clear()
        self._pending_cursor = None
        self._pending_polyline_for_brush = False
        self._update_pending_path()
        self.clear_preview_rect()

    def finish_pending_polygon(self) -> bool:
        if not self.can_add_polygon():
            self.cancel_pending_polygon()
            return False
        self._pending_points = collapse_redundant_polyline_vertices(
            self._pending_points,
            closed=True,
            min_vertices=3,
        )
        acceptable, reason = polygon_commit_acceptability(self._pending_points)
        if not acceptable:
            if reason == POLYGON_COMMIT_TOO_FEW_VERTICES:
                self.cancel_pending_polygon()
                self._log_polygon_commit_rejection(reason)
                return False
            if reason == POLYGON_COMMIT_INVALID_RING:
                self.warn_invalid_polygon_geometry()
                return False
            self._log_polygon_commit_rejection(reason)
            return False
        area, perimeter, bbox = compute_polygon_metrics(self._pending_points)
        polygon = PolygonData(
            id=self._next_polygon_id,
            points=integer_points(self._pending_points),
            is_hole=False,
            parent_id=None,
            shape_hint="manual_outline",
            area=area,
            perimeter=perimeter,
            bbox=bbox,
        )
        clipped_polygons = self._clip_authored_polygon_to_image(polygon)
        if not clipped_polygons:
            self.cancel_pending_polygon()
            return False
        clipped_polygons = [
            clipped_polygon
            for clipped_polygon in clipped_polygons
            if not self._is_small_inner_candidate(clipped_polygon.points, thickness=None)
        ]
        if not clipped_polygons:
            self.cancel_pending_polygon()
            return False
        for clipped_polygon in clipped_polygons:
            self._add_or_merge_polygon(clipped_polygon)
        self.cancel_pending_polygon()
        return True

    def pending_last_point(self) -> QPointF | None:
        if not self._pending_points:
            return None
        x_coord, y_coord = self._pending_points[-1]
        return QPointF(x_coord, y_coord)

    def pending_points_snapshot(self) -> list[tuple[float, float]]:
        return [(float(x_coord), float(y_coord)) for x_coord, y_coord in self._pending_points]

    def has_pending_polygon(self) -> bool:
        return bool(self._pending_points)

    def set_preview_rect(self, start: QPointF, end: QPointF) -> None:
        rect = QRectF(start, end).normalized()
        path = QPainterPath()
        path.addRect(rect)
        self._preview_rect_item.setPath(path)

    def preview_delete_vertices_in_rect(self, start: QPointF, end: QPointF) -> None:
        rect = QRectF(start, end).normalized()
        self.set_preview_rect(start, end)
        if rect.width() < 1.0 and rect.height() < 1.0:
            if self._delete_area_highlight_ids:
                self._delete_area_highlight_ids.clear()
                self._refresh_all_items()
            return
        highlighted = {
            polygon_id
            for polygon_id, polygon in self._polygons.items()
            if any(rect.contains(QPointF(x_coord, y_coord)) for x_coord, y_coord in polygon.points)
        }
        if highlighted != self._delete_area_highlight_ids:
            self._delete_area_highlight_ids = highlighted
            self._refresh_all_items()

    def clear_preview_rect(self) -> None:
        self._preview_rect_item.setPath(QPainterPath())
        if self._delete_area_highlight_ids:
            self._delete_area_highlight_ids.clear()
            self._refresh_all_items()

    def set_measurement(self, start: QPointF, end: QPointF, label_text: str = "") -> None:
        path = QPainterPath()
        path.moveTo(start)
        path.lineTo(end)
        self._measurement_item.setPath(path)
        self._set_measurement_marker(self._measurement_start_marker, start)
        self._set_measurement_marker(self._measurement_end_marker, end)
        if label_text:
            self._measurement_label_item.setText(label_text)
            self._measurement_label_item.setPos(_measurement_label_position(start, end))
            self._measurement_label_item.show()
        else:
            self._measurement_label_item.hide()

    def clear_measurement(self) -> None:
        self._measurement_item.setPath(QPainterPath())
        self._measurement_start_marker.hide()
        self._measurement_end_marker.hide()
        self._measurement_label_item.hide()

    def set_brush_cursor(self, scene_pos: QPointF | None, thickness: float, visible: bool) -> None:
        if not visible or scene_pos is None:
            self._brush_cursor_item.hide()
            return
        radius = max(1.0, float(thickness)) / 2.0
        self._brush_cursor_item.setRect(
            QRectF(scene_pos.x() - radius, scene_pos.y() - radius, radius * 2.0, radius * 2.0)
        )
        self._brush_cursor_item.show()

    def set_via_cursor(self, scene_pos: QPointF | None, width: float, height: float, visible: bool) -> None:
        if not visible or scene_pos is None:
            self._via_cursor_item.hide()
            return
        rect = _centered_rect(scene_pos, width, height)
        path = QPainterPath()
        if normalize_via_display_mode(self._display_settings.via_display_mode) == "rectangle":
            path.addRect(rect)
        else:
            path.addEllipse(rect)
        blocked = self._via_rect_overlaps_existing(rect.normalized())
        color = QColor("#EF4444" if blocked else "#A78BFA")
        pen = QPen(color, 1.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        brush = QColor(color)
        brush.setAlpha(42 if not blocked else 58)
        self._via_cursor_item.setPen(pen)
        self._via_cursor_item.setBrush(QBrush(brush))
        self._via_cursor_item.setPath(path)
        self._via_cursor_item.show()

    def hide_tool_cursors(self) -> None:
        self._brush_cursor_item.hide()
        self._via_cursor_item.hide()

    def add_via_at(self, scene_pos: QPointF, width: float, height: float) -> bool:
        if not self.can_add_via():
            return False
        rect = _centered_rect(scene_pos, width, height).normalized()
        if rect.width() < 1.0 or rect.height() < 1.0:
            return False
        image_rect = QRectF(self._image_rect).normalized()
        if image_rect.width() > 1.0 and image_rect.height() > 1.0 and not image_rect.contains(rect):
            return False
        if self._via_rect_overlaps_existing(rect):
            return False
        points = [
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
            (rect.right(), rect.bottom()),
            (rect.left(), rect.bottom()),
        ]
        points = integer_points(points)
        area, perimeter, bbox = compute_polygon_metrics(points)
        polygon = PolygonData(
            id=self._next_polygon_id,
            points=points,
            is_hole=False,
            parent_id=None,
            category="via",
            shape_hint="box",
            area=area,
            perimeter=perimeter,
            bbox=bbox,
        )
        self.undo_stack.push(AddPolygonCommand(self, polygon, select_after_redo=True))
        return True

    def _via_rect_overlaps_existing(self, rect: QRectF) -> bool:
        for polygon in self._polygons.values():
            if not _is_via_polygon(polygon):
                continue
            if rect.intersects(_polygon_data_rect(polygon)):
                return True
        return False

    def add_rectangle_polygon(self, start: QPointF, end: QPointF, erase: bool = False) -> bool:
        if not self.can_add_polygon():
            self.clear_preview_rect()
            return False
        rect = QRectF(start, end).normalized()
        if rect.width() < 1.0 or rect.height() < 1.0:
            self.clear_preview_rect()
            return False
        points = [
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
            (rect.right(), rect.bottom()),
            (rect.left(), rect.bottom()),
        ]
        points = integer_points(points)
        acceptable, reason = polygon_commit_acceptability(points)
        if not acceptable:
            self.clear_preview_rect()
            if reason == POLYGON_COMMIT_INVALID_RING:
                self.warn_invalid_polygon_geometry()
            else:
                self._log_polygon_commit_rejection(reason)
            return False
        area, perimeter, bbox = compute_polygon_metrics(points)
        polygon = PolygonData(
            id=self._next_polygon_id,
            points=points,
            is_hole=False,
            parent_id=None,
            area=area,
            perimeter=perimeter,
            bbox=bbox,
        )
        clipped_polygons = self._clip_authored_polygon_to_image(polygon, allocate_ids=not erase)
        if not clipped_polygons:
            self.clear_preview_rect()
            return False
        polygon = clipped_polygons[0]
        if erase:

            erased_ok = bool(
                self._subtract_shape_from_scene(points=list(polygon.points), thickness=None, label="Erase rectangle")
            )

            self.clear_preview_rect()


            return erased_ok

        if self._is_small_inner_candidate(polygon.points, thickness=None):
            self.clear_preview_rect()
            return False

        self._add_or_merge_polygon(polygon, label="Add rectangle")

        self.clear_preview_rect()


        return True

    def _clip_authored_polygon_to_image(
        self,
        polygon: PolygonData,
        *,
        allocate_ids: bool = True,
    ) -> list[PolygonData]:
        image_rect = QRectF(self._image_rect).normalized()
        if image_rect.width() <= 1.0 or image_rect.height() <= 1.0:
            out = [polygon.clone()]
        else:
            try:
                polygon_geom = tool_geometry(polygon.points, None, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
                image_geom = shapely_box(
                    float(image_rect.left()),
                    float(image_rect.top()),
                    float(image_rect.right()),
                    float(image_rect.bottom()),
                )
                clipped_geom = unary_union(make_valid(polygon_geom.intersection(image_geom)))
                out = shapely_to_polygon_data_list(clipped_geom)
            except Exception:
                out = [polygon.clone()]
        if not out:
            return []
        id_map: dict[int, int] = {}
        for index, clipped in enumerate(out):
            clipped.category = polygon.category
            clipped.shape_hint = polygon.shape_hint
            if allocate_ids:
                if index == 0:
                    new_id = polygon.id
                    self._next_polygon_id = max(self._next_polygon_id, int(new_id) + 1)
                else:
                    new_id = self._next_polygon_id
                    self._next_polygon_id += 1
            else:
                new_id = polygon.id if index == 0 else self._next_polygon_id
                if index > 0:
                    self._next_polygon_id += 1
            id_map[clipped.id] = new_id
            clipped.id = new_id
        for clipped in out:
            clipped.parent_id = None if clipped.parent_id is None else id_map.get(clipped.parent_id)
        return out

    def _clip_authored_polygons_to_image(self, polygons: list[PolygonData]) -> list[PolygonData]:
        image_rect = QRectF(self._image_rect).normalized()
        if image_rect.width() <= 1.0 or image_rect.height() <= 1.0:
            return [polygon.clone() for polygon in polygons]
        if all(
            image_rect.contains(QPointF(float(x_coord), float(y_coord)))
            for polygon in polygons
            for x_coord, y_coord in polygon.points
        ):
            # Preserve the complete root/hole family when clipping is a no-op.
            # Clipping rings independently turns every hole into an outer ring.
            return [polygon.clone() for polygon in polygons]
        try:
            by_id = {polygon.id: polygon.clone() for polygon in polygons}
            family_geom = region_geometry(by_id, list(by_id))
            image_geom = shapely_box(
                float(image_rect.left()),
                float(image_rect.top()),
                float(image_rect.right()),
                float(image_rect.bottom()),
            )
            clipped_family = shapely_to_polygon_data_list(
                unary_union(make_valid(family_geom.intersection(image_geom)))
            )
            if not clipped_family:
                return []
            reference = next((polygon for polygon in polygons if not polygon.is_hole), polygons[0])
            for candidate in clipped_family:
                candidate.category = reference.category
                candidate.shape_hint = reference.shape_hint
            return self._assign_polygon_ids(clipped_family, [polygon.id for polygon in polygons])
        except Exception:
            # Fall back to the older per-ring path only for malformed legacy
            # families that cannot be represented by GEOS.
            pass
        clipped: list[PolygonData] = []
        used_ids: set[int] = set()
        for polygon in polygons:
            for candidate in self._clip_authored_polygon_to_image(polygon, allocate_ids=False):
                if candidate.id in used_ids:
                    candidate.id = self._next_polygon_id
                    self._next_polygon_id += 1
                used_ids.add(candidate.id)
                clipped.append(candidate)
        return clipped

    def add_brush_stroke(self, points: list[tuple[float, float]], thickness: float, erase: bool = False) -> bool:
        if not self.can_add_polygon():
            self.cancel_pending_polygon()
            return False
        if len(points) < 1:
            self.cancel_pending_polygon()
            return False
        if erase:
            changed = self._subtract_shape_from_scene(points=list(points), thickness=thickness, label="Erase brush stroke")
            self.cancel_pending_polygon()
            return changed
        if self._is_small_inner_candidate(list(points), thickness=thickness):
            self.cancel_pending_polygon()
            return False
        merged_polygons, overlapping_ids = self._merge_shape_into_scene(points=list(points), thickness=thickness)
        if merged_polygons is None:
            self.cancel_pending_polygon()
            return False
        if not merged_polygons:
            self.cancel_pending_polygon()
            return False
        merged_polygons = self._clip_authored_polygons_to_image(merged_polygons)
        if not merged_polygons:
            self.cancel_pending_polygon()
            return False
        self.undo_stack.beginMacro("Add brush stroke")
        try:
            for polygon_id in overlapping_ids:
                self.undo_stack.push(DeletePolygonCommand(self, self._polygons[polygon_id]))
            for polygon in merged_polygons:
                self.undo_stack.push(AddPolygonCommand(self, polygon))
        finally:
            self.undo_stack.endMacro()
        self.select_polygon(merged_polygons[0].id)
        self._maybe_push_vector_postprocess("Vector geometry cleanup")
        self.cancel_pending_polygon()
        return True

    def add_trace_stroke(self, points: list[tuple[float, float]], width: float, erase: bool = False) -> bool:
        if not self.can_add_polygon():
            self.cancel_pending_polygon()
            return False
        if len(points) < 2:
            self.cancel_pending_polygon()
            return False
        label = "Erase trace" if erase else "Add trace"
        if erase:
            changed = self._subtract_shape_from_scene(points=list(points), thickness=width, label=label)
            self.cancel_pending_polygon()
            return changed
        if self._is_small_inner_candidate(list(points), thickness=width):
            self.cancel_pending_polygon()
            return False
        merged_polygons, overlapping_ids = self._merge_shape_into_scene(points=list(points), thickness=width)
        if merged_polygons is None:
            self.cancel_pending_polygon()
            return False
        if not merged_polygons:
            self.cancel_pending_polygon()
            return False
        merged_polygons = self._clip_authored_polygons_to_image(merged_polygons)
        if not merged_polygons:
            self.cancel_pending_polygon()
            return False
        for polygon in merged_polygons:
            if not str(getattr(polygon, "category", "") or ""):
                polygon.category = "conductor"
            if str(getattr(polygon, "shape_hint", "") or "") in {"", "polygon"}:
                polygon.shape_hint = "trace_pen"
        self.undo_stack.beginMacro(label)
        try:
            for polygon_id in overlapping_ids:
                self.undo_stack.push(DeletePolygonCommand(self, self._polygons[polygon_id]))
            for polygon in merged_polygons:
                self.undo_stack.push(AddPolygonCommand(self, polygon))
        finally:
            self.undo_stack.endMacro()
        self.select_polygon(merged_polygons[0].id)
        self._maybe_push_vector_postprocess("Vector geometry cleanup")
        self.cancel_pending_polygon()
        return True

    def subtract_pending_polygon(self) -> bool:
        if not self.can_add_polygon():
            self.cancel_pending_polygon()
            return False
        acceptable, reason = polygon_commit_acceptability(self._pending_points)
        if not acceptable:
            if reason == POLYGON_COMMIT_TOO_FEW_VERTICES:
                self.cancel_pending_polygon()
                return False
            if reason == POLYGON_COMMIT_INVALID_RING:
                self.warn_invalid_polygon_geometry()
                return False
            self._log_polygon_commit_rejection(reason)
            return False
        changed = self._subtract_shape_from_scene(points=self._pending_points, thickness=None, label="Erase polygon")
        self.cancel_pending_polygon()
        return changed

    def delete_vertices_in_rect(self, rect: QRectF) -> int:
        if not self.can_add_polygon():
            return 0
        normalized = rect.normalized()
        if normalized.width() < 1.0 and normalized.height() < 1.0:
            return 0
        min_ring_vertices = 3
        remaining_by_id: dict[int, list[tuple[float, float]]] = {}
        for polygon_id in sorted(self._polygons):
            polygon = self._polygons[polygon_id]
            remaining_points = [
                (float(x_coord), float(y_coord))
                for x_coord, y_coord in polygon.points
                if not normalized.contains(QPointF(x_coord, y_coord))
            ]
            if len(remaining_points) == len(polygon.points):
                continue
            remaining_by_id[polygon_id] = collapse_redundant_polyline_vertices(
                remaining_points,
                closed=True,
                min_vertices=0,
            )

        ids_to_delete: set[int] = set()
        for polygon_id, remaining_points in remaining_by_id.items():
            if len(remaining_points) >= min_ring_vertices:
                continue
            polygon = self._polygons[polygon_id]
            if polygon.is_hole:
                ids_to_delete.add(polygon_id)
            else:
                ids_to_delete.update(self._polygon_family_ids(polygon_id))

        point_updates = [
            (polygon_id, remaining_points)
            for polygon_id, remaining_points in remaining_by_id.items()
            if polygon_id not in ids_to_delete and len(remaining_points) >= min_ring_vertices
        ]
        if not point_updates and not ids_to_delete:
            return 0

        before = self.get_polygons()
        remaining: list[PolygonData] = []
        deleted = 0
        remaining_lookup = dict(point_updates)
        for polygon in before:
            if polygon.id in ids_to_delete:
                deleted += len(polygon.points)
                continue
            clone = polygon.clone()
            if clone.id in remaining_lookup:
                updated_points = remaining_lookup[clone.id]
                deleted += len(clone.points) - len(updated_points)
                clone.points = updated_points
                area, perimeter, bbox = compute_polygon_metrics(clone.points)
                clone.area = float(area)
                clone.perimeter = float(perimeter)
                clone.bbox = bbox
            remaining.append(clone)
        remaining = dissolve_self_intersecting_polygons(remaining)
        remaining = union_after_removing_polygon_ids(remaining, ids_to_delete)
        self.undo_stack.push(ReplacePolygonSetCommand(self, before, remaining, "Delete vertices in area"))
        return deleted

    def _add_or_merge_polygon(self, polygon: PolygonData, label: str = "Add polygon") -> None:
        if not self._polygons:
            self.undo_stack.push(AddPolygonCommand(self, polygon, select_after_redo=True))
            self._maybe_push_vector_postprocess("Vector geometry cleanup")
            return
        merged_polygons, overlapping_ids = self._merge_shape_into_scene(points=polygon.points, thickness=None)
        if merged_polygons is None:
            # Boolean union failed — keep the authored ring so simple shapes still land in the undo stack.
            self.undo_stack.push(AddPolygonCommand(self, polygon, select_after_redo=True))
            self._maybe_push_vector_postprocess("Vector geometry cleanup")
            return
        if not overlapping_ids:
            self.undo_stack.push(AddPolygonCommand(self, polygon, select_after_redo=True))
            self._maybe_push_vector_postprocess("Vector geometry cleanup")
            return
        if not merged_polygons:
            self.undo_stack.push(AddPolygonCommand(self, polygon, select_after_redo=True))
            self._maybe_push_vector_postprocess("Vector geometry cleanup")
            return
        self.undo_stack.beginMacro(label)
        try:
            for polygon_id in overlapping_ids:
                self.undo_stack.push(DeletePolygonCommand(self, self._polygons[polygon_id]))
            for merged_polygon in merged_polygons:
                self.undo_stack.push(AddPolygonCommand(self, merged_polygon))
        finally:
            self.undo_stack.endMacro()
        self.select_polygon(merged_polygons[0].id)
        self._maybe_push_vector_postprocess("Vector geometry cleanup")

    def _subtract_shape_from_scene(
        self,
        *,
        points: list[tuple[float, float]],
        thickness: float | None,
        label: str,
    ) -> bool:
        if not points or not self.can_add_polygon():
            return False
        shape_bbox = _bbox_from_points(points, padding=(round(thickness / 2.0) + 2) if thickness else 2)
        overlapping_ids = self._find_overlapping_polygon_ids(points=points, thickness=thickness, shape_bbox=shape_bbox)
        if not overlapping_ids:
            return False
        render_ids = self._render_polygon_ids(overlapping_ids)
        touched_ids = self._touched_polygon_ids(render_ids, shape_bbox, points, thickness)
        preserved_polygons = self._preserved_polygons(render_ids, touched_ids, overlapping_ids)
        remaining_polygons, err_msg = self._apply_tool_boolean_to_polygon_subset(
            render_ids=list(render_ids),
            points=list(points),
            thickness=thickness,
            erase=True,
        )
        if err_msg is not None:
            detail = err_msg.strip() or repr(err_msg)

            prefix = tr("brush_boolean_failed_log", language=self._ui_language)

            self.logRequested.emit(f"{prefix} ({detail})")
            return False
        assert remaining_polygons is not None
        rebuilt_polygons = self._restore_preserved_polygons(remaining_polygons, render_ids, preserved_polygons)

        self.undo_stack.beginMacro(label)
        try:
            for polygon_id in render_ids:
                self.undo_stack.push(DeletePolygonCommand(self, self._polygons[polygon_id]))
            for polygon in rebuilt_polygons:
                self.undo_stack.push(AddPolygonCommand(self, polygon))
        finally:
            self.undo_stack.endMacro()

        if rebuilt_polygons:
            self.select_polygon(rebuilt_polygons[0].id)
        else:
            self.select_polygon(None)
        self._maybe_push_vector_postprocess("Vector geometry cleanup")
        return True

    def _merge_shape_into_scene(
        self,
        *,
        points: list[tuple[float, float]],
        thickness: float | None,
    ) -> tuple[list[PolygonData] | None, list[int]]:
        if not points:
            return [], []
        shape_bbox = _bbox_from_points(points, padding=(round(thickness / 2.0) + 2) if thickness else 2)
        overlapping_ids = self._find_overlapping_polygon_ids(points=points, thickness=thickness, shape_bbox=shape_bbox)
        render_ids = self._render_polygon_ids(overlapping_ids)
        touched_ids = self._touched_polygon_ids(render_ids, shape_bbox, points, thickness)
        preserved_polygons = self._preserved_polygons(render_ids, touched_ids, overlapping_ids)
        merged_contours, err_msg = self._apply_tool_boolean_to_polygon_subset(
            render_ids=list(render_ids),
            points=list(points),
            thickness=thickness,
            erase=False,
        )
        if err_msg is not None:
            prefix = tr("brush_boolean_failed_log", language=self._ui_language)

            detail = err_msg.strip() or repr(err_msg)

            self.logRequested.emit(f"{prefix} ({detail})")
            return None, render_ids

        assert merged_contours is not None
        return self._restore_preserved_polygons(merged_contours, render_ids, preserved_polygons), render_ids

    def _is_small_inner_candidate(self, points: list[tuple[float, float]], thickness: float | None) -> bool:
        min_area = float(self._vector_geometry_settings.min_hole_area_to_remove_px2)
        if min_area <= 0.0 or not self._polygons or not points:
            return False
        try:
            candidate_shape = tool_geometry(points, thickness, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
        except Exception:
            return False
        if candidate_shape.is_empty or float(candidate_shape.area) >= min_area:
            return False
        candidate_shape = make_valid(candidate_shape)
        for polygon in self._polygons.values():
            if _is_via_polygon(polygon) or len(polygon.points) < 3:
                continue
            try:
                containing_shape = polygon_paint_footprint_geom(polygon.points)
            except Exception:
                continue
            if containing_shape.is_empty:
                continue
            if make_valid(containing_shape).covers(candidate_shape):
                return True
        return False

    def _apply_tool_boolean_to_polygon_subset(
        self,
        *,
        render_ids: list[int],
        points: list[tuple[float, float]],
        thickness: float | None,
        erase: bool,
    ) -> tuple[list[PolygonData] | None, str | None]:
        try:
            brush_tool = tool_geometry(points, thickness, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
            if thickness is not None:
                brush_tool = simplify_polygonal_geometry(brush_tool)
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
        try:
            base_region = region_geometry(self._polygons, render_ids)
        except ValueError:
            return None, "invalid region geometry"
        result_geom, err_msg = apply_boolean(base_region, brush_tool, subtract=erase)
        if err_msg is not None:
            return None, err_msg
        assert result_geom is not None

        polygons_list_out = shapely_to_polygon_data_list(result_geom)
        return polygons_list_out, None

    def _find_overlapping_polygon_ids(
        self,
        *,
        points: list[tuple[float, float]],
        thickness: float | None,
        shape_bbox: tuple[int, int, int, int],
    ) -> list[int]:
        overlapping_ids: list[int] = []
        try:
            tool_shape = tool_geometry(points, thickness, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
        except Exception:
            return []
        if tool_shape.is_empty:
            return []

        brush_bounds_xy = tuple(tool_shape.bounds)

        buffered_tool = tool_shape.buffer(1e-7)

        for polygon_id, polygon in self._polygons.items():
            if polygon.is_hole:
                continue
            if not _bboxes_intersect(shape_bbox, polygon.bbox):
                continue
            if not bbox_intersects_geom_bounds(brush_bounds_xy, polygon.bbox):
                continue
            try:
                family_geometry_shape = region_geometry(self._polygons, self._polygon_family_ids(polygon_id))
            except ValueError:
                continue
            if family_geometry_shape.intersects(buffered_tool):
                overlapping_ids.append(polygon_id)
        return overlapping_ids

    def _allocate_polygon_ids(self, overlapping_ids: list[int], count: int) -> list[int]:
        ids: list[int] = []
        reusable_ids = sorted(overlapping_ids)
        for index in range(count):
            if index < len(reusable_ids):
                ids.append(reusable_ids[index])
            else:
                ids.append(self._next_polygon_id)
                self._next_polygon_id += 1
        return ids

    def _update_pending_path(self) -> None:

        brush_width = self._pending_brush_width if self._pending_polyline_for_brush else float(self._pending_path_item.pen().widthF())

        if self._pending_polyline_for_brush:

            outline_color = QColor("#F7B801")

            if self._pending_points and brush_width >= 1.0:
                preview_points = list(self._pending_points)
                if self._pending_cursor is not None:
                    preview_points.append(self._pending_cursor)

                fill_color = QColor("#F7B801")

                fill_color.setAlpha(55)

                self._pending_path_item.setBrush(QBrush(fill_color))

                outline_pen = QPen(outline_color, 1.25)

                outline_pen.setCosmetic(True)

                self._pending_path_item.setPen(outline_pen)

                self._pending_path_item.setPath(self._tool_preview_path(preview_points, brush_width))

                return

            if len(self._pending_points) == 1 and self._pending_cursor is None:

                radius = max(1.0, brush_width) / 2.0

                hub_x, hub_y = self._pending_points[0]

                dot = QPainterPath()

                dot.addEllipse(QRectF(hub_x - radius, hub_y - radius, radius * 2.0, radius * 2.0))

                fill_color = QColor("#F7B801")

                fill_color.setAlpha(55)

                self._pending_path_item.setBrush(QBrush(fill_color))

                outline_pen = QPen(outline_color, 1.25)

                outline_pen.setCosmetic(True)

                self._pending_path_item.setPen(outline_pen)

                self._pending_path_item.setPath(dot)

                return

        preview_points = list(self._pending_points)
        if self._pending_cursor is not None:
            preview_points.append(self._pending_cursor)
        can_preview_closed = len(preview_points) >= 3
        preview_valid = False
        if can_preview_closed:
            preview_valid, _reason = polygon_commit_acceptability(preview_points)
        preview_color = QColor("#38BDF8" if preview_valid else "#EF4444" if can_preview_closed else "#F7B801")

        dashed_pen_setup = QPen(preview_color, 1.5, Qt.PenStyle.DashLine)

        dashed_pen_setup.setCosmetic(True)

        dashed_pen_setup.setCapStyle(Qt.PenCapStyle.RoundCap)

        dashed_pen_setup.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

        dashed_pen_setup.setWidthF(self._pending_path_item.pen().widthF())

        if self._pending_polyline_for_brush:

            dashed_pen_setup.setCosmetic(False)

        self._pending_path_item.setPen(dashed_pen_setup)

        if can_preview_closed:
            fill_color = QColor(preview_color)
            fill_color.setAlpha(48)
            self._pending_path_item.setBrush(QBrush(fill_color))
        else:
            self._pending_path_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))

        backbone = QPainterPath()

        if preview_points:

            head = preview_points[0]

            backbone.moveTo(head[0], head[1])

            for tail in preview_points[1:]:
                backbone.lineTo(tail[0], tail[1])

            if can_preview_closed:
                backbone.closeSubpath()

        self._pending_path_item.setPath(backbone)

    def _tool_preview_path(self, points: list[tuple[float, float]], thickness: float) -> QPainterPath:
        path = QPainterPath()
        try:
            geom = extract_polygonal_union(
                make_valid(tool_geometry(points, float(thickness), quad_segs=QUAD_SEGS_BRUSH_DEFAULT))
            )
        except Exception:
            return path
        if geom.is_empty:
            return path

        def add_ring(coords: object) -> None:
            ring_points = list(coords)
            if len(ring_points) < 2:
                return
            ring = QPainterPath()
            start_x, start_y = float(ring_points[0][0]), float(ring_points[0][1])
            ring.moveTo(start_x, start_y)
            for x_coord, y_coord in ring_points[1:]:
                ring.lineTo(float(x_coord), float(y_coord))
            ring.closeSubpath()
            path.addPath(ring)

        def add_polygon(poly: ShapelyPolygon) -> None:
            add_ring(poly.exterior.coords)
            for interior in poly.interiors:
                add_ring(interior.coords)

        geom_type = getattr(geom, "geom_type", "")
        if geom_type == "Polygon":
            add_polygon(geom)
        elif geom_type == "MultiPolygon":
            for part in geom.geoms:
                add_polygon(part)
        path.setFillRule(Qt.FillRule.WindingFill)
        return path

    def refresh_polygon_items(self) -> None:
        self._refresh_all_items()

    def _rebuild_outer_pick_z_ranks(self) -> None:
        outer_ids = sorted(
            polygon_id for polygon_id, polygon in self._polygons.items() if not polygon.is_hole
        )
        outer_ids.sort(key=lambda polygon_id: float(self._polygons[polygon_id].area))
        self._outer_pick_z_rank = {polygon_id: rank for rank, polygon_id in enumerate(outer_ids)}

    def _pick_z_value_for_polygon(self, polygon: PolygonData) -> float:
        if polygon.is_hole:
            return _HOLE_POLYGON_ITEM_Z
        rank = self._outer_pick_z_rank.get(polygon.id, 0)
        count = len(self._outer_pick_z_rank)
        if count <= 1:
            return _OUTER_POLYGON_ITEM_Z + _OUTER_PICK_Z_SPAN
        return _OUTER_POLYGON_ITEM_Z + _OUTER_PICK_Z_SPAN * (1.0 - rank / (count - 1))

    def _refresh_all_items(self, *, rebuild_repair_cache: bool = True) -> None:
        if rebuild_repair_cache:
            self._rebuild_polygons_needing_repair_cache()
        self._rebuild_outer_pick_z_ranks()
        editable_vertex_ids = self._editable_vertex_polygon_ids()
        for polygon_id, item in self._polygon_items.items():
            self._refresh_polygon_item(polygon_id, item, editable_vertex_ids=editable_vertex_ids)

    def _refresh_polygon_items_by_id(self, *polygon_ids: int | None) -> None:
        editable_vertex_ids = self._editable_vertex_polygon_ids()
        for polygon_id in {polygon_id for polygon_id in polygon_ids if polygon_id is not None}:
            item = self._polygon_items.get(polygon_id)
            if item is not None:
                self._refresh_polygon_item(polygon_id, item, editable_vertex_ids=editable_vertex_ids)

    def _refresh_polygon_selection_item(
        self,
        polygon_id: int,
        item: EditablePolygonItem,
        *,
        editable_vertex_ids: set[int],
    ) -> None:
        polygon = self._polygons.get(polygon_id)
        if polygon is not None and _is_via_polygon(polygon):
            item.update_selection_appearance(
                self._display_settings,
                selected=polygon_id in self._selected_polygon_ids,
                custom_color=(
                    self._via_score_color(polygon)
                    or self._object_color_for(polygon_id)
                ),
                needs_repair=self.polygon_needs_repair(polygon_id),
            )
            return
        self._refresh_polygon_item(
            polygon_id,
            item,
            editable_vertex_ids=editable_vertex_ids,
        )

    def _refresh_polygon_item(
        self,
        polygon_id: int,
        item: EditablePolygonItem | None = None,
        *,
        editable_vertex_ids: set[int] | None = None,
    ) -> None:
        polygon = self._polygons.get(polygon_id)
        if polygon is None:
            return
        if item is None:
            item = self._polygon_items.get(polygon_id)
        if item is None:
            return
        conductor_hover_highlight = (
            self._hover_conductor_polygon_id is not None
            and polygon_id == self._hover_conductor_polygon_id
            and polygon_id not in self._selected_polygon_ids
        ) or (polygon_id in self._delete_area_highlight_ids and polygon_id not in self._selected_polygon_ids)
        item.update_from_polygon(
            polygon,
            self._display_settings,
            selected=polygon_id in self._selected_polygon_ids,
            cutout_polygons=self._cutout_polygons_for(polygon_id),
            custom_color=self._via_score_color(polygon) or self._object_color_for(polygon_id),
            conductor_hover_highlight=conductor_hover_highlight,
            preview_vertices=polygon_id
            in (
                editable_vertex_ids
                if editable_vertex_ids is not None
                else self._editable_vertex_polygon_ids()
            ),
            needs_repair=self.polygon_needs_repair(polygon_id),
        )
        item.setZValue(self._pick_z_value_for_polygon(polygon))
        cat = str(getattr(polygon, "category", "") or "")
        vis = self._polygon_category_visible.get(cat, True)
        item.setVisible(bool(vis) and self._polygon_overlays_visible)
        if self.polygon_needs_repair(polygon_id):
            item.setToolTip(tr("invalid_polygon_description_tooltip", language=self._ui_language))
        else:
            item.setToolTip("")

    @staticmethod
    def _via_score_color(polygon: PolygonData) -> str | None:
        if not _is_via_polygon(polygon):
            return None
        score = getattr(polygon, "recognition_score", None)
        if score is None:
            return None
        normalized = max(0.0, min(1.0, float(score) / 100.0))
        # HSL hue 0° is red, 60° yellow and 120° green.
        return QColor.fromHslF(normalized / 3.0, 0.82, 0.50).name()

    def set_polygon_category_visible(self, category: str, visible: bool) -> None:
        self._polygon_category_visible[str(category)] = bool(visible)
        for polygon_id, polygon in self._polygons.items():
            if str(getattr(polygon, "category", "") or "") != str(category):
                continue
            item = self._polygon_items.get(polygon_id)
            if item is not None:
                item.setVisible(
                    bool(visible)
                    and self._polygon_overlays_visible
                    and polygon_id not in self._zoom_hidden_contact_ids
                )
        if str(category) == "via":
            for item in self._zoom_contact_composite_items:
                item.setVisible(bool(visible) and self._polygon_overlays_visible)

    def _object_color_for(self, polygon_id: int) -> str | None:
        if not self._random_object_colors_enabled:
            return None
        if polygon_id not in self._object_colors:
            self._object_colors[polygon_id] = _stable_object_color(polygon_id)
        return self._object_colors[polygon_id]

    def _cutout_polygons_for(self, polygon_id: int) -> list[PolygonData]:
        polygon = self._polygons.get(polygon_id)
        if polygon is None or polygon.is_hole:
            return []
        return self._hole_children_by_parent.get(polygon_id, [])

    def _rebuild_polygon_relationship_cache(self) -> None:
        self._hole_children_by_parent.clear()
        self._polygon_child_ids_by_parent.clear()
        for polygon in self._polygons.values():
            self._index_polygon_relationship(polygon)

    def _index_polygon_relationship(self, polygon: PolygonData) -> None:
        if polygon.parent_id is None:
            return
        self._polygon_child_ids_by_parent.setdefault(
            polygon.parent_id,
            set(),
        ).add(polygon.id)
        if polygon.is_hole:
            self._hole_children_by_parent.setdefault(polygon.parent_id, []).append(polygon)

    def _unindex_polygon_relationship(self, polygon: PolygonData | None) -> None:
        if polygon is None or polygon.parent_id is None:
            return
        child_ids = self._polygon_child_ids_by_parent.get(polygon.parent_id)
        if child_ids is not None:
            child_ids.discard(polygon.id)
            if not child_ids:
                self._polygon_child_ids_by_parent.pop(polygon.parent_id, None)
        if not polygon.is_hole:
            return
        children = self._hole_children_by_parent.get(polygon.parent_id)
        if not children:
            return
        self._hole_children_by_parent[polygon.parent_id] = [child for child in children if child.id != polygon.id]
        if not self._hole_children_by_parent[polygon.parent_id]:
            self._hole_children_by_parent.pop(polygon.parent_id, None)

    def _render_polygon_ids(self, overlapping_ids: list[int]) -> list[int]:
        render_ids: list[int] = []
        for polygon_id in overlapping_ids:
            for family_id in self._polygon_family_ids(polygon_id):
                if family_id not in render_ids:
                    render_ids.append(family_id)
        return sorted(render_ids)

    def _editable_vertex_polygon_ids(self, extra_ids: set[int | None] | None = None) -> set[int]:
        if self._show_all_editable_vertices:
            return {
                polygon_id
                for polygon_id, polygon in self._polygons.items()
                if not _is_via_polygon(polygon)
            }
        active_ids: set[int] = {
            polygon_id for polygon_id in self._selected_polygon_ids if polygon_id in self._polygons
        }
        if self._vertex_preview_polygon_id in self._polygons:
            active_ids.add(self._vertex_preview_polygon_id)
        for polygon_id in extra_ids or set():
            if polygon_id in self._polygons:
                active_ids.add(polygon_id)
        editable_ids: set[int] = set()
        for polygon_id in active_ids:
            if _is_via_polygon(self._polygons[polygon_id]):
                continue
            editable_ids.update(self._polygon_edit_family_ids(polygon_id))
        return editable_ids

    def _polygon_edit_family_ids(self, polygon_id: int) -> list[int]:
        root_id = self._polygon_root_id(polygon_id)
        if root_id is None:
            return []
        return self._polygon_family_ids(root_id)

    def _polygon_root_id(self, polygon_id: int) -> int | None:
        current_id: int | None = polygon_id
        seen: set[int] = set()
        while current_id is not None and current_id in self._polygons and current_id not in seen:
            seen.add(current_id)
            polygon = self._polygons[current_id]
            if polygon.parent_id is None or polygon.parent_id not in self._polygons:
                return current_id
            current_id = polygon.parent_id
        return polygon_id if polygon_id in self._polygons else None

    def _touched_polygon_ids(
        self,
        candidate_ids: list[int],
        shape_bbox: tuple[int, int, int, int],
        points: list[tuple[float, float]],
        thickness: float | None,
    ) -> set[int]:
        touched_ids: set[int] = set()
        try:
            tool_shape = tool_geometry(points, thickness, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
            buffered_tool = tool_shape.buffer(1e-7)
        except Exception:
            return touched_ids

        if tool_shape.is_empty:
            return touched_ids

        for polygon_id in candidate_ids:

            polygon = self._polygons.get(polygon_id)

            if polygon is None or not _bboxes_intersect(shape_bbox, polygon.bbox):
                continue
            polygon_region = polygon_footprint_geom(polygon.points)

            if polygon_region.is_empty:

                continue
            if polygon_region.intersects(buffered_tool):
                touched_ids.add(polygon_id)
        return touched_ids

    def _preserved_polygons(
        self,
        render_ids: list[int],
        touched_ids: set[int],
        root_ids: list[int],
    ) -> list[PolygonData]:
        root_id_set = set(root_ids)
        return [
            self._polygons[polygon_id].clone()
            for polygon_id in render_ids
            if polygon_id not in root_id_set and polygon_id not in touched_ids
        ]

    def _polygon_family_ids(self, polygon_id: int) -> list[int]:
        family_ids: list[int] = []
        pending = [polygon_id]
        while pending:
            current_id = pending.pop()
            if current_id in family_ids or current_id not in self._polygons:
                continue
            family_ids.append(current_id)
            pending.extend(self._polygon_child_ids_by_parent.get(current_id, ()))
        return sorted(family_ids)

    def _assign_polygon_ids(
        self,
        polygons: list[PolygonData],
        replaced_ids: list[int],
        *,
        reserved_ids: set[int] | None = None,
    ) -> list[PolygonData]:
        reserved = reserved_ids or set()
        reusable_ids = [
            polygon_id
            for polygon_id in sorted(
                replaced_ids,
                key=lambda current_id: (
                    -abs(float(self._polygons[current_id].area)) if current_id in self._polygons else 0.0,
                    current_id,
                ),
            )
            if polygon_id not in reserved
        ]
        sorted_polygons = sorted(polygons, key=lambda polygon: -abs(float(polygon.area)))
        allocated_ids = self._allocate_polygon_ids(reusable_ids, len(sorted_polygons))
        id_map = {
            polygon.id: allocated_id for polygon, allocated_id in zip(sorted_polygons, allocated_ids, strict=False)
        }
        for polygon, allocated_id in zip(sorted_polygons, allocated_ids, strict=False):
            polygon.parent_id = None if polygon.parent_id is None else id_map.get(polygon.parent_id)
            polygon.id = allocated_id
        return sorted(sorted_polygons, key=lambda polygon: polygon.id)

    def _restore_preserved_polygons(
        self,
        rebuilt_polygons: list[PolygonData],
        deleted_ids: list[int],
        preserved_polygons: list[PolygonData],
    ) -> list[PolygonData]:
        if not preserved_polygons:
            return self._assign_polygon_ids(rebuilt_polygons, deleted_ids)
        filtered_rebuilt = [
            polygon
            for polygon in rebuilt_polygons
            if not self._matches_any_preserved_polygon(polygon, preserved_polygons)
        ]
        assigned_rebuilt = self._assign_polygon_ids(
            filtered_rebuilt,
            deleted_ids,
            reserved_ids={polygon.id for polygon in preserved_polygons},
        )
        restored_polygons = assigned_rebuilt + [polygon.clone() for polygon in preserved_polygons]
        self._repair_preserved_parent_links(restored_polygons, preserved_polygons)
        return sorted(restored_polygons, key=lambda polygon: polygon.id)

    def _matches_any_preserved_polygon(self, polygon: PolygonData, preserved_polygons: list[PolygonData]) -> bool:
        return polygon_equivalent_preserved(polygon, preserved_polygons)

    def _repair_preserved_parent_links(
        self,
        polygons: list[PolygonData],
        preserved_polygons: list[PolygonData],
    ) -> None:
        non_hole_polygons = [polygon for polygon in polygons if not polygon.is_hole]
        for preserved in preserved_polygons:
            restored = next((polygon for polygon in polygons if polygon.id == preserved.id), None)
            if restored is None:
                continue
            parent = _smallest_containing_polygon(restored, non_hole_polygons)
            restored.parent_id = None if parent is None else parent.id

    def _create_polygon_snapshot(self, polygon_id: int, points: list[tuple[float, float]]) -> PolygonData:
        existing = self._polygons[polygon_id]
        area, perimeter, bbox = compute_polygon_metrics(points)
        return PolygonData(
            id=existing.id,
            points=integer_points(points),
            is_hole=existing.is_hole,
            parent_id=existing.parent_id,
            category=existing.category,
            shape_hint=existing.shape_hint,
            area=area,
            perimeter=perimeter,
            bbox=bbox,
            recognition_score=existing.recognition_score,
            reject_reason=existing.reject_reason,
        )

    def _add_polygon_internal(
        self,
        polygon: PolygonData,
        emit_signal: bool = True,
        refresh: bool = True,
        *,
        paint: bool | None = None,
    ) -> None:
        if polygon.id in self._polygon_items:
            self._remove_polygon_internal(polygon.id, emit_signal=False, refresh=False)
        self._polygons[polygon.id] = polygon.clone()
        self._index_polygon_relationship(self._polygons[polygon.id])
        self._next_polygon_id = max(self._next_polygon_id, polygon.id + 1)
        should_paint = True if paint is None else bool(paint)
        custom_color = self._via_score_color(self._polygons[polygon.id]) or self._object_color_for(
            polygon.id
        )
        if self._recycled_polygon_items:
            item = self._recycled_polygon_items.pop()
            if should_paint:
                item.update_from_polygon(
                    self._polygons[polygon.id],
                    self._display_settings,
                    selected=False,
                    cutout_polygons=self._cutout_polygons_for(polygon.id) if refresh else None,
                    custom_color=custom_color,
                    needs_repair=self.polygon_needs_repair(polygon.id),
                )
            else:
                item.bind_polygon_data(self._polygons[polygon.id])
        else:
            item = EditablePolygonItem(
                self._polygons[polygon.id],
                self._display_settings,
                custom_color=custom_color,
                paint=should_paint,
            )
            self.addItem(item)
        if refresh:
            self._refresh_all_items()
        else:
            self._rebuild_outer_pick_z_ranks()
            item.setZValue(self._pick_z_value_for_polygon(self._polygons[polygon.id]))
        category = str(getattr(polygon, "category", "") or "")
        item.setVisible(
            self._polygon_category_visible.get(category, True)
            and self._polygon_overlays_visible
        )
        self._polygon_items[polygon.id] = item
        if emit_signal:
            self.polygonsChanged.emit()

    def _recycle_polygon_items(self) -> None:
        self._recycled_polygon_cleanup_timer.stop()
        for item in self._polygon_items.values():
            item.setVisible(False)
            self._recycled_polygon_items.append(item)
        self._polygon_items.clear()

    def _start_recycled_polygon_cleanup(self) -> None:
        if self._recycled_polygon_items:
            self._recycled_polygon_cleanup_timer.start()

    def _drain_recycled_polygon_items(self) -> None:
        # Removing a very large previous result synchronously can block the
        # display of a replacement frame for tens of seconds. Dispose of the
        # unused items in small event-loop batches after the new result is ready.
        batch_size = min(32, len(self._recycled_polygon_items))
        for _index in range(batch_size):
            self.removeItem(self._recycled_polygon_items.pop())
        if not self._recycled_polygon_items:
            self._recycled_polygon_cleanup_timer.stop()

    def _remove_polygon_internal(self, polygon_id: int, emit_signal: bool = True, refresh: bool = True) -> None:
        item = self._polygon_items.pop(polygon_id, None)
        self._unindex_polygon_relationship(self._polygons.pop(polygon_id, None))
        if self._hover_conductor_polygon_id == polygon_id:
            self._hover_conductor_polygon_id = None
        if self._vertex_preview_polygon_id == polygon_id:
            self._vertex_preview_polygon_id = None
        if item is not None:
            self.removeItem(item)
        if self._selected_polygon_id == polygon_id:
            self._selected_polygon_id = None
            self.activePolygonChanged.emit(None)
        self._selected_polygon_ids.discard(polygon_id)
        if refresh:
            self._refresh_all_items()
        if emit_signal:
            self.polygonsChanged.emit()

    def _remove_polygons_internal(
        self,
        polygon_ids: list[int],
        *,
        emit_signal: bool = True,
        refresh: bool = True,
    ) -> None:
        for polygon_id in polygon_ids:
            self._remove_polygon_internal(polygon_id, emit_signal=False, refresh=False)
        if refresh:
            self._refresh_all_items()
        if emit_signal:
            self.polygonsChanged.emit()

    def _add_polygons_internal(
        self,
        polygons: list[PolygonData],
        *,
        emit_signal: bool = True,
        refresh: bool = True,
    ) -> None:
        for polygon in polygons:
            self._add_polygon_internal(
                polygon,
                emit_signal=False,
                refresh=False,
                paint=not refresh,
            )
        if refresh:
            self._refresh_all_items()
        if emit_signal:
            self.polygonsChanged.emit()

    def _replace_polygon_points_internal(
        self, polygon_id: int, points: list[tuple[float, float]], emit_signal: bool = True
    ) -> None:
        if polygon_id not in self._polygons:
            return
        refresh_ids = set(self._polygon_edit_family_ids(polygon_id))
        self._unindex_polygon_relationship(self._polygons.get(polygon_id))
        self._polygons[polygon_id] = self._create_polygon_snapshot(polygon_id, points)
        self._index_polygon_relationship(self._polygons[polygon_id])
        refresh_ids.update(self._polygon_edit_family_ids(polygon_id))
        self._refresh_polygon_items_by_id(*refresh_ids)
        if emit_signal:
            self.polygonsChanged.emit()

    def _set_vertex_internal(
        self,
        polygon_id: int,
        vertex_index: int,
        point: tuple[float, float],
        emit_signal: bool = True,
    ) -> None:
        if polygon_id not in self._polygons:
            return
        points = self.polygon_points(polygon_id)
        if vertex_index < 0 or vertex_index >= len(points):
            return
        new_point = integer_point(point)
        closed_duplicate_endpoint = (
            len(points) > 2 and hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) < 1e-5
        )
        points[vertex_index] = new_point
        if closed_duplicate_endpoint:
            if vertex_index == 0:
                points[-1] = new_point
            elif vertex_index == len(points) - 1:
                points[0] = new_point
        self._replace_polygon_points_internal(polygon_id, points, emit_signal=emit_signal)

    def _insert_vertex_internal(
        self,
        polygon_id: int,
        insert_index: int,
        point: tuple[float, float],
        emit_signal: bool = True,
    ) -> None:
        points = self.polygon_points(polygon_id)
        insert_at = max(0, min(len(points), insert_index))
        points.insert(insert_at, integer_point(point))
        self._replace_polygon_points_internal(polygon_id, points, emit_signal=emit_signal)

    def _remove_vertex_internal(self, polygon_id: int, vertex_index: int, emit_signal: bool = True) -> None:
        points = self.polygon_points(polygon_id)
        if len(points) <= 3:
            return
        points.pop(vertex_index)
        self._replace_polygon_points_internal(polygon_id, points, emit_signal=emit_signal)

    def _nearest_segment_insert_index(self, polygon_id: int, scene_pos: QPointF) -> int:
        polygon = self._polygons[polygon_id]
        points = polygon.points
        if len(points) < 2:
            return len(points)
        target_x = scene_pos.x()
        target_y = scene_pos.y()
        best_index = 1
        best_distance = float("inf")
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            distance = _distance_to_segment((target_x, target_y), start, end)
            if distance < best_distance:
                best_distance = distance
                best_index = index + 1
        return best_index

    def _set_measurement_marker(self, marker: QGraphicsEllipseItem, point: QPointF) -> None:
        radius = 3.0
        marker.setPos(point)
        marker.setRect(QRectF(-radius, -radius, radius * 2.0, radius * 2.0))
        marker.show()
