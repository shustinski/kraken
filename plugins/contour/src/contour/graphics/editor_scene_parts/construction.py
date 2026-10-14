"""editor_scene: construction."""
from __future__ import annotations
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QPen,
    QUndoStack,
)
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)
from ...application.processing import DisplaySettings
from ...application.vector_geometry_postprocess import (
    VectorGeometrySettings,
)
from contour.domain import PolygonData
from contour.graphics_items import (
    EditablePolygonItem,
)
from contour.i18n import active_language
from ...infrastructure.polygon_change_profiler import PolygonChangeProfile

from contour.graphics.editor_scene_parts.module_state import _DEFERRED_GEOMETRY_REPAIR_DELAY_MS

from contour.graphics.editor_scene_parts.polygon_spatial_index import _PolygonSpatialIndex


class ConstructionMixin:
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
        self._overlap_repair_patch_timer = QTimer(self)
        self._overlap_repair_patch_timer.setSingleShot(True)
        self._overlap_repair_patch_timer.setInterval(0)
        self._overlap_repair_patch_timer.timeout.connect(self._run_pending_overlap_repair_patch)
        self._pending_overlap_repair_roots: set[int] = set()
        self._deferred_geometry_repair_timer = QTimer(self)
        self._deferred_geometry_repair_timer.setSingleShot(True)
        self._deferred_geometry_repair_timer.setInterval(_DEFERRED_GEOMETRY_REPAIR_DELAY_MS)
        self._deferred_geometry_repair_timer.timeout.connect(self._run_pending_deferred_geometry_repair)
        self._pending_deferred_geometry_repair: _DeferredGeometryRepairRequest | None = None
        self._applying_deferred_geometry_repair = False
        self._hole_children_by_parent: dict[int, list[PolygonData]] = {}
        self._polygon_child_ids_by_parent: dict[int, set[int]] = {}
        self._polygon_spatial_index = _PolygonSpatialIndex()
        self._polygon_hover_contours: dict[int, np.ndarray] = {}
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
        self._object_colors_cache_signature: tuple[object, ...] | None = None
        self._object_colors_cache: dict[int, str] = {}
        self._object_colors_refresh_pending = False
        self._hover_conductor_polygon_id: int | None = None
        self._vertex_preview_polygon_id: int | None = None
        self._move_target_preview: tuple[str, int, int] | None = None
        self._last_geometry_changed_polygon_id: int | None = None
        self._pending_workspace_layer_patch: tuple[frozenset[int], list[PolygonData]] | None = None
        self._show_all_editable_vertices = False
        self._delete_area_highlight_ids: set[int] = set()
        self._vector_geometry_settings = VectorGeometrySettings()
        self._polygons_needing_repair: dict[int, list[str]] = {}
        self._polygon_change_profile: PolygonChangeProfile | None = None

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
        self._polygon_edit_preview_id: int | None = None
        self._polygon_edit_preview_points: list[tuple[float, float]] | None = None
        self._polygon_edit_preview_origin_points: list[tuple[float, float]] | None = None
        self._polygon_edit_preview_item = QGraphicsPathItem()
        self._polygon_edit_preview_item.setZValue(9)
        self._polygon_edit_preview_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.addItem(self._polygon_edit_preview_item)
        self._polygon_edit_preview_item.hide()
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
