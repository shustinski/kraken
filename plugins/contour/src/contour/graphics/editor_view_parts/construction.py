"""editor_view: construction."""
from __future__ import annotations
from collections import OrderedDict
from collections.abc import Callable
from PyQt6.QtCore import (
    QPoint,
    QPointF,
    Qt,
    QThreadPool,
    QTimer,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsView
from ...application.frame_lod import FixedGridFrameLayout, PyramidFrameStore
from ...application.vector_geometry_postprocess import (
    VectorGeometrySettings,
)
from contour.domain import PolygonData
from ...infrastructure.contact_placement_profiler import (
    ContactDragProfile,
    MoveVertexToolActivationProfile,
    SceneZoomProfile,
)
from contour.graphics.editor_hotkeys import shortcut_sequence, tool_shortcut_sequence
from contour.graphics.editor_scene import PolygonEditorScene
from contour.graphics.minimap_widget import MinimapWidget
from contour.graphics.tool_mode_logic import (
    available_editor_tools,
)
from contour.graphics.tools import BrushMode, DeleteVertexMode, EditorTool, PolygonCreateMode

from contour.graphics.editor_view_parts.module_state import (
    _PYRAMID_VISIBLE_UPDATE_MS,
    _WHEEL_ZOOM_COALESCE_MS,
    _ZOOM_ANIMATION_FRAME_MS,
)


class ConstructionMixin:
    def __init__(self, parent=None) -> None:
        self._editor_scene = PolygonEditorScene()
        super().__init__(self._editor_scene, parent)
        self._opengl_viewport_enabled = self._configure_opengl_viewport()
        self._steady_render_hints = QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        self._zooming_render_hints = QPainter.RenderHint(0)
        self.setRenderHints(self._steady_render_hints)
        # Manual tools invalidate one small overlay or polygon. Repainting the
        # complete image and vector layer for every pointer sample makes input
        # latency scale with viewport size rather than with the edited region.
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, False)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(QBrush(QColor("#171B22")))
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.horizontalScrollBar().valueChanged.connect(self._schedule_pyramid_visible_update)
        self.verticalScrollBar().valueChanged.connect(self._schedule_pyramid_visible_update)
        self.horizontalScrollBar().valueChanged.connect(self._schedule_gradient_field_arrows_update)
        self.verticalScrollBar().valueChanged.connect(self._schedule_gradient_field_arrows_update)

        self._tool = EditorTool.SELECT
        self._available_tools = available_editor_tools(())
        self._contact_recognition_mode = False
        self._conductor_recognition_mode = False
        self._polygon_create_mode = PolygonCreateMode.RECTANGLE
        self._brush_mode = BrushMode.ANGLED
        self._brush_thickness = 12.0
        self._trace_mode = BrushMode.ANGLED
        self._trace_width = 12.0
        self._via_width = 12.0
        self._via_height = 12.0
        self._antialias_grade = 1
        self._delete_vertex_mode = DeleteVertexMode.AREA
        self._select_press_polygon_id: int | None = None
        self._select_press_start: QPointF | None = None
        self._drag_kind: str | None = None
        self._drag_polygon_id: int | None = None
        self._drag_vertex_index: int | None = None
        self._drag_edge_index: int | None = None
        self._drag_origin_points: list[tuple[float, float]] | None = None
        self._drag_preview_points: list[tuple[float, float]] | None = None
        self._drag_start_scene_pos: QPointF | None = None
        self._last_pointer_scene_pos: QPointF | None = None
        self._drag_erases = False
        self._pending_polygon_erases: bool | None = None
        self._middle_pan_active = False
        self._middle_pan_last_viewport: QPointF | None = None
        self._polygon_overlay_hide_holds: set[str] = set()
        self._gradient_overlay_hide_holds: set[str] = set()
        self._polygon_overlays_visible_before_holds: bool | None = None
        self._gradient_overlay_visible_before_holds: bool | None = None
        self._filter_preview_hold_active = False
        self._last_pointer_viewport_pos: QPointF | None = None
        self._image_click_mode = False
        self._image_region_selection_mode = False
        self._ctrl_image_region_selection_enabled = False
        self._via_debug_inspection_enabled = False
        self._clipboard_polygons: list[PolygonData] = []
        self._clipboard_anchor = QPointF(0.0, 0.0)
        self._paste_mode = False
        self._paste_preview_items: list[QGraphicsPathItem] = []
        self._vector_geometry_settings = VectorGeometrySettings()
        self._drag_polygons_snapshot: list[PolygonData] | None = None
        self._drag_polygon_is_contact = False
        self._contact_drag_profile: ContactDragProfile | None = None
        self._move_vertex_tool_profile: MoveVertexToolActivationProfile | None = None
        self._move_vertex_tool_profile_generation = 0
        self._move_vertex_tool_profile_paint_started_at: float | None = None
        self._brush_pan_guard = False
        self._frame_navigation_guard: Callable[[], bool] | None = None
        self._pending_wheel_zoom_factor = 1.0
        self._pending_wheel_zoom_viewport_pixel: QPoint | None = None
        self._wheel_zoom_timer = QTimer(self)
        self._wheel_zoom_timer.setSingleShot(True)
        self._wheel_zoom_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._wheel_zoom_timer.setInterval(_WHEEL_ZOOM_COALESCE_MS)
        self._wheel_zoom_timer.timeout.connect(self._flush_queued_wheel_zoom)
        self._zoom_animation_timer = QTimer(self)
        self._zoom_animation_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._zoom_animation_timer.setInterval(_ZOOM_ANIMATION_FRAME_MS)
        self._zoom_animation_timer.timeout.connect(self._advance_zoom_animation)
        self._zoom_animation_viewport_pixel: QPoint | None = None
        self._zoom_animation_target_zoom = 1.0
        self._scene_zoom_profile: SceneZoomProfile | None = None
        self._pyramid_store: PyramidFrameStore | None = None
        self._pyramid_layout: FixedGridFrameLayout | None = None
        self._pyramid_enabled = False
        self._pyramid_current_frame_id: int | None = None
        self._pyramid_current_lod = 0
        self._pyramid_visible_items: dict[int, QGraphicsPixmapItem] = {}
        self._pyramid_pixmap_cache: OrderedDict[tuple[int, int], QPixmap] = OrderedDict()
        self._pyramid_pending_loads: set[tuple[int, int]] = set()
        self._pyramid_generation = 0
        self._pyramid_thread_pool = QThreadPool(self)
        self._pyramid_thread_pool.setMaxThreadCount(2)
        self._pyramid_thread_pool.setExpiryTimeout(30000)
        self._pyramid_visible_timer = QTimer(self)
        self._pyramid_visible_timer.setSingleShot(True)
        self._pyramid_visible_timer.setInterval(_PYRAMID_VISIBLE_UPDATE_MS)
        self._pyramid_visible_timer.timeout.connect(self._refresh_pyramid_visible_frames)
        self._gradient_field_x = None
        self._gradient_field_y = None
        self._gradient_field_peak = 0.0
        self._gradient_arrows_timer = QTimer(self)
        self._gradient_arrows_timer.setSingleShot(True)
        self._gradient_arrows_timer.setInterval(_PYRAMID_VISIBLE_UPDATE_MS)
        self._gradient_arrows_timer.timeout.connect(self._refresh_gradient_field_arrows)
        self.zoomChanged.connect(self._schedule_gradient_field_arrows_update)
        self._pyramid_selection_item = QGraphicsRectItem()
        selection_pen = QPen(QColor("#22D3EE"), 2.0)
        selection_pen.setCosmetic(True)
        self._pyramid_selection_item.setPen(selection_pen)
        self._pyramid_selection_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._pyramid_selection_item.setZValue(-17)
        self._editor_scene.addItem(self._pyramid_selection_item)
        self._pyramid_selection_item.hide()

        self._editor_scene.polygonsChanged.connect(self.polygonsEdited.emit)
        self._editor_scene.polygonsChanged.connect(self._refresh_available_tools_from_scene)
        self._editor_scene.activePolygonChanged.connect(self.activePolygonChanged.emit)
        self._editor_scene.logRequested.connect(self.logRequested.emit)

        for action, default, slot in (
            ("undo", "Ctrl+Z", self.undo),
            ("redo", "Ctrl+Y", self.redo),
            ("copy", "Ctrl+C", self.copy_selected),
            ("cut", "Ctrl+X", self.cut_selected),
            ("paste", "Ctrl+V", self.start_paste_mode),
        ):
            sequence = shortcut_sequence(action, default)
            if sequence is None:
                continue
            shortcut = QShortcut(sequence, self)
            shortcut.activated.connect(slot)

        for tool in EditorTool:
            sequence = tool_shortcut_sequence(tool)
            if sequence is None:
                continue
            shortcut = QShortcut(sequence, self)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(lambda t=tool: self._activate_tool_shortcut(t))

        self._minimap = MinimapWidget(self)
        self._minimap.scenePointRequested.connect(self._on_minimap_scene_point)
        self._minimap.sceneDeltaRequested.connect(self._on_minimap_scene_delta)
        self._minimap.raise_()
        self.horizontalScrollBar().valueChanged.connect(self._update_minimap_overlay)
        self.verticalScrollBar().valueChanged.connect(self._update_minimap_overlay)
