from __future__ import annotations

import cProfile
import io
import pstats
from collections import OrderedDict
from collections.abc import Callable
from math import hypot, log2
from time import perf_counter

from PyQt6.QtCore import (
    QEvent,
    QPoint,
    QPointF,
    QRectF,
    Qt,
    QThreadPool,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush,
    QCloseEvent,
    QColor,
    QContextMenuEvent,
    QGuiApplication,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QResizeEvent,
    QShortcut,
    QTabletEvent,
    QTransform,
    QUndoStack,
    QWheelEvent,
)
from PyQt6.QtWidgets import QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsView, QMenu, QWidget

from ..adapters.qt.pyramid import PyramidFrameLoadRunnable
from ..application.frame_lod import FixedGridFrameLayout, PyramidFrameStore
from ..application.processing import DisplaySettings
from ..application.vector_geometry_postprocess import (
    VectorGeometrySettings,
    apply_edge_translation_to_clone,
    apply_polygon_points_to_clone,
    collapse_redundant_vertices_in_polygons,
    merge_overlapping_root_families_near_polygons,
    postprocess_changed_polygon_edit,
    postprocess_vertex_move_edit,
    resolve_focus_id_after_geometry_pass,
)
from ..commands import MovePolygonCommand
from ..domain import PolygonData
from ..domain.polygon_ring import is_valid_closed_polygon_edge_move, is_valid_closed_polygon_vertex_move
from ..infrastructure.contact_placement_profiler import (
    ContactDragProfile,
    MoveVertexToolActivationProfile,
    SceneZoomProfile,
)
from ..infrastructure.profiling import (
    contact_drag_profiling_enabled,
    delete_area_profiling_enabled,
    delete_area_top_lines,
    move_vertex_tool_profiling_enabled,
    scene_zoom_profiling_enabled,
    try_disable_profiler,
    try_enable_profiler,
    vertex_move_profiling_enabled,
    vertex_move_top_lines,
)
from .editor_hotkeys import tool_shortcut_sequence
from .editor_scene import PolygonEditorScene
from .geometry import (
    _points_different,
    _polygon_points_different,
    _polygons_center,
    _snap_to_45,
    is_valid_closed_polygon_ring,
)
from .minimap_geometry import MINIMAP_VIEWPORT_MARGIN_PX
from .minimap_widget import MinimapWidget
from .tool_mode_logic import (
    apply_conductor_recognition_tool_lock,
    available_editor_tools,
    effective_polygon_create_mode,
    is_via_polygon,
    normalize_editor_tool,
)
from .tools import MIN_MANUAL_STROKE_WIDTH_PX, BrushMode, DeleteVertexMode, EditorTool, PolygonCreateMode
from .viewport_navigation import (
    DEFAULT_ZOOM_STEP_FACTOR,
    clamp_zoom_factor,
    viewport_scroll_correction_after_scale_reanchor,
    zoom_factor_for_wheel_delta,
)

_WHEEL_ZOOM_COALESCE_MS = 3
_ZOOM_ANIMATION_FRAME_MS = 16
_ZOOM_EASING_FRACTION = 0.55
_ZOOM_SETTLE_RATIO = 0.001
_ZOOM_VECTOR_BATCH_THRESHOLD = 1000
_MINIMAP_DRAWING_TOOLS = frozenset(
    {
        EditorTool.ADD_POLYGON,
        EditorTool.BRUSH,
        EditorTool.TRACE_PEN,
    }
)
_OPENGL_VIEWPORT_ENABLED = True
_OPENGL_DISABLED_PLATFORMS = {"offscreen", "minimal"}
_PYRAMID_VISIBLE_UPDATE_MS = 24
_PYRAMID_CACHE_LIMIT = 192


class PolygonEditorView(QGraphicsView):
    polygonsEdited = pyqtSignal()
    activePolygonChanged = pyqtSignal(object)
    logRequested = pyqtSignal(str)
    imageClicked = pyqtSignal(float, float)
    imageRegionSelected = pyqtSignal(float, float, float, float)
    rulerMeasurementChanged = pyqtSignal(str)
    toolChanged = pyqtSignal(object)
    effectivePolygonCreateModeChanged = pyqtSignal(object)
    polygonCreateModeChanged = pyqtSignal(object)
    brushModeChanged = pyqtSignal(object)
    traceModeChanged = pyqtSignal(object)
    deleteVertexModeChanged = pyqtSignal(object)
    zoomChanged = pyqtSignal(float)
    neighborFrameActivated = pyqtSignal(str)
    viaDebugRequested = pyqtSignal(object)
    manualViaAdded = pyqtSignal(float, float)
    contactPlacementHotkeyPressed = pyqtSignal()
    contactPlacementAttemptStarted = pyqtSignal()
    contactPlacementAttemptFinished = pyqtSignal(bool)
    contactMultiSelectionStarted = pyqtSignal()
    contactMultiSelectionApplyStarted = pyqtSignal()
    contactMultiSelectionFinished = pyqtSignal(int)
    contactDeletionStarted = pyqtSignal(int)
    contactDeletionFinished = pyqtSignal(int)
    contactCopyStarted = pyqtSignal()
    contactCopyFinished = pyqtSignal(int)
    contactPasteStarted = pyqtSignal(int)
    contactPasteFinished = pyqtSignal(int)
    contactUndoStarted = pyqtSignal()
    contactUndoFinished = pyqtSignal(bool, int)
    contactRedoStarted = pyqtSignal()
    contactRedoFinished = pyqtSignal(bool, int)
    recognizedViasDeleted = pyqtSignal(object)
    metalOverlayDetailRequested = pyqtSignal(str, str)
    filterPreviewHoldChanged = pyqtSignal(bool)
    frameNavigationRequested = pyqtSignal(object)
    currentFrameChanged = pyqtSignal(object)
    editorViewportChanged = pyqtSignal(object)
    availableToolsChanged = pyqtSignal(object)

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

        for sequence, slot in (
            (QKeySequence.StandardKey.Undo, self.undo),
            (QKeySequence.StandardKey.Redo, self.redo),
            (QKeySequence.StandardKey.Copy, self.copy_selected),
            (QKeySequence.StandardKey.Cut, self.cut_selected),
            (QKeySequence.StandardKey.Paste, self.start_paste_mode),
        ):
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

    def _activate_tool_shortcut(self, tool: EditorTool) -> None:
        if tool not in self._available_tools:
            return
        if tool == EditorTool.ADD_VIA:
            self.contactPlacementHotkeyPressed.emit()
        self.set_tool(tool)

    def _configure_opengl_viewport(self) -> bool:
        if not _OPENGL_VIEWPORT_ENABLED:
            return False
        app = QGuiApplication.instance()
        platform = str(app.platformName()).lower() if app is not None else ""
        if platform in _OPENGL_DISABLED_PLATFORMS:
            return False
        try:
            from PyQt6.QtOpenGLWidgets import QOpenGLWidget
        except Exception:
            return False
        try:
            viewport = QOpenGLWidget(self)
            viewport.setUpdateBehavior(QOpenGLWidget.UpdateBehavior.PartialUpdate)
            self.setViewport(viewport)
        except Exception:
            return False
        return True

    def _require_viewport(self) -> QWidget:
        viewport = self.viewport()
        if viewport is None:
            raise RuntimeError("Graphics view has no viewport")
        return viewport

    @property
    def undo_stack(self) -> QUndoStack:
        return self._editor_scene.undo_stack

    @property
    def current_tool(self) -> EditorTool:
        return self._tool

    def set_frame_navigation_guard(self, guard: Callable[[], bool] | None) -> None:
        self._frame_navigation_guard = guard

    def _confirm_frame_navigation(self) -> bool:
        if self._frame_navigation_guard is None:
            return True
        return bool(self._frame_navigation_guard())

    def set_tool(self, tool: EditorTool) -> None:
        tool = normalize_editor_tool(tool)
        if tool not in self._available_tools:
            if self._tool in self._available_tools:
                return
            tool = EditorTool.SELECT
        profile_activation = (
            tool == EditorTool.MOVE_VERTEX
            and move_vertex_tool_profiling_enabled()
        )
        profile_started_at: float | None = None
        if profile_activation:
            self._cancel_move_vertex_tool_profile_finish()
            if self._move_vertex_tool_profile is not None:
                self._finish_move_vertex_tool_profile("superseded")
            profile_started_at = perf_counter()
            polygons = self._editor_scene.get_polygons()
            self._move_vertex_tool_profile = MoveVertexToolActivationProfile.begin(
                polygon_count=len(polygons),
                vertex_count=sum(len(polygon.points) for polygon in polygons),
            )
            print("[contour move-vertex-tool profiling] started", flush=True)
        elif self._move_vertex_tool_profile is not None:
            self._cancel_move_vertex_tool_profile_finish()
            self._finish_move_vertex_tool_profile("superseded")
        self._editor_scene.cancel_polygon_edit_preview()
        self._drag_preview_points = None
        self._tool = tool
        self._select_press_polygon_id = None
        self._select_press_start = None
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag if tool == EditorTool.PAN else QGraphicsView.DragMode.NoDrag
        )
        if tool == EditorTool.ADD_POLYGON:
            self._editor_scene.set_pending_path_width(1.5, cosmetic=True)
        elif tool == EditorTool.BRUSH:
            self._editor_scene.set_pending_path_width(self._brush_thickness, cosmetic=False)
        elif tool == EditorTool.TRACE_PEN:
            self._editor_scene.set_pending_path_width(self._trace_width, cosmetic=False)
        if tool not in (EditorTool.ADD_POLYGON, EditorTool.BRUSH, EditorTool.TRACE_PEN):
            self._editor_scene.cancel_pending_polygon()
            self._pending_polygon_erases = None
        if tool != EditorTool.ADD_POLYGON:
            self._cancel_rectangle_polygon()
        if tool != EditorTool.DELETE_VERTEX:
            self._editor_scene.clear_preview_rect()
        if tool != EditorTool.RULER:
            self._editor_scene.clear_measurement()
            self.rulerMeasurementChanged.emit("")
        if tool != EditorTool.ANTIALIAS:
            self._editor_scene.clear_vertex_preview()
        sync_phase_started_at = perf_counter()
        if profile_started_at is not None and self._move_vertex_tool_profile is not None:
            self._move_vertex_tool_profile.note_timing(
                "tool_setup",
                (sync_phase_started_at - profile_started_at) * 1000.0,
            )
        sync_refresh_ms = self._sync_all_editable_vertices_display()
        ui_phase_started_at = perf_counter()
        if profile_started_at is not None and self._move_vertex_tool_profile is not None:
            self._move_vertex_tool_profile.note_timing(
                "sync_vertices",
                sync_refresh_ms
                if sync_refresh_ms > 0.0
                else (ui_phase_started_at - sync_phase_started_at) * 1000.0,
            )
        self._update_tool_cursors()
        self._update_minimap_mouse_interaction()
        self.toolChanged.emit(tool)
        self._emit_effective_polygon_create_mode_changed()
        if profile_started_at is not None and self._move_vertex_tool_profile is not None:
            self._move_vertex_tool_profile.note_timing(
                "ui_finish",
                (perf_counter() - ui_phase_started_at) * 1000.0,
            )
            self._schedule_move_vertex_tool_profile_finish()

    def available_tools(self) -> frozenset[EditorTool]:
        return self._available_tools

    def _refresh_available_tools_from_scene(self) -> None:
        available = apply_conductor_recognition_tool_lock(
            self._editor_scene.available_editor_tools(),
            enabled=self._conductor_recognition_mode,
        )
        changed = available != self._available_tools
        self._available_tools = available
        if self._tool not in available:
            self.set_tool(EditorTool.SELECT)
        if self._paste_mode and (
            self._conductor_recognition_mode
            or not self._editor_scene.can_add_polygon_set(self._clipboard_polygons)
        ):
            self._exit_paste_mode()
        if changed:
            self.availableToolsChanged.emit(available)

    def vector_edits_locked(self) -> bool:
        return bool(self._conductor_recognition_mode)

    def set_conductor_recognition_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._conductor_recognition_mode:
            self._refresh_available_tools_from_scene()
            self.availableToolsChanged.emit(self._available_tools)
            return
        self._conductor_recognition_mode = enabled
        if enabled:
            self._exit_paste_mode()
            self._editor_scene.cancel_pending_polygon()
            self._editor_scene.clear_preview_rect()
            self._select_press_polygon_id = None
            self._select_press_start = None
            self._drag_kind = None
        self._refresh_available_tools_from_scene()
        self.availableToolsChanged.emit(self._available_tools)

    def set_contact_recognition_mode(self, enabled: bool) -> None:
        self._contact_recognition_mode = bool(enabled)
        # Recognized contacts remain selectable on left click, but right click /
        # Delete must be able to turn one into negative heuristic feedback.
        self._editor_scene.set_protect_recognized_vias(False)

    def set_polygon_create_mode(self, mode: PolygonCreateMode) -> None:
        mode = PolygonCreateMode(mode)
        changed = mode != self._polygon_create_mode
        self._polygon_create_mode = mode
        self._cancel_rectangle_polygon()
        self._editor_scene.cancel_pending_polygon()
        self._pending_polygon_erases = None
        if changed:
            self.polygonCreateModeChanged.emit(mode)
        self._emit_effective_polygon_create_mode_changed()

    def set_brush_mode(self, mode: BrushMode) -> None:
        mode = BrushMode(mode)
        changed = mode != self._brush_mode
        self._brush_mode = mode
        self._editor_scene.cancel_pending_polygon()
        if changed:
            self.brushModeChanged.emit(mode)
        self._update_tool_cursors()

    def set_brush_thickness(self, thickness: float) -> None:
        self._brush_thickness = max(MIN_MANUAL_STROKE_WIDTH_PX, float(thickness))
        if self._tool == EditorTool.BRUSH:
            self._editor_scene.set_pending_path_width(self._brush_thickness, cosmetic=False)
        self._update_tool_cursors()

    def set_trace_width(self, width: float) -> None:
        self._trace_width = max(MIN_MANUAL_STROKE_WIDTH_PX, float(width))
        if self._tool == EditorTool.TRACE_PEN:
            self._editor_scene.set_pending_path_width(self._trace_width, cosmetic=False)
        self._update_tool_cursors()

    def set_trace_mode(self, mode: BrushMode) -> None:
        mode = BrushMode(mode)
        changed = mode != self._trace_mode
        self._trace_mode = mode
        if changed:
            self.traceModeChanged.emit(mode)
        if self._editor_scene.has_pending_polygon() and self._tool == EditorTool.TRACE_PEN:
            self._editor_scene.cancel_pending_polygon()
            self._pending_polygon_erases = None

    def set_via_size(self, width: float, height: float) -> None:
        self._via_width = max(1.0, float(width))
        self._via_height = max(1.0, float(height))

    def set_minimum_contact_distance(self, distance: float) -> None:
        self._editor_scene.set_minimum_contact_distance(distance)
        self._update_tool_cursors()

    def set_vector_geometry_settings(self, settings: VectorGeometrySettings | None) -> None:
        self._vector_geometry_settings = settings if settings is not None else VectorGeometrySettings()
        self._editor_scene.set_vector_geometry_settings(settings)

    def set_delete_vertex_mode(self, mode: DeleteVertexMode) -> None:
        mode = DeleteVertexMode(mode)
        changed = mode != self._delete_vertex_mode
        self._delete_vertex_mode = mode
        self._editor_scene.clear_preview_rect()
        self._sync_all_editable_vertices_display()
        if changed:
            self.deleteVertexModeChanged.emit(mode)

    def _should_show_all_editable_vertices(self) -> bool:
        if self._tool in {EditorTool.ANTIALIAS, EditorTool.MOVE_VERTEX}:
            return True
        return self._tool == EditorTool.DELETE_VERTEX and self._delete_vertex_mode == DeleteVertexMode.SINGLE

    def _sync_all_editable_vertices_display(self) -> float:
        return self._editor_scene.set_show_all_editable_vertices(
            self._should_show_all_editable_vertices()
        )

    def _effective_polygon_create_mode(self) -> PolygonCreateMode:
        return effective_polygon_create_mode(
            tool=self._tool,
            base=self._polygon_create_mode,
            shift_held=False,
            has_pending_polygon=self._editor_scene.has_pending_polygon(),
        )

    def effective_polygon_create_mode(self) -> PolygonCreateMode:
        """Polygon draw mode including Shift override (read-only, for UI/tests)."""
        return self._effective_polygon_create_mode()

    def _emit_effective_polygon_create_mode_changed(self) -> None:
        self.effectivePolygonCreateModeChanged.emit(self._effective_polygon_create_mode())

    def _viewport_view_anchor(self) -> tuple[QPoint, QPointF, QTransform]:
        viewport = self._require_viewport()
        pixel = viewport.rect().center()
        return pixel, self.mapToScene(pixel), QTransform(self.transform())

    def _restore_viewport_view_anchor(
        self,
        pixel: QPoint,
        scene_anchor: QPointF,
        transform: QTransform,
    ) -> None:
        self.setTransform(transform)
        mapped = self.mapFromScene(scene_anchor)
        dx = int(round(mapped.x() - pixel.x()))
        dy = int(round(mapped.y() - pixel.y()))
        if dx or dy:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + dx)
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + dy)

    def set_image(self, image, *, preserve_view: bool = False) -> None:
        previous_rect = QRectF(self._editor_scene.sceneRect())
        view_anchor = self._viewport_view_anchor() if preserve_view else None
        self._editor_scene.set_image(image)
        previous_was_placeholder = previous_rect.width() <= 1.0 and previous_rect.height() <= 1.0
        if previous_was_placeholder:
            self.fit_to_view()
        elif view_anchor is not None:
            self._restore_viewport_view_anchor(*view_anchor)
        self._refresh_minimap_thumbnail()
        self._update_navigation_scene_rect()
        if view_anchor is not None:
            self._restore_viewport_view_anchor(*view_anchor)
    def set_image_pixmap(self, pixmap: QPixmap, *, preserve_view: bool = False) -> None:
        previous_rect = QRectF(self._editor_scene.sceneRect())
        view_anchor = self._viewport_view_anchor() if preserve_view else None
        self._editor_scene.set_image_pixmap(pixmap)
        previous_was_placeholder = previous_rect.width() <= 1.0 and previous_rect.height() <= 1.0
        if previous_was_placeholder:
            self.fit_to_view()
        elif view_anchor is not None:
            self._restore_viewport_view_anchor(*view_anchor)
        self._refresh_minimap_thumbnail()
        self._update_navigation_scene_rect()
        if view_anchor is not None:
            self._restore_viewport_view_anchor(*view_anchor)
    def set_polygons(
        self,
        polygons: list[PolygonData],
        *,
        emit_signal: bool = True,
        repair_reasons: dict[int, list[str]] | None = None,
        scan_repair: bool = True,
        defer_hit_paths: bool = False,
        defer_object_colors: bool = False,
    ) -> None:
        self._editor_scene.set_polygons(
            polygons,
            emit_signal=emit_signal,
            repair_reasons=repair_reasons,
            scan_repair=scan_repair,
            defer_hit_paths=defer_hit_paths,
            defer_object_colors=defer_object_colors,
        )
        self._refresh_available_tools_from_scene()

    def apply_polygons_needing_repair(
        self,
        reasons: dict[int, list[str]] | None,
        *,
        refresh_items: bool = True,
    ) -> None:
        self._editor_scene.apply_polygons_needing_repair(reasons, refresh_items=refresh_items)

    def polygons_needing_repair_map(self) -> dict[int, list[str]]:
        return self._editor_scene.polygons_needing_repair_map()

    def refresh_polygon_overlays(self) -> None:
        self._editor_scene.refresh_polygon_items()

    def get_polygons(self) -> list[PolygonData]:
        return self._editor_scene.get_polygons()

    def antialias_selected_polygons(self, grade: int) -> bool:
        return self._editor_scene.antialias_selected_polygons(grade)

    def repair_invalid_polygon_descriptions(self) -> bool:
        return self._editor_scene.repair_invalid_polygon_descriptions()

    def set_antialias_grade(self, grade: int) -> None:
        self._antialias_grade = max(1, int(grade))

    def set_neighbor_frames(
        self,
        frames: list[tuple],
        opacity: float,
        overlap_pixels: int = 0,
        show_main_frame: bool = True,
    ) -> None:
        self._editor_scene.set_neighbor_frames(frames, opacity, overlap_pixels, show_main_frame)
        self._refresh_neighbor_viewport()

    def _refresh_neighbor_viewport(self) -> None:
        """Expand scroll range for the neighbor grid without changing pan/zoom."""
        self._update_navigation_scene_rect()
        self._editor_scene.update(self._editor_scene.sceneRect())
        viewport = self.viewport()
        if viewport is not None:
            viewport.update()

    def set_debug_candidates(self, candidates: list[object]) -> None:
        self._editor_scene.set_debug_candidates(candidates)

    def set_metal_overlays(self, layers: dict[str, list[PolygonData]], visibility: dict[str, bool]) -> None:
        self._editor_scene.set_metal_overlays(layers, visibility)

    def set_via_debug_inspection_enabled(self, enabled: bool) -> None:
        self._via_debug_inspection_enabled = bool(enabled)
        self._editor_scene.set_debug_candidates([])

    def zoom_factor(self) -> float:
        return max(1e-6, float(self.transform().m11()))

    def set_pyramid_frame_store(
        self,
        store: PyramidFrameStore | None,
        *,
        frame_count: int | None = None,
        columns: int | None = None,
        current_frame_id: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Enable a virtualized multi-frame pyramid display when a store is available."""

        self._clear_pyramid_items()
        self._pyramid_generation += 1
        self._pyramid_store = store
        self._pyramid_layout = None
        self._pyramid_pixmap_cache.clear()
        self._pyramid_pending_loads.clear()
        if store is None:
            self._pyramid_enabled = False
            self._pyramid_current_frame_id = None
            self._pyramid_selection_item.hide()
            self._update_navigation_scene_rect()
            return
        count = max(0, int(frame_count if frame_count is not None else store.frame_count()))
        should_enable = bool(store.has_lod()) if enabled is None else bool(enabled and store.has_lod())
        if count <= 0 or not should_enable:
            self._pyramid_enabled = False
            self._pyramid_current_frame_id = current_frame_id
            self._pyramid_selection_item.hide()
            self._update_navigation_scene_rect()
            return
        if columns is None:
            columns = max(1, int(round(count ** 0.5)))
        self._pyramid_layout = FixedGridFrameLayout(
            frame_count=count,
            columns=max(1, int(columns)),
            frame_store=store,
            gap=16,
        )
        self._pyramid_enabled = True
        self._pyramid_current_lod = self.choose_lod(self.zoom_factor(), store.max_lod())
        self.set_current_frame_id(0 if current_frame_id is None else current_frame_id, center=False, emit_signal=False)
        self._update_navigation_scene_rect()
        self._schedule_pyramid_visible_update()

    def set_pyramid_frames(
        self,
        store: PyramidFrameStore | None,
        *,
        frame_count: int | None = None,
        columns: int | None = None,
        current_frame_id: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.set_pyramid_frame_store(
            store,
            frame_count=frame_count,
            columns=columns,
            current_frame_id=current_frame_id,
            enabled=enabled,
        )

    def pyramid_mode_enabled(self) -> bool:
        return bool(self._pyramid_enabled and self._pyramid_store is not None and self._pyramid_layout is not None)

    def choose_lod(self, zoom: float, max_lod: int) -> int:
        max_lod = max(0, int(max_lod))
        zoom = max(1e-6, float(zoom))
        target = max(0, min(max_lod, int(round(log2(1.0 / zoom)))))
        current = max(0, min(max_lod, int(getattr(self, "_pyramid_current_lod", 0))))
        if target == current:
            return current
        # Hysteresis keeps the pyramid from swapping LODs repeatedly near 2x boundaries.
        if target > current:
            switch_zoom = (2.0 ** (-(current + 0.65)))
            return target if zoom < switch_zoom else current
        switch_zoom = (2.0 ** (-(current - 0.35)))
        return target if zoom > switch_zoom else current

    def current_frame_id(self) -> int | None:
        return self._pyramid_current_frame_id

    def set_current_frame_id(self, frame_id: int | None, *, center: bool = True, emit_signal: bool = True) -> None:
        layout = self._pyramid_layout
        if frame_id is None:
            self._pyramid_current_frame_id = None
            self._pyramid_selection_item.hide()
            return
        frame_id = int(frame_id)
        if layout is not None:
            frame_id = max(0, min(layout.frame_count - 1, frame_id))
        changed = frame_id != self._pyramid_current_frame_id
        self._pyramid_current_frame_id = frame_id
        self._update_pyramid_selection_rect()
        if center:
            self.center_on_frame(frame_id)
        if changed and emit_signal:
            self.currentFrameChanged.emit(frame_id)

    def center_on_frame(self, frame_id: int | None) -> None:
        if frame_id is None:
            return
        layout = self._pyramid_layout
        if layout is None:
            self.center_main_image()
            return
        rect = layout.frame_id_to_scene_rect(int(frame_id), self._pyramid_current_lod)
        if rect.width() > 0 and rect.height() > 0:
            self.centerOn(rect.center())
            self._update_navigation_scene_rect()
            self._schedule_pyramid_visible_update()

    def _clear_pyramid_items(self) -> None:
        self._pyramid_visible_timer.stop()
        for item in self._pyramid_visible_items.values():
            item.setPixmap(QPixmap())
            if item.scene() is not None:
                self._editor_scene.removeItem(item)
        self._pyramid_visible_items.clear()
        self._pyramid_pixmap_cache.clear()

    def _pyramid_viewport_scene_rect(self) -> QRectF:
        viewport = self._require_viewport().rect()
        polygon = self.mapToScene(viewport)
        return polygon.boundingRect()

    def _clear_gradient_field_maps(self) -> None:
        self._gradient_arrows_timer.stop()
        self._gradient_field_x = None
        self._gradient_field_y = None
        self._gradient_field_peak = 0.0
        self._editor_scene.clear_gradient_field_arrows()

    def _schedule_gradient_field_arrows_update(self, *_args) -> None:
        if self._gradient_field_x is None or self._gradient_field_y is None:
            return
        self._gradient_arrows_timer.stop()
        self._gradient_arrows_timer.start()

    def _refresh_gradient_field_arrows(self) -> None:
        gradient_x = self._gradient_field_x
        gradient_y = self._gradient_field_y
        if gradient_x is None or gradient_y is None:
            self._editor_scene.clear_gradient_field_arrows()
            return
        from .gradient_field_arrows import sample_gradient_field_arrows

        visible = self._pyramid_viewport_scene_rect()
        zoom = self.zoom_factor()
        scene_units_per_view_px = 1.0 / max(zoom, 1e-12)
        arrows = sample_gradient_field_arrows(
            gradient_x,
            gradient_y,
            (visible.left(), visible.top(), visible.right(), visible.bottom()),
            scene_units_per_view_px,
            peak_magnitude=self._gradient_field_peak,
        )
        self._editor_scene.set_gradient_field_arrows(arrows)

    def _schedule_pyramid_visible_update(self) -> None:
        if not self.pyramid_mode_enabled():
            return
        self._pyramid_visible_timer.stop()
        self._pyramid_visible_timer.start()

    def _refresh_pyramid_visible_frames(self) -> None:
        store = self._pyramid_store
        layout = self._pyramid_layout
        if store is None or layout is None or not self._pyramid_enabled:
            return
        new_lod = self.choose_lod(self.zoom_factor(), store.max_lod())
        if new_lod != self._pyramid_current_lod:
            self._pyramid_current_lod = new_lod
            self._clear_pyramid_items()
            self._update_navigation_scene_rect()
            self._update_pyramid_selection_rect()
        viewport_rect = self._pyramid_viewport_scene_rect()
        if store.max_lod() <= 0 and self._pyramid_current_lod == 0:
            visible = set()
        else:
            visible = set(layout.frame_ids_intersecting(viewport_rect, self._pyramid_current_lod, buffer_cells=0))
        if self._pyramid_current_frame_id is not None:
            visible.add(int(self._pyramid_current_frame_id))
        for frame_id in list(self._pyramid_visible_items):
            if frame_id in visible:
                continue
            item = self._pyramid_visible_items.pop(frame_id)
            item.setPixmap(QPixmap())
            if item.scene() is not None:
                self._editor_scene.removeItem(item)
            for cache_key in list(self._pyramid_pixmap_cache):
                if cache_key[0] == frame_id:
                    self._pyramid_pixmap_cache.pop(cache_key, None)
        self._prune_pyramid_pixmap_cache(visible)
        for frame_id in sorted(visible):
            self._ensure_pyramid_frame_item(frame_id)
        self._update_pyramid_selection_rect()
        self.editorViewportChanged.emit(viewport_rect)

    def _ensure_pyramid_frame_item(self, frame_id: int) -> None:
        layout = self._pyramid_layout
        if layout is None:
            return
        key = (int(frame_id), int(self._pyramid_current_lod))
        item = self._pyramid_visible_items.get(frame_id)
        if item is None:
            item = QGraphicsPixmapItem()
            item.setZValue(-30)
            item.setTransformationMode(Qt.TransformationMode.FastTransformation)
            self._editor_scene.addItem(item)
            self._pyramid_visible_items[frame_id] = item
        rect = layout.frame_id_to_scene_rect(frame_id, self._pyramid_current_lod)
        item.setPos(rect.topLeft())
        pixmap = self._pyramid_cached_pixmap(key)
        if pixmap is not None and not pixmap.isNull():
            item.setPixmap(pixmap)
            item.setScale(rect.width() / max(1, pixmap.width()))
            item.show()
            return
        item.hide()
        self._queue_pyramid_frame_load(frame_id, self._pyramid_current_lod)

    def _pyramid_cached_pixmap(self, key: tuple[int, int]) -> QPixmap | None:
        pixmap = self._pyramid_pixmap_cache.get(key)
        if pixmap is not None:
            self._pyramid_pixmap_cache.move_to_end(key)
        return pixmap

    def _cache_pyramid_pixmap(self, key: tuple[int, int], pixmap: QPixmap) -> None:
        if key[0] not in self._pyramid_visible_items:
            return
        self._pyramid_pixmap_cache[key] = pixmap
        self._pyramid_pixmap_cache.move_to_end(key)
        self._prune_pyramid_pixmap_cache(set(self._pyramid_visible_items))

    def _prune_pyramid_pixmap_cache(self, visible_frame_ids: set[int]) -> None:
        visible_keys = {(int(frame_id), int(self._pyramid_current_lod)) for frame_id in visible_frame_ids}
        for key in list(self._pyramid_pixmap_cache):
            if key not in visible_keys:
                self._pyramid_pixmap_cache.pop(key, None)

    def _queue_pyramid_frame_load(self, frame_id: int, lod: int) -> None:
        store = self._pyramid_store
        if store is None:
            return
        key = (int(frame_id), int(lod))
        if key in self._pyramid_pending_loads:
            return
        self._pyramid_pending_loads.add(key)
        generation = self._pyramid_generation
        runnable = PyramidFrameLoadRunnable(generation, frame_id, lod, store)
        runnable.signals.result.connect(self._on_pyramid_frame_loaded)
        runnable.signals.error.connect(self._on_pyramid_frame_error)
        self._pyramid_thread_pool.start(runnable)

    def _on_pyramid_frame_loaded(self, generation: int, frame_id: int, lod: int, qimage: object) -> None:
        key = (int(frame_id), int(lod))
        self._pyramid_pending_loads.discard(key)
        if int(generation) != int(self._pyramid_generation):
            return
        pixmap = QPixmap.fromImage(qimage) if hasattr(qimage, "isNull") and not qimage.isNull() else QPixmap()
        if pixmap.isNull():
            return
        self._cache_pyramid_pixmap(key, pixmap)
        if lod == self._pyramid_current_lod and frame_id in self._pyramid_visible_items:
            self._ensure_pyramid_frame_item(int(frame_id))

    def _on_pyramid_frame_error(self, generation: int, frame_id: int, lod: int, message: str) -> None:
        self._pyramid_pending_loads.discard((int(frame_id), int(lod)))
        if int(generation) == int(self._pyramid_generation):
            self.logRequested.emit(f"[contour pyramid] frame={frame_id} lod={lod} load failed: {message}")

    def _update_pyramid_selection_rect(self) -> None:
        layout = self._pyramid_layout
        frame_id = self._pyramid_current_frame_id
        if not self._pyramid_enabled or layout is None or frame_id is None:
            self._pyramid_selection_item.hide()
            return
        rect = layout.frame_id_to_scene_rect(int(frame_id), self._pyramid_current_lod)
        self._pyramid_selection_item.setRect(rect.adjusted(-2.0, -2.0, 2.0, 2.0))
        self._pyramid_selection_item.show()

    def _pyramid_frame_at_viewport_pos(self, viewport_pos: QPoint) -> int | None:
        layout = self._pyramid_layout
        if not self.pyramid_mode_enabled() or layout is None:
            return None
        scene_pos = self.mapToScene(self._viewport_to_view_point(viewport_pos))
        return layout.scene_pos_to_frame_id(scene_pos.x(), scene_pos.y(), self._pyramid_current_lod)

    def set_display_settings(self, settings: DisplaySettings) -> None:
        self._editor_scene.set_display_settings(settings)
        self._update_tool_cursors()

    def set_random_object_colors_enabled(self, enabled: bool) -> None:
        self._editor_scene.set_random_object_colors_enabled(enabled)

    def flush_deferred_object_colors(self) -> None:
        self._editor_scene.flush_deferred_object_colors()

    def set_extra_layers(self, layers: list[dict[str, object]]) -> None:
        self._editor_scene.set_extra_layers(layers)

    def set_gradient_overlay(self, image, opacity: float = 0.45) -> None:
        self._clear_gradient_field_maps()
        self._editor_scene.set_gradient_overlay(image, opacity)

    def clear_gradient_overlay(self) -> None:
        self._clear_gradient_field_maps()
        self._editor_scene.clear_gradient_overlay()

    def set_gradient_field_maps(self, gradient_x, gradient_y) -> None:
        self._gradient_field_x = gradient_x
        self._gradient_field_y = gradient_y
        self._gradient_field_peak = 0.0
        if gradient_x is not None and gradient_y is not None:
            from .gradient_field_arrows import peak_gradient_magnitude

            self._gradient_field_peak = peak_gradient_magnitude(gradient_x, gradient_y)
        self._refresh_gradient_field_arrows()

    def set_gradient_overlay_opacity(self, opacity: float) -> None:
        self._editor_scene.set_gradient_overlay_opacity(opacity)

    def set_polygon_category_visible(self, category: str, visible: bool) -> None:
        self._editor_scene.set_polygon_category_visible(category, visible)

    def set_polygon_overlays_visible(self, visible: bool) -> None:
        if self._polygon_overlay_hide_holds:
            self._polygon_overlays_visible_before_holds = bool(visible)
            self._editor_scene.set_polygon_overlays_visible(False)
            return
        self._editor_scene.set_polygon_overlays_visible(visible)

    def polygon_overlays_visible(self) -> bool:
        return self._editor_scene.polygon_overlays_visible()

    def set_ui_language(self, language: str | None) -> None:
        self._editor_scene.set_ui_language(language)

    def set_image_click_mode(self, enabled: bool) -> None:
        self._image_click_mode = bool(enabled)

    def set_image_region_selection_mode(self, enabled: bool) -> None:
        self._image_region_selection_mode = bool(enabled)
        if not enabled:
            self._editor_scene.clear_preview_rect()

    def set_ctrl_image_region_selection_enabled(self, enabled: bool) -> None:
        self._ctrl_image_region_selection_enabled = bool(enabled)

    def fit_to_view(self) -> None:
        rect = self._editor_scene.main_image_rect()
        if rect.width() > 0 and rect.height() > 0:
            self._stop_zoom_animation()
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._clamp_current_zoom_at_viewport_pixel(self._require_viewport().rect().center())
            self._update_navigation_scene_rect()
            self.zoomChanged.emit(self.zoom_factor())

    def _set_polygon_overlay_hide_hold(
        self,
        source: str,
        active: bool,
        *,
        hide_gradient_overlay: bool = True,
    ) -> None:
        if active:
            if source in self._polygon_overlay_hide_holds:
                return
            if not self._polygon_overlay_hide_holds:
                self._polygon_overlays_visible_before_holds = self._editor_scene.polygon_overlays_visible()
            self._polygon_overlay_hide_holds.add(source)
            self._editor_scene.set_polygon_overlays_visible(False)
            if not hide_gradient_overlay:
                return
            if not self._gradient_overlay_hide_holds:
                self._gradient_overlay_visible_before_holds = (
                    self._editor_scene.gradient_overlay_user_visible()
                )
            self._gradient_overlay_hide_holds.add(source)
            self._editor_scene.set_gradient_overlay_visible(False)
            return
        if source not in self._polygon_overlay_hide_holds:
            return
        self._polygon_overlay_hide_holds.discard(source)
        if not self._polygon_overlay_hide_holds:
            if self._polygon_overlays_visible_before_holds is not None:
                self._editor_scene.set_polygon_overlays_visible(self._polygon_overlays_visible_before_holds)
            self._polygon_overlays_visible_before_holds = None
        self._gradient_overlay_hide_holds.discard(source)
        if not self._gradient_overlay_hide_holds:
            if self._gradient_overlay_visible_before_holds is not None:
                self._editor_scene.set_gradient_overlay_visible(
                    self._gradient_overlay_visible_before_holds
                )
            self._gradient_overlay_visible_before_holds = None

    def center_main_image(self) -> None:
        rect = self._editor_scene.main_image_rect()
        if rect.width() > 0 and rect.height() > 0:
            self.centerOn(rect.center())
            self._update_navigation_scene_rect()

    def main_image_visible_fraction(self) -> float:
        rect = self._editor_scene.main_image_rect()
        if rect.width() <= 0.0 or rect.height() <= 0.0:
            return 1.0
        viewport_scene = self.mapToScene(self._require_viewport().rect()).boundingRect()
        intersection = rect.intersected(viewport_scene)
        if intersection.isEmpty():
            return 0.0
        image_area = float(rect.width()) * float(rect.height())
        visible_area = float(intersection.width()) * float(intersection.height())
        return max(0.0, min(1.0, visible_area / image_area))

    def should_auto_reposition_view(self, *, force: bool = False) -> bool:
        if force:
            return True
        return self.main_image_visible_fraction() < 0.5

    def zoom_in(self) -> None:
        self._start_zoom_animation(self._zoom_focus_viewport_pixel(), DEFAULT_ZOOM_STEP_FACTOR)

    def zoom_out(self) -> None:
        self._start_zoom_animation(self._zoom_focus_viewport_pixel(), 1.0 / DEFAULT_ZOOM_STEP_FACTOR)

    def undo(self) -> None:
        if self.vector_edits_locked():
            return
        can_undo = self.undo_stack.canUndo()
        contacts_before = self._editor_scene.contact_count()
        self.contactUndoStarted.emit()
        self.undo_stack.undo()
        contacts_changed = abs(self._editor_scene.contact_count() - contacts_before)
        self.contactUndoFinished.emit(can_undo, contacts_changed)

    def redo(self) -> None:
        if self.vector_edits_locked():
            return
        can_redo = self.undo_stack.canRedo()
        contacts_before = self._editor_scene.contact_count()
        self.contactRedoStarted.emit()
        self.undo_stack.redo()
        contacts_changed = abs(self._editor_scene.contact_count() - contacts_before)
        self.contactRedoFinished.emit(can_redo, contacts_changed)

    def copy_selected(self) -> None:
        self.contactCopyStarted.emit()
        polygons = self._editor_scene.selected_polygons()
        if not polygons:
            self.contactCopyFinished.emit(0)
            return
        self._clipboard_polygons = [polygon.clone() for polygon in polygons]
        self._clipboard_anchor = _polygons_center(self._clipboard_polygons)
        self.contactCopyFinished.emit(len(polygons))
        self.start_paste_mode()

    def cut_selected(self) -> None:
        if self.vector_edits_locked():
            return
        polygons = self._editor_scene.selected_deletable_polygons()
        if polygons:
            contacts = [polygon for polygon in polygons if is_via_polygon(polygon)]
            if contacts:
                self.contactDeletionStarted.emit(len(contacts))
            self._clipboard_polygons = [polygon.clone() for polygon in polygons]
            self._clipboard_anchor = _polygons_center(self._clipboard_polygons)
            deleted = self._editor_scene.delete_polygon()
            if contacts:
                self.contactDeletionFinished.emit(len(contacts) if deleted else 0)

    def start_paste_mode(self) -> None:
        if self.vector_edits_locked():
            self.contactPasteFinished.emit(0)
            return
        if self._paste_mode:
            self._update_paste_preview(
                self._last_pointer_scene_pos or self.mapToScene(self._require_viewport().rect().center())
            )
            return
        self.contactPasteStarted.emit(len(self._clipboard_polygons))
        if not self._clipboard_polygons or not self._editor_scene.can_add_polygon_set(self._clipboard_polygons):
            self.contactPasteFinished.emit(0)
            return
        self._paste_mode = True
        self._rebuild_paste_preview()
        self._update_paste_preview(
            self._last_pointer_scene_pos or self.mapToScene(self._require_viewport().rect().center())
        )

    def _clear_paste_preview(self) -> None:
        for item in self._paste_preview_items:
            if item.scene() is not None:
                self._editor_scene.removeItem(item)
        self._paste_preview_items.clear()

    def _exit_paste_mode(self) -> None:
        self._paste_mode = False
        self._clear_paste_preview()

    def _rebuild_paste_preview(self) -> None:
        self._clear_paste_preview()
        if not self._paste_mode:
            return
        pen = QPen(QColor("#38BDF8"), 1.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        brush = QColor("#38BDF8")
        brush.setAlpha(42)
        for polygon in self._clipboard_polygons:
            path = QPainterPath()
            if polygon.shape_hint == "box" or polygon.category == "via":
                x_coord, y_coord, width, height = polygon.bbox
                path.addEllipse(
                    QRectF(
                        float(x_coord),
                        float(y_coord),
                        float(max(0, width - 1)),
                        float(max(0, height - 1)),
                    )
                )
            else:
                if polygon.points:
                    path.moveTo(polygon.points[0][0], polygon.points[0][1])
                    for x_coord, y_coord in polygon.points[1:]:
                        path.lineTo(x_coord, y_coord)
                    path.closeSubpath()
            item = QGraphicsPathItem(path)
            item.setZValue(40)
            item.setPen(pen)
            item.setBrush(QBrush(brush))
            self._editor_scene.addItem(item)
            self._paste_preview_items.append(item)

    def _update_paste_preview(self, scene_pos: QPointF | None) -> None:
        if not self._paste_mode or scene_pos is None:
            return
        if len(self._paste_preview_items) != len(self._clipboard_polygons):
            self._rebuild_paste_preview()
        dx = scene_pos.x() - self._clipboard_anchor.x()
        dy = scene_pos.y() - self._clipboard_anchor.y()
        for item in self._paste_preview_items:
            item.setPos(dx, dy)

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is None:
            return
        viewport_point = self._wheel_event_viewport_pixel(event)
        delta = event.angleDelta()
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            if delta.y() == 0:
                event.accept()
                return
            self._queue_wheel_zoom(viewport_point, zoom_factor_for_wheel_delta(delta.y()))
            event.accept()
            return
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            delta_value = delta.x() if delta.x() else delta.y()
            scrollbar = self.horizontalScrollBar()
            if scrollbar is not None:
                scrollbar.setValue(scrollbar.value() - delta_value)
            event.accept()
            return
        super().wheelEvent(event)
        self._update_tool_cursors()
        event.accept()

    def closeEvent(self, event: QCloseEvent | None) -> None:
        self._wheel_zoom_timer.stop()
        self._pending_wheel_zoom_factor = 1.0
        self._pending_wheel_zoom_viewport_pixel = None
        self._stop_zoom_animation()
        self._gradient_arrows_timer.stop()
        if event is not None:
            super().closeEvent(event)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
            Qt.MouseButton.MiddleButton,
        ):
            self.setFocus(Qt.FocusReason.MouseFocusReason)
        viewport_pixel = self._require_viewport().mapFrom(self, event.position().toPoint())
        self._last_pointer_viewport_pos = QPointF(viewport_pixel)
        scene_pos = self.mapToScene(self._viewport_to_view_point(viewport_pixel))
        self._last_pointer_scene_pos = scene_pos
        tolerance = self._scene_tolerance(8)

        if event.button() == Qt.MouseButton.MiddleButton:
            if self._drag_kind == "brush":
                self._brush_pan_guard = True
            self._middle_pan_active = True
            self._middle_pan_last_viewport = QPointF(viewport_pixel)
            self._set_polygon_overlay_hide_hold(
                "middle",
                True,
                hide_gradient_overlay=False,
            )
            event.accept()
            return

        if self._paste_mode and event.button() == Qt.MouseButton.LeftButton:
            if self.vector_edits_locked():
                self._exit_paste_mode()
                event.accept()
                return
            pasted_ids = self._editor_scene.add_cloned_polygons_at(
                self._clipboard_polygons,
                self._clipboard_anchor,
                scene_pos,
            )
            self._update_paste_preview(scene_pos)
            self.contactPasteFinished.emit(len(pasted_ids))
            event.accept()
            return

        ctrl_region_selection = self._ctrl_image_region_selection_enabled and bool(
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        )
        if (self._image_region_selection_mode or ctrl_region_selection) and event.button() == Qt.MouseButton.LeftButton:
            self._drag_kind = "image_region"
            self._drag_start_scene_pos = scene_pos
            self._editor_scene.set_preview_rect(scene_pos, scene_pos)
            event.accept()
            return

        if self._image_click_mode and event.button() == Qt.MouseButton.LeftButton:
            self.imageClicked.emit(scene_pos.x(), scene_pos.y())
            event.accept()
            return

        if self._tool == EditorTool.PAN:
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            metal_hit = self._editor_scene.metal_overlay_pick(scene_pos)
            if metal_hit is not None:
                self.metalOverlayDetailRequested.emit(metal_hit[0], metal_hit[1])
                event.accept()
                return

        if self._tool == EditorTool.ADD_POLYGON:
            create_mode = self._effective_polygon_create_mode()
            if create_mode == PolygonCreateMode.RECTANGLE and event.button() in (
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.RightButton,
            ):
                if self._drag_kind == "rect_polygon":
                    started_with_right = self._drag_erases
                    clicked_right = event.button() == Qt.MouseButton.RightButton
                    if clicked_right == started_with_right:
                        self._commit_rectangle_polygon(scene_pos)
                    else:
                        self._cancel_rectangle_polygon()
                    event.accept()
                    return
                self._start_rectangle_polygon(
                    scene_pos,
                    erase=event.button() == Qt.MouseButton.RightButton,
                )
                event.accept()
                return
            if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                requested_erase = event.button() == Qt.MouseButton.RightButton
                if self._editor_scene.has_pending_polygon():
                    if self._pending_polygon_erases is None:
                        self._pending_polygon_erases = requested_erase
                    elif requested_erase != self._pending_polygon_erases:
                        self._finish_pending_polygon()
                        event.accept()
                        return
                else:
                    self._pending_polygon_erases = requested_erase
                self._editor_scene.append_pending_point(scene_pos)
                self._emit_effective_polygon_create_mode_changed()
                event.accept()
                return

        if self._tool == EditorTool.BRUSH and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._start_brush_drag(scene_pos, erase=event.button() == Qt.MouseButton.RightButton)
            event.accept()
            return

        if self._tool == EditorTool.TRACE_PEN and event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        ):
            self._append_trace_point(
                scene_pos,
                erase=event.button() == Qt.MouseButton.RightButton,
                snap=bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier),
            )
            event.accept()
            return

        if self._tool == EditorTool.ADD_VIA and event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        ):
            if event.button() == Qt.MouseButton.LeftButton:
                self.contactPlacementAttemptStarted.emit()
            polygon_id = self._editor_scene.polygon_at(scene_pos)
            polygon = self._editor_scene.polygon_snapshot(polygon_id) if polygon_id is not None else None
            if event.button() == Qt.MouseButton.RightButton:
                self.contactDeletionStarted.emit(1)
                deleted = self._editor_scene.delete_via_at(scene_pos)
                if deleted:
                    self._emit_recognized_vias_deleted([polygon] if polygon is not None else [])
                self.contactDeletionFinished.emit(1 if deleted else 0)
                self.contactPlacementAttemptFinished.emit(False)
            elif (
                self._contact_recognition_mode
                and polygon is not None
                and is_via_polygon(polygon)
            ):
                self._editor_scene.select_polygon(polygon.id)
                if self._via_debug_inspection_enabled:
                    self.viaDebugRequested.emit(polygon)
                self.contactPlacementAttemptFinished.emit(False)
            else:
                added = self._editor_scene.add_via_at(scene_pos, self._via_width, self._via_height)
                if added:
                    self.manualViaAdded.emit(float(scene_pos.x()), float(scene_pos.y()))
                self.contactPlacementAttemptFinished.emit(added)
            self._update_tool_cursors()
            event.accept()
            return

            self._update_tool_cursors()
            event.accept()
            return

        if event.button() == Qt.MouseButton.RightButton:
            polygon_id = self._editor_scene.polygon_at(scene_pos)
            polygon = self._editor_scene.polygon_snapshot(polygon_id)
            is_contact = polygon is not None and is_via_polygon(polygon)
            if is_contact:
                self.contactDeletionStarted.emit(1)
            deleted = self._editor_scene.delete_via_at(scene_pos)
            if deleted:
                self._emit_recognized_vias_deleted([polygon] if polygon is not None else [])
                event.accept()
            if is_contact:
                self.contactDeletionFinished.emit(1 if deleted else 0)
                return
            super().mousePressEvent(event)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        if self._tool == EditorTool.RULER:
            self._drag_kind = "ruler"
            self._drag_start_scene_pos = scene_pos
            measurement_text = self._format_ruler_measurement(scene_pos, scene_pos)
            self._editor_scene.set_measurement(scene_pos, scene_pos, measurement_text)
            self.rulerMeasurementChanged.emit(measurement_text)
            event.accept()
            return

        if self._tool == EditorTool.DELETE_POLYGON:
            polygon_id = self._editor_scene.polygon_at(scene_pos)
            polygon = self._editor_scene.polygon_snapshot(polygon_id)
            is_contact = polygon is not None and is_via_polygon(polygon)
            if is_contact:
                self.contactDeletionStarted.emit(1)
            deleted = self._editor_scene.delete_polygon_at(scene_pos)
            if is_contact:
                self.contactDeletionFinished.emit(1 if deleted else 0)
            event.accept()
            return

        if self._tool == EditorTool.ANTIALIAS:
            self._drag_kind = "antialias_area"
            self._drag_start_scene_pos = scene_pos
            self._editor_scene.set_preview_rect(scene_pos, scene_pos)
            event.accept()
            return

        if self._tool == EditorTool.SELECT:
            additive_selection = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            polygon_id = self._editor_scene.polygon_at(scene_pos, cycle=not additive_selection)
            if polygon_id is None:
                neighbor_path = self._editor_scene.neighbor_frame_path_at(scene_pos)
                if neighbor_path is not None:
                    event.accept()
                    return
                target_frame_id = self._pyramid_frame_at_viewport_pos(viewport_pixel)
                if target_frame_id is not None and target_frame_id != self._pyramid_current_frame_id:
                    if not self._confirm_frame_navigation():
                        event.accept()
                        return
                    self.set_current_frame_id(target_frame_id, center=False, emit_signal=True)
                    self.frameNavigationRequested.emit(target_frame_id)
                    event.accept()
                    return
                self._drag_kind = "select_area"
                self.contactMultiSelectionStarted.emit()
                self._drag_start_scene_pos = scene_pos
                self._select_press_polygon_id = None
                self._select_press_start = None
                self._editor_scene.set_preview_rect(scene_pos, scene_pos)
                event.accept()
                return
            if additive_selection:
                self.contactMultiSelectionStarted.emit()
                self.contactMultiSelectionApplyStarted.emit()
            self._editor_scene.select_polygon(polygon_id, additive=additive_selection)
            if additive_selection:
                self.contactMultiSelectionFinished.emit(self._selected_contact_count())
            if self._via_debug_inspection_enabled and polygon_id is not None:
                polygon = self._editor_scene.polygon_snapshot(polygon_id)
                if polygon is not None:
                    self.viaDebugRequested.emit(polygon)
                    event.accept()
                    return
            if self._via_debug_inspection_enabled:
                event.accept()
                return
            self._select_press_polygon_id = polygon_id
            self._select_press_start = QPointF(scene_pos)
            event.accept()
            return

        if self._tool == EditorTool.ADD_VERTEX:
            clicked_polygon_id = self._editor_scene.polygon_at_nearest_edge(scene_pos, tolerance)
            selected_polygon_id = self._editor_scene.selected_polygon_id()
            if selected_polygon_id is None:
                if clicked_polygon_id is not None:
                    self._editor_scene.select_polygon(clicked_polygon_id)
                event.accept()
                return
            if clicked_polygon_id is not None and clicked_polygon_id != selected_polygon_id:
                self._editor_scene.select_polygon(clicked_polygon_id)
                event.accept()
                return
            self._editor_scene.add_vertex_at(selected_polygon_id, scene_pos)
            event.accept()
            return

        if self._tool == EditorTool.DELETE_VERTEX:
            if self._delete_vertex_mode == DeleteVertexMode.AREA:
                self._drag_kind = "delete_area"
                self._drag_start_scene_pos = scene_pos
                self._editor_scene.preview_delete_vertices_in_rect(scene_pos, scene_pos)
                event.accept()
                return
            self._editor_scene.delete_vertex_at(scene_pos, tolerance)
            event.accept()
            return

        if self._tool == EditorTool.MOVE_VERTEX:
            selected_polygon_id = self._editor_scene.selected_polygon_id()
            clicked_polygon_id = self._editor_scene.polygon_at(scene_pos)
            if clicked_polygon_id is not None and clicked_polygon_id != selected_polygon_id:
                self._editor_scene.select_polygon(clicked_polygon_id)
            target = self._editor_scene.pick_move_target(
                scene_pos,
                self._scene_tolerance(8),
                self._scene_tolerance(10),
            )
            if target is not None:
                kind, polygon_id, target_index = target
                self._editor_scene.select_polygon(polygon_id)
                self._editor_scene.clear_move_target_preview()
                self._drag_polygon_id = polygon_id
                self._drag_origin_points = self._editor_scene.polygon_points(polygon_id)
                self._drag_preview_points = list(self._drag_origin_points)
                self._editor_scene.begin_polygon_edit_preview(polygon_id)
                self._drag_polygons_snapshot = self._editor_scene.get_polygons()
                self._drag_start_scene_pos = scene_pos
                if kind == "vertex":
                    self._drag_kind = "vertex"
                    self._drag_vertex_index = target_index
                    self._drag_edge_index = None
                else:
                    self._drag_kind = "edge"
                    self._drag_edge_index = target_index
                    self._drag_vertex_index = None
            event.accept()
            return

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        move_event_started_at = perf_counter()
        viewport_pixel = self._require_viewport().mapFrom(self, event.position().toPoint())
        self._last_pointer_viewport_pos = QPointF(viewport_pixel)
        scene_pos = self.mapToScene(self._viewport_to_view_point(viewport_pixel))
        if not (self._middle_pan_active and self._drag_kind == "brush"):
            self._last_pointer_scene_pos = scene_pos
        self._update_tool_cursors()
        if self._middle_pan_active and self._middle_pan_last_viewport is not None:
            cur = QPointF(viewport_pixel)
            dv = cur - self._middle_pan_last_viewport
            h_scroll = self.horizontalScrollBar()
            v_scroll = self.verticalScrollBar()
            if h_scroll is not None:
                h_scroll.setValue(round(h_scroll.value() - dv.x()))
            if v_scroll is not None:
                v_scroll.setValue(round(v_scroll.value() - dv.y()))
            self._middle_pan_last_viewport = cur
            event.accept()
            return
        if self._paste_mode:
            self._editor_scene.clear_conductor_hover_highlight()
            self._update_paste_preview(scene_pos)
            event.accept()
            return
        brush_drag_active = self._drag_kind == "brush"
        trace_drag_active = self._drag_kind == "trace"
        selection_drag_active = self._drag_kind == "select_area"
        polygon_drag_active = self._drag_kind == "polygon"
        zoom_active = self._zoom_animation_timer.isActive()
        if (
            brush_drag_active
            or trace_drag_active
            or selection_drag_active
            or polygon_drag_active
            or zoom_active
        ):
            self._editor_scene.clear_conductor_hover_highlight()
        else:
            self._editor_scene.sync_conductor_hover_highlight(scene_pos)
        if self._tool == EditorTool.ANTIALIAS and self._drag_kind is None:
            self._editor_scene.sync_vertex_preview(scene_pos)
        else:
            self._editor_scene.clear_vertex_preview()
        if self._tool == EditorTool.MOVE_VERTEX and self._drag_kind is None and not zoom_active and not self._middle_pan_active:
            self._editor_scene.sync_move_target_preview(
                scene_pos,
                vertex_tolerance=self._scene_tolerance(8),
                edge_tolerance=self._scene_tolerance(10),
            )
        elif not zoom_active and not self._middle_pan_active:
            self._editor_scene.clear_move_target_preview()
        if (
            self._tool == EditorTool.SELECT
            and not self.vector_edits_locked()
            and self._select_press_polygon_id is not None
            and self._drag_kind is None
            and self._select_press_start is not None
            and bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        ):
            dx = scene_pos.x() - self._select_press_start.x()
            dy = scene_pos.y() - self._select_press_start.y()
            if hypot(dx, dy) >= self._scene_tolerance(4.0):
                self._start_contact_drag_profile(self._select_press_polygon_id)
                self._drag_kind = "polygon"
                self._drag_polygon_id = self._select_press_polygon_id
                self._drag_polygon_is_contact = self._editor_scene.polygon_is_contact(
                    self._select_press_polygon_id
                )
                self._drag_origin_points = self._editor_scene.polygon_points(self._select_press_polygon_id)
                self._drag_preview_points = list(self._drag_origin_points)
                self._editor_scene.begin_polygon_edit_preview(self._select_press_polygon_id)
                self._drag_polygons_snapshot = (
                    None
                    if self._drag_polygon_is_contact
                    else self._editor_scene.get_polygons()
                )
                self._drag_start_scene_pos = QPointF(self._select_press_start)
                self._select_press_polygon_id = None
                self._select_press_start = None
        if self._tool == EditorTool.PAN:
            super().mouseMoveEvent(event)
            return
        if self._drag_kind == "rect_polygon" and self._drag_start_scene_pos is not None:
            self._editor_scene.set_preview_rect(self._drag_start_scene_pos, scene_pos)
            event.accept()
            return
        if self._tool == EditorTool.ADD_POLYGON and (
            self._effective_polygon_create_mode() == PolygonCreateMode.POINTS
            or self._editor_scene.has_pending_polygon()
        ):
            self._editor_scene.update_pending_cursor(scene_pos)
            event.accept()
            return
        if self._drag_kind == "ruler" and self._drag_start_scene_pos is not None:
            target_pos = self._ruler_target(self._drag_start_scene_pos, scene_pos, event.modifiers())
            measurement_text = self._format_ruler_measurement(self._drag_start_scene_pos, target_pos)
            self._editor_scene.set_measurement(self._drag_start_scene_pos, target_pos, measurement_text)
            self.rulerMeasurementChanged.emit(measurement_text)
            event.accept()
            return
        if self._drag_kind == "delete_area" and self._drag_start_scene_pos is not None:
            self._editor_scene.preview_delete_vertices_in_rect(self._drag_start_scene_pos, scene_pos)
            event.accept()
            return
        if self._drag_kind == "select_area" and self._drag_start_scene_pos is not None:
            self._editor_scene.set_preview_rect(self._drag_start_scene_pos, scene_pos)
            event.accept()
            return
        if self._drag_kind == "antialias_area" and self._drag_start_scene_pos is not None:
            self._editor_scene.set_preview_rect(self._drag_start_scene_pos, scene_pos)
            event.accept()
            return
        if self._drag_kind == "image_region" and self._drag_start_scene_pos is not None:
            self._editor_scene.set_preview_rect(self._drag_start_scene_pos, scene_pos)
            event.accept()
            return
        if self._drag_kind == "brush":
            if self._brush_mode == BrushMode.ANGLED and self._drag_start_scene_pos is not None:
                self._editor_scene.update_pending_cursor(_snap_to_45(self._drag_start_scene_pos, scene_pos))
            else:
                if self._brush_pan_guard:
                    # Ignore one post-pan pointer sample to avoid accidental long segment jump.
                    self._brush_pan_guard = False
                    event.accept()
                    return
                self._append_brush_point(scene_pos)
            event.accept()
            return
        if self._tool == EditorTool.TRACE_PEN and self._editor_scene.has_pending_polygon():
            last_point = self._editor_scene.pending_last_point()
            target = (
                _snap_to_45(last_point, scene_pos)
                if last_point is not None
                and (
                    self._trace_mode == BrushMode.ANGLED
                    or event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                )
                else scene_pos
            )
            self._editor_scene.update_pending_cursor(target)
            event.accept()
            return
        if self._drag_kind == "vertex" and self._drag_polygon_id is not None and self._drag_vertex_index is not None:
            self._drag_preview_points = self._editor_scene.preview_vertex_move(
                self._drag_polygon_id,
                self._drag_vertex_index,
                scene_pos,
            )
            event.accept()
            return
        if (
            self._drag_kind == "edge"
            and self._drag_polygon_id is not None
            and self._drag_edge_index is not None
            and self._drag_origin_points is not None
            and self._drag_start_scene_pos is not None
        ):
            delta = (
                scene_pos.x() - self._drag_start_scene_pos.x(),
                scene_pos.y() - self._drag_start_scene_pos.y(),
            )
            self._drag_preview_points = self._editor_scene.preview_edge_move(
                self._drag_polygon_id,
                self._drag_edge_index,
                self._drag_origin_points,
                delta,
            )
            event.accept()
            return
        if (
            self._drag_kind == "polygon"
            and self._drag_polygon_id is not None
            and self._drag_origin_points is not None
            and self._drag_start_scene_pos is not None
        ):
            dx = scene_pos.x() - self._drag_start_scene_pos.x()
            dy = scene_pos.y() - self._drag_start_scene_pos.y()
            moved = [(x_coord + dx, y_coord + dy) for x_coord, y_coord in self._drag_origin_points]
            self._drag_preview_points = self._editor_scene.preview_polygon_move(
                self._drag_polygon_id,
                moved,
            )
            if self._contact_drag_profile is not None:
                self._contact_drag_profile.record_frame(move_event_started_at)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        contact_drag_commit_started_at = (
            perf_counter()
            if self._contact_drag_profile is not None
            else None
        )
        contact_drag_status = "released"
        if event.button() == Qt.MouseButton.MiddleButton and self._middle_pan_active:
            self._middle_pan_active = False
            self._middle_pan_last_viewport = None
            self._set_polygon_overlay_hide_hold("middle", False)
            event.accept()
            return
        if self._tool == EditorTool.PAN:
            super().mouseReleaseEvent(event)
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._tool == EditorTool.SELECT
            and self._drag_kind is None
            and self._select_press_polygon_id is not None
        ):
            self._select_press_polygon_id = None
            self._select_press_start = None
            event.accept()
            return
        if self._drag_kind == "rect_polygon":
            event.accept()
            return
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton) and self._drag_kind is not None:
            if self._drag_kind == "brush":
                release_pos = self.mapToScene(event.position().toPoint())
                self._commit_brush_drag(release_pos)
            elif self._drag_kind == "ruler" and self._drag_start_scene_pos is not None:
                release_pos = self.mapToScene(event.position().toPoint())
                target_pos = self._ruler_target(self._drag_start_scene_pos, release_pos, event.modifiers())
                measurement_text = self._format_ruler_measurement(self._drag_start_scene_pos, target_pos)
                self._editor_scene.set_measurement(self._drag_start_scene_pos, target_pos, measurement_text)
                self.rulerMeasurementChanged.emit(measurement_text)
            elif self._drag_kind == "delete_area" and self._drag_start_scene_pos is not None:
                release_pos = self.mapToScene(event.position().toPoint())
                rect = QRectF(self._drag_start_scene_pos, release_pos)
                self._commit_delete_vertices_in_area(rect)
            elif self._drag_kind == "select_area" and self._drag_start_scene_pos is not None:
                release_pos = self.mapToScene(event.position().toPoint())
                rect = QRectF(self._drag_start_scene_pos, release_pos).normalized()
                self._editor_scene.clear_preview_rect()
                self.contactMultiSelectionApplyStarted.emit()
                if rect.width() < self._scene_tolerance(3) and rect.height() < self._scene_tolerance(3):
                    polygon_id = self._editor_scene.polygon_at(release_pos)
                    self._editor_scene.select_polygon(
                        polygon_id, additive=bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    )
                else:
                    self._editor_scene.select_polygons_in_rect(
                        rect,
                        additive=bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier),
                    )
                self.contactMultiSelectionFinished.emit(self._selected_contact_count())
            elif self._drag_kind == "antialias_area" and self._drag_start_scene_pos is not None:
                release_pos = self.mapToScene(event.position().toPoint())
                rect = QRectF(self._drag_start_scene_pos, release_pos).normalized()
                self._editor_scene.clear_preview_rect()
                if rect.width() < self._scene_tolerance(3) and rect.height() < self._scene_tolerance(3):
                    polygon_id = self._editor_scene.polygon_at(release_pos)
                    if polygon_id is not None:
                        self._editor_scene.antialias_polygon(polygon_id, self._antialias_grade)
                else:
                    self._editor_scene.antialias_polygons_in_rect(rect, self._antialias_grade)
            elif self._drag_kind == "image_region" and self._drag_start_scene_pos is not None:
                rect = QRectF(self._drag_start_scene_pos, self.mapToScene(event.position().toPoint())).normalized()
                image_rect = self._editor_scene.main_image_rect()
                clipped = rect.intersected(image_rect)
                self._editor_scene.clear_preview_rect()
                if clipped.width() >= 2.0 and clipped.height() >= 2.0:
                    self.imageRegionSelected.emit(clipped.x(), clipped.y(), clipped.width(), clipped.height())
            elif (
                self._drag_kind == "vertex"
                and self._drag_polygon_id is not None
                and self._drag_vertex_index is not None
                and self._drag_origin_points is not None
                and self._drag_polygons_snapshot is not None
            ):
                profile_timings: dict[str, float] = {}
                profile_total_start = perf_counter()
                profiler = cProfile.Profile() if vertex_move_profiling_enabled() else None
                profiler_enabled = False
                if profiler is not None:
                    profiler_enabled = try_enable_profiler(profiler)
                new_points = list(
                    self._drag_preview_points
                    or self._editor_scene.polygon_edit_preview_points(self._drag_polygon_id)
                    or self._drag_origin_points
                )
                old_point = self._drag_origin_points[self._drag_vertex_index]
                new_point = (
                    new_points[self._drag_vertex_index]
                    if new_points and self._drag_vertex_index < len(new_points)
                    else old_point
                )
                if (
                    new_points
                    and self._drag_vertex_index < len(new_points)
                    and _points_different(old_point, new_point)
                ):
                    phase_start = perf_counter()
                    if not is_valid_closed_polygon_vertex_move(new_points, self._drag_vertex_index):
                        profile_timings["validate"] = (perf_counter() - phase_start) * 1000.0
                        self._editor_scene.preview_vertex_move(
                            self._drag_polygon_id, self._drag_vertex_index, QPointF(old_point[0], old_point[1])
                        )
                        self._editor_scene.warn_invalid_polygon_geometry()
                    else:
                        profile_timings["validate"] = (perf_counter() - phase_start) * 1000.0
                        phase_start = perf_counter()
                        processed, accepted, focus_id = postprocess_vertex_move_edit(
                            self._drag_polygons_snapshot,
                            self._vector_geometry_settings,
                            polygon_id=self._drag_polygon_id,
                            vertex_index=self._drag_vertex_index,
                            new_point=new_point,
                        )
                        profile_timings["postprocess"] = (perf_counter() - phase_start) * 1000.0
                        if not accepted:
                            self._editor_scene.preview_vertex_move(
                                self._drag_polygon_id,
                                self._drag_vertex_index,
                                QPointF(old_point[0], old_point[1]),
                            )
                            self._editor_scene.warn_invalid_polygon_geometry()
                        else:
                            phase_start = perf_counter()
                            if focus_id is None:
                                focus_id = self._drag_polygon_id
                            profile_timings["focus"] = (perf_counter() - phase_start) * 1000.0
                            phase_start = perf_counter()
                            committed = self._editor_scene._try_commit_single_polygon_points_change(
                                self._drag_polygons_snapshot,
                                processed,
                                description="Move vertex",
                                select_polygon_id=focus_id,
                            )
                            if not committed:
                                self._editor_scene._push_polygon_set_change(
                                    self._drag_polygons_snapshot,
                                    processed,
                                    "Move vertex",
                                    select_polygon_id=focus_id,
                                )
                            profile_timings["undo_push"] = (perf_counter() - phase_start) * 1000.0
                    profile_timings["total_wall"] = (perf_counter() - profile_total_start) * 1000.0
                    if profiler_enabled and profiler is not None:
                        try_disable_profiler(profiler)
                    self._emit_vertex_move_profile(
                        profile_timings,
                        polygon_count=len(self._drag_polygons_snapshot),
                        vertex_count=sum(len(polygon.points) for polygon in self._drag_polygons_snapshot),
                        profiler=profiler if profiler_enabled else None,
                    )
                elif profiler_enabled and profiler is not None:
                    try_disable_profiler(profiler)
            elif (
                self._drag_kind == "edge"
                and self._drag_polygon_id is not None
                and self._drag_edge_index is not None
                and self._drag_origin_points is not None
                and self._drag_polygons_snapshot is not None
                and self._drag_start_scene_pos is not None
            ):
                release_pos = self.mapToScene(event.position().toPoint())
                delta = (
                    release_pos.x() - self._drag_start_scene_pos.x(),
                    release_pos.y() - self._drag_start_scene_pos.y(),
                )
                new_points = list(
                    self._drag_preview_points
                    or self._editor_scene.polygon_edit_preview_points(self._drag_polygon_id)
                    or self._drag_origin_points
                )
                if new_points and _polygon_points_different(self._drag_origin_points, new_points):
                    if not is_valid_closed_polygon_edge_move(new_points, self._drag_edge_index):
                        self._editor_scene.preview_polygon_move(self._drag_polygon_id, self._drag_origin_points)
                        self._editor_scene.warn_invalid_polygon_geometry()
                    else:
                        trial = apply_edge_translation_to_clone(
                            self._drag_polygons_snapshot,
                            self._drag_polygon_id,
                            self._drag_edge_index,
                            delta,
                        )
                        processed, accepted, _changed = postprocess_changed_polygon_edit(
                            trial,
                            self._vector_geometry_settings,
                            polygon_id=self._drag_polygon_id,
                        )
                        if accepted:
                            processed = collapse_redundant_vertices_in_polygons(processed)
                            if self._vector_geometry_settings.merge_overlapping_on_edit:
                                processed = merge_overlapping_root_families_near_polygons(
                                    processed,
                                    self._drag_polygon_id,
                                )
                            processed = collapse_redundant_vertices_in_polygons(processed)
                        if not accepted:
                            self._editor_scene.preview_polygon_move(self._drag_polygon_id, self._drag_origin_points)
                            self._editor_scene.warn_invalid_polygon_geometry()
                        else:
                            focus_id = resolve_focus_id_after_geometry_pass(
                                self._drag_polygons_snapshot,
                                self._drag_polygon_id,
                                processed,
                            )
                            self._editor_scene._push_polygon_set_change(
                                self._drag_polygons_snapshot,
                                processed,
                                "Move edge",
                                select_polygon_id=focus_id,
                            )
            elif (
                self._drag_kind == "polygon"
                and self._drag_polygon_id is not None
                and self._drag_origin_points is not None
                and (
                    self._drag_polygon_is_contact
                    or self._drag_polygons_snapshot is not None
                )
            ):
                new_points = list(
                    self._drag_preview_points
                    or self._editor_scene.polygon_edit_preview_points(self._drag_polygon_id)
                    or self._drag_origin_points
                )
                if new_points and _polygon_points_different(self._drag_origin_points, new_points):
                    if not is_valid_closed_polygon_ring(new_points):
                        contact_drag_status = "rejected"
                        self._editor_scene.preview_polygon_move(self._drag_polygon_id, self._drag_origin_points)
                        self._editor_scene.warn_invalid_polygon_geometry()
                    elif self._drag_polygon_is_contact:
                        contact_drag_status = "displayed"
                        self.undo_stack.push(
                            MovePolygonCommand(
                                self._editor_scene,
                                self._drag_polygon_id,
                                self._drag_origin_points,
                                new_points,
                            )
                        )
                    else:
                        assert self._drag_polygons_snapshot is not None
                        trial = apply_polygon_points_to_clone(
                            self._drag_polygons_snapshot,
                            self._drag_polygon_id,
                            new_points,
                        )
                        processed, accepted, _changed = postprocess_changed_polygon_edit(
                            trial,
                            self._vector_geometry_settings,
                            polygon_id=self._drag_polygon_id,
                        )
                        if accepted:
                            processed = collapse_redundant_vertices_in_polygons(processed)
                            if self._vector_geometry_settings.merge_overlapping_on_edit:
                                processed = merge_overlapping_root_families_near_polygons(
                                    processed,
                                    self._drag_polygon_id,
                                )
                            processed = collapse_redundant_vertices_in_polygons(processed)
                        if not accepted:
                            contact_drag_status = "rejected"
                            self._editor_scene.preview_polygon_move(self._drag_polygon_id, self._drag_origin_points)
                            self._editor_scene.warn_invalid_polygon_geometry()
                        else:
                            contact_drag_status = "displayed"
                            focus_id = resolve_focus_id_after_geometry_pass(
                                self._drag_polygons_snapshot,
                                self._drag_polygon_id,
                                processed,
                            )
                            self._editor_scene._push_polygon_set_change(
                                self._drag_polygons_snapshot,
                                processed,
                                "Move polygon",
                                select_polygon_id=focus_id,
                            )
            self._drag_kind = None
            self._drag_polygon_id = None
            self._drag_vertex_index = None
            self._drag_edge_index = None
            self._drag_origin_points = None
            self._drag_preview_points = None
            self._drag_start_scene_pos = None
            self._drag_polygons_snapshot = None
            self._drag_polygon_is_contact = False
            self._drag_erases = False
            self._brush_pan_guard = False
            self._editor_scene.cancel_polygon_edit_preview()
            self._editor_scene.clear_move_target_preview()
            self._update_tool_cursors()
            if contact_drag_commit_started_at is not None:
                self._finish_contact_drag_profile(
                    contact_drag_status,
                    commit_ms=(
                        perf_counter() - contact_drag_commit_started_at
                    )
                    * 1000.0,
                )
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def tabletEvent(self, event: QTabletEvent) -> None:
        scene_pos = self.mapToScene(event.position().toPoint())
        self._last_pointer_scene_pos = scene_pos
        self._update_tool_cursors()
        if self._tool != EditorTool.BRUSH:
            super().tabletEvent(event)
            return
        if event.type() == event.Type.TabletPress:
            self._start_brush_drag(scene_pos, erase=False)
            event.accept()
            return
        if event.type() == event.Type.TabletMove and self._drag_kind == "brush":
            if self._brush_mode == BrushMode.ANGLED and self._drag_start_scene_pos is not None:
                self._editor_scene.update_pending_cursor(_snap_to_45(self._drag_start_scene_pos, scene_pos))
            else:
                self._append_brush_point(scene_pos)
            event.accept()
            return
        if event.type() == event.Type.TabletRelease and self._drag_kind == "brush":
            self._commit_brush_drag(scene_pos)
            # Keep cleanup symmetrical with mouse release branch.
            self._drag_kind = None
            self._drag_polygon_id = None
            self._drag_vertex_index = None
            self._drag_edge_index = None
            self._drag_origin_points = None
            self._drag_start_scene_pos = None
            self._drag_polygons_snapshot = None
            self._drag_erases = False
            self._update_tool_cursors()
            event.accept()
            return
        super().tabletEvent(event)

    def contextMenuEvent(self, event: QContextMenuEvent | None) -> None:
        if event is None:
            return
        scene_pos = self.mapToScene(event.pos())
        polygon_id = self._editor_scene.polygon_at(scene_pos)
        polygon = self._editor_scene.polygon_snapshot(polygon_id) if polygon_id is not None else None
        if polygon is None or (polygon.category != "via" and polygon.shape_hint != "box"):
            event.ignore()
            return
        if self.vector_edits_locked() or not self._editor_scene.polygon_is_deletable(polygon):
            event.accept()
            return
        language = getattr(self._editor_scene, "_ui_language", "en")
        delete_label = (
            "Удалить переходное отверстие" if language == "ru" else "Delete transition hole"
        )
        menu = QMenu(self)
        delete_action = menu.addAction(delete_label)
        chosen = menu.exec(event.globalPos())
        if chosen == delete_action:
            self.contactDeletionStarted.emit(1)
            deleted = self._editor_scene.delete_via_at(scene_pos)
            self.contactDeletionFinished.emit(1 if deleted else 0)
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            neighbor_path = self._editor_scene.neighbor_frame_path_at(self.mapToScene(event.position().toPoint()))
            if neighbor_path:
                if self._confirm_frame_navigation():
                    self.neighborFrameActivated.emit(neighbor_path)
                event.accept()
                return
        if (
            self._tool == EditorTool.ADD_POLYGON
            and self._effective_polygon_create_mode() == PolygonCreateMode.RECTANGLE
            and self._drag_kind == "rect_polygon"
            and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton)
        ):
            started_with_right = self._drag_erases
            clicked_right = event.button() == Qt.MouseButton.RightButton
            if clicked_right == started_with_right:
                self._commit_rectangle_polygon(self.mapToScene(event.position().toPoint()))
            else:
                self._cancel_rectangle_polygon()
            event.accept()
            return
        if (
            self._tool in (EditorTool.ADD_POLYGON, EditorTool.TRACE_PEN)
            and self._editor_scene.has_pending_polygon()
            and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton)
        ):
            if self._tool == EditorTool.TRACE_PEN:
                self._finish_pending_trace()
            else:
                self._finish_pending_polygon()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is None:
            return
        if (
            event.key() == Qt.Key.Key_F
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
            and self.isEnabled()
        ):
            if not event.isAutoRepeat():
                self.fit_to_view()
            event.accept()
            return
        if (
            event.key() == Qt.Key.Key_Space
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
            and self.isEnabled()
            and (self.hasFocus() or self._require_viewport().hasFocus())
        ):
            if event.isAutoRepeat():
                event.accept()
                return
            if "space" not in self._polygon_overlay_hide_holds:
                self._set_polygon_overlay_hide_hold("space", True)
            event.accept()
            return
        if (
            event.key() == Qt.Key.Key_X
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
            and self.isEnabled()
            and (self.hasFocus() or self._require_viewport().hasFocus())
        ):
            if event.isAutoRepeat():
                event.accept()
                return
            if not self._filter_preview_hold_active:
                self._filter_preview_hold_active = True
                self.filterPreviewHoldChanged.emit(True)
            event.accept()
            return
        if (
            event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return)
            and self._tool == EditorTool.ADD_VERTEX
        ):
            self._editor_scene.select_polygon(None)
            event.accept()
            return
        if (
            event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return)
            and self._tool in (EditorTool.ADD_POLYGON, EditorTool.TRACE_PEN)
            and self._editor_scene.has_pending_polygon()
        ):
            if self._tool == EditorTool.TRACE_PEN:
                self._finish_pending_trace()
            else:
                self._finish_pending_polygon()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            paste_was_active = self._paste_mode
            if self._drag_kind == "select_area":
                self.contactMultiSelectionFinished.emit(0)
            if self._contact_drag_profile is not None:
                self._finish_contact_drag_profile("cancelled", commit_ms=0.0)
            self._editor_scene.cancel_pending_polygon()
            self._editor_scene.clear_measurement()
            self._editor_scene.clear_preview_rect()
            self._exit_paste_mode()
            if paste_was_active:
                self.contactPasteFinished.emit(0)
            if self._tool in (EditorTool.SELECT, EditorTool.ADD_VERTEX):
                self._editor_scene.select_polygon(None)
            self._select_press_polygon_id = None
            self._select_press_start = None
            if self._tool == EditorTool.RULER:
                self.rulerMeasurementChanged.emit("")
            self._drag_kind = None
            self._drag_erases = False
            self._drag_polygon_is_contact = False
            self._pending_polygon_erases = None
            self._update_tool_cursors()
            self._emit_effective_polygon_create_mode_changed()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Delete:
            if self.vector_edits_locked():
                event.accept()
                return
            deleted = self._editor_scene.selected_deletable_polygons()
            contacts = [polygon for polygon in deleted if is_via_polygon(polygon)]
            if contacts:
                self.contactDeletionStarted.emit(len(contacts))
            did_delete = self._editor_scene.delete_polygon()
            if did_delete:
                self._emit_recognized_vias_deleted(deleted)
            if contacts:
                self.contactDeletionFinished.emit(len(contacts) if did_delete else 0)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Shift and self._drag_kind is None and not event.isAutoRepeat():
            if self._cycle_active_tool_mode():
                event.accept()
                return
        if (
            event.key() == Qt.Key.Key_Shift
            and self._drag_kind == "ruler"
            and self._drag_start_scene_pos is not None
            and self._last_pointer_scene_pos is not None
        ):
            target_pos = self._ruler_target(self._drag_start_scene_pos, self._last_pointer_scene_pos, event.modifiers())
            measurement_text = self._format_ruler_measurement(self._drag_start_scene_pos, target_pos)
            self._editor_scene.set_measurement(self._drag_start_scene_pos, target_pos, measurement_text)
            self.rulerMeasurementChanged.emit(measurement_text)
            event.accept()
            return
        super().keyPressEvent(event)

    def _emit_recognized_vias_deleted(self, polygons: list[PolygonData]) -> None:
        recognized = [
            polygon.clone()
            for polygon in polygons
            if is_via_polygon(polygon) and polygon.recognition_score is not None
        ]
        if recognized:
            self.recognizedViasDeleted.emit(recognized)

    def _selected_contact_count(self) -> int:
        return self._editor_scene.selected_contact_count()

    def selected_object_counts(self) -> tuple[int, int]:
        return self._editor_scene.selected_object_counts()

    def keyReleaseEvent(self, event: QKeyEvent | None) -> None:
        if event is None:
            return
        if (
            event.key() == Qt.Key.Key_Space
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
            and not event.isAutoRepeat()
            and "space" in self._polygon_overlay_hide_holds
        ):
            self._set_polygon_overlay_hide_hold("space", False)
            event.accept()
            return
        if (
            event.key() == Qt.Key.Key_X
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
            and not event.isAutoRepeat()
            and self._filter_preview_hold_active
        ):
            self._filter_preview_hold_active = False
            self.filterPreviewHoldChanged.emit(False)
            event.accept()
            return
        if (
            event.key() == Qt.Key.Key_Shift
            and self._drag_kind == "ruler"
            and self._drag_start_scene_pos is not None
            and self._last_pointer_scene_pos is not None
        ):
            target_pos = self._ruler_target(self._drag_start_scene_pos, self._last_pointer_scene_pos, event.modifiers())
            measurement_text = self._format_ruler_measurement(self._drag_start_scene_pos, target_pos)
            self._editor_scene.set_measurement(self._drag_start_scene_pos, target_pos, measurement_text)
            self.rulerMeasurementChanged.emit(measurement_text)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _wheel_event_viewport_pixel(self, event: QWheelEvent) -> QPoint:
        # QGraphicsView::viewportEvent forwards wheel events with viewport-local position.
        return event.position().toPoint()

    def _viewport_to_view_point(self, viewport_pixel: QPoint) -> QPoint:
        return self._require_viewport().mapTo(self, viewport_pixel)

    def _view_to_viewport_point(self, view_pixel: QPoint) -> QPoint:
        return self._require_viewport().mapFrom(self, view_pixel)

    def _zoom_focus_viewport_pixel(self) -> QPoint:
        if self._last_pointer_viewport_pos is not None:
            p = self._last_pointer_viewport_pos.toPoint()
            vr = self._require_viewport().rect()
            clamped = QPoint(p.x(), p.y())
            if not vr.contains(clamped):
                return vr.center()
            return clamped
        return self._require_viewport().rect().center()

    def _queue_wheel_zoom(self, viewport_pixel: QPoint, factor: float) -> None:
        if factor == 1.0 or factor <= 0:
            return
        self._pending_wheel_zoom_factor *= factor
        self._pending_wheel_zoom_viewport_pixel = QPoint(viewport_pixel)
        if self._wheel_zoom_timer.isActive():
            return
        self._wheel_zoom_timer.start()

    def _flush_queued_wheel_zoom(self) -> None:
        factor = self._pending_wheel_zoom_factor
        viewport_pixel = self._pending_wheel_zoom_viewport_pixel
        self._pending_wheel_zoom_factor = 1.0
        self._pending_wheel_zoom_viewport_pixel = None
        if viewport_pixel is None or factor == 1.0 or factor <= 0:
            return
        self._start_zoom_animation(viewport_pixel, factor)

    def _start_zoom_animation(self, viewport_pixel: QPoint, factor: float) -> None:
        if factor == 1.0 or factor <= 0:
            return
        current_zoom = self.zoom_factor()
        base_zoom = self._zoom_animation_target_zoom if self._zoom_animation_timer.isActive() else current_zoom
        target_zoom = clamp_zoom_factor(base_zoom * float(factor))
        if abs(target_zoom - current_zoom) <= 1e-9:
            return
        self._editor_scene.pause_deferred_geometry_repair()
        self._start_scene_zoom_profile(current_zoom, target_zoom)
        self._zoom_animation_viewport_pixel = QPoint(viewport_pixel)
        self._zoom_animation_target_zoom = target_zoom
        self._update_navigation_scene_rect(target_zoom)
        self._enter_zoom_render_mode()
        if not self._zoom_animation_timer.isActive():
            self._zoom_animation_timer.start()

    def _stop_zoom_animation(self) -> None:
        self._zoom_animation_timer.stop()
        self._zoom_animation_viewport_pixel = None
        self._leave_zoom_render_mode()
        self._finish_scene_zoom_profile("interrupted")
        self._editor_scene.resume_deferred_geometry_repair()

    def _advance_zoom_animation(self) -> None:
        frame_started_at = perf_counter()
        viewport_pixel = self._zoom_animation_viewport_pixel
        if viewport_pixel is None:
            self._finish_zoom_animation(frame_started_at=frame_started_at)
            return
        current_zoom = self.zoom_factor()
        target_zoom = self._zoom_animation_target_zoom
        remaining = target_zoom - current_zoom
        if abs(remaining) <= max(_ZOOM_SETTLE_RATIO, abs(target_zoom) * _ZOOM_SETTLE_RATIO):
            next_zoom = target_zoom
            finish = True
        else:
            next_zoom = current_zoom + remaining * _ZOOM_EASING_FRACTION
            finish = False
        factor = next_zoom / current_zoom if current_zoom > 0 else 1.0
        self._apply_zoom_at_viewport_pixel(viewport_pixel, factor, update_navigation=False)
        self.zoomChanged.emit(self.zoom_factor())
        if finish:
            self._finish_zoom_animation(frame_started_at=frame_started_at)
        else:
            self._require_viewport().update()
            if self._scene_zoom_profile is not None:
                self._scene_zoom_profile.record_frame(frame_started_at)

    def _finish_zoom_animation(self, *, frame_started_at: float | None = None) -> None:
        self._zoom_animation_timer.stop()
        self._zoom_animation_viewport_pixel = None
        self._leave_zoom_render_mode()
        self._update_navigation_scene_rect()
        self._schedule_pyramid_visible_update()
        self._refresh_gradient_field_arrows()
        self._update_tool_cursors()
        self._require_viewport().update()
        if frame_started_at is not None and self._scene_zoom_profile is not None:
            self._scene_zoom_profile.record_frame(frame_started_at)
        self._finish_scene_zoom_profile("displayed")
        self._editor_scene.resume_deferred_geometry_repair()
        if (
            self._tool == EditorTool.MOVE_VERTEX
            and self._drag_kind is None
            and self._last_pointer_scene_pos is not None
        ):
            self._editor_scene.sync_move_target_preview(
                self._last_pointer_scene_pos,
                vertex_tolerance=self._scene_tolerance(8),
                edge_tolerance=self._scene_tolerance(10),
            )

    def _start_scene_zoom_profile(
        self,
        initial_zoom: float,
        target_zoom: float,
    ) -> None:
        profile = self._scene_zoom_profile
        if profile is not None:
            profile.update_target(target_zoom)
            return
        if not scene_zoom_profiling_enabled():
            return
        self._scene_zoom_profile = SceneZoomProfile.begin(
            initial_zoom=initial_zoom,
            target_zoom=target_zoom,
        )
        print(
            "[contour scene zoom profiling] "
            f"started zoom={initial_zoom:.4f} target={target_zoom:.4f}",
            flush=True,
        )

    def _finish_scene_zoom_profile(self, status: str) -> None:
        profile = self._scene_zoom_profile
        if profile is None:
            return
        self._scene_zoom_profile = None
        profile.finish()
        print(
            profile.format_summary(status=status, final_zoom=self.zoom_factor()),
            flush=True,
        )
        print(profile.format_stats(), flush=True)

    def _enter_zoom_render_mode(self) -> None:
        self._editor_scene.begin_zoom_vector_render_mode(
            minimum_contacts=_ZOOM_VECTOR_BATCH_THRESHOLD,
        )
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.NoViewportUpdate)
        self.setRenderHints(self._zooming_render_hints)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)

    def _leave_zoom_render_mode(self) -> None:
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.setRenderHints(self._steady_render_hints)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, False)
        self._editor_scene.end_zoom_vector_render_mode()

    def _clamp_current_zoom_at_viewport_pixel(self, viewport_pixel: QPoint) -> None:
        current_zoom = self.zoom_factor()
        target_zoom = clamp_zoom_factor(current_zoom)
        if abs(target_zoom - current_zoom) <= 1e-9:
            return
        self._apply_zoom_at_viewport_pixel(viewport_pixel, target_zoom / current_zoom)

    def _apply_zoom_at_viewport_pixel(
        self,
        viewport_pixel: QPoint,
        factor: float,
        *,
        update_navigation: bool = True,
    ) -> None:
        if factor == 1.0 or factor <= 0:
            return
        view_point = self._viewport_to_view_point(viewport_pixel)
        scene_anchor = self.mapToScene(view_point)
        old_anchor = self.transformationAnchor()
        try:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
            self.scale(factor, factor)
        finally:
            self.setTransformationAnchor(old_anchor)
        view_mapped = self.mapFromScene(scene_anchor)
        vp_mapped = self._view_to_viewport_point(view_mapped)
        dh, dv = viewport_scroll_correction_after_scale_reanchor(
            (viewport_pixel.x(), viewport_pixel.y()),
            (vp_mapped.x(), vp_mapped.y()),
        )
        h_scroll = self.horizontalScrollBar()
        v_scroll = self.verticalScrollBar()
        if h_scroll is not None:
            h_scroll.setValue(h_scroll.value() + dh)
        if v_scroll is not None:
            v_scroll.setValue(v_scroll.value() + dv)
        if update_navigation:
            self._update_navigation_scene_rect()

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        if event is None:
            return
        super().resizeEvent(event)
        self._update_navigation_scene_rect()
        self._schedule_pyramid_visible_update()
        self._schedule_gradient_field_arrows_update()

    def leaveEvent(self, event: QEvent | None) -> None:
        if event is None:
            return
        self._editor_scene.clear_conductor_hover_highlight()
        self._editor_scene.clear_vertex_preview()
        self._editor_scene.clear_move_target_preview()
        self._editor_scene.hide_tool_cursors()
        super().leaveEvent(event)

    def _scene_tolerance(self, pixels: float | int) -> float:
        px = max(1, int(round(pixels)))
        start = self.mapToScene(QPoint(0, 0))
        end = self.mapToScene(QPoint(px, 0))
        return max(1.0, abs(end.x() - start.x()))

    def _update_navigation_scene_rect(self, zoom: float | None = None) -> None:
        if self.pyramid_mode_enabled() and self._pyramid_layout is not None:
            base_rect = QRectF(self._pyramid_layout.scene_rect(self._pyramid_current_lod))
            base_rect = base_rect.united(QRectF(self._editor_scene.navigation_base_rect()))
        else:
            base_rect = QRectF(self._editor_scene.navigation_base_rect())
        if base_rect.width() <= 0.0 or base_rect.height() <= 0.0:
            self.setSceneRect(base_rect)
            self._update_minimap_overlay()
            return
        zoom = self.zoom_factor() if zoom is None else max(1e-6, float(zoom))
        viewport_rect = self._require_viewport().rect()
        margin_x = float(viewport_rect.width()) / max(zoom, 1e-6) + 2.0
        margin_y = float(viewport_rect.height()) / max(zoom, 1e-6) + 2.0
        self.setSceneRect(base_rect.adjusted(-margin_x, -margin_y, margin_x, margin_y))
        self._update_minimap_overlay()

    def _refresh_minimap_thumbnail(self) -> None:
        pixmap = self._editor_scene.main_image_pixmap()
        image_rect = self._editor_scene.main_image_rect()
        self._minimap.set_image(pixmap, image_rect)
        self._update_minimap_overlay()

    def _position_minimap(self) -> None:
        viewport = self.viewport()
        if viewport is None:
            return
        viewport_geom = viewport.geometry()
        margin = int(MINIMAP_VIEWPORT_MARGIN_PX)
        x = viewport_geom.right() - self._minimap.width() - margin
        y = viewport_geom.bottom() - self._minimap.height() - margin
        self._minimap.move(max(viewport_geom.left() + margin, x), max(viewport_geom.top() + margin, y))

    def _update_minimap_overlay(self) -> None:
        if not self._minimap.has_image():
            self._minimap.hide()
            return
        self._minimap.show()
        self._minimap.raise_()
        self._position_minimap()
        self._update_minimap_mouse_interaction()
        viewport = self.viewport()
        if viewport is None:
            return
        viewport_scene = self.mapToScene(viewport.rect()).boundingRect()
        self._minimap.set_viewport_scene_rect(viewport_scene)

    def _update_minimap_mouse_interaction(self) -> None:
        self._minimap.set_mouse_interactive(self._tool not in _MINIMAP_DRAWING_TOOLS)

    def _on_minimap_scene_point(self, scene_pos: QPointF) -> None:
        self.centerOn(scene_pos)
        self._update_navigation_scene_rect()

    def _on_minimap_scene_delta(self, delta: QPointF) -> None:
        viewport = self.viewport()
        if viewport is None:
            return
        center = self.mapToScene(viewport.rect().center())
        self.centerOn(QPointF(center.x() + delta.x(), center.y() + delta.y()))
        self._update_navigation_scene_rect()

    def _append_brush_point(self, scene_pos: QPointF) -> None:
        target = scene_pos
        if self._brush_mode == BrushMode.ANGLED:
            last_point = self._editor_scene.pending_last_point()
            if last_point is not None:
                target = _snap_to_45(last_point, scene_pos)
        last_point = self._editor_scene.pending_last_point()
        if last_point is not None and hypot(target.x() - last_point.x(), target.y() - last_point.y()) < 1.0:
            # Brush vertex spacing rule: at least 1 image pixel in scene/image coordinates.
            return
        self._editor_scene.append_brush_vertex(target, self._brush_thickness)

    def _start_rectangle_polygon(self, scene_pos: QPointF, *, erase: bool) -> None:
        self._drag_kind = "rect_polygon"
        self._drag_start_scene_pos = QPointF(scene_pos)
        self._drag_erases = bool(erase)
        self._editor_scene.set_preview_rect(scene_pos, scene_pos)

    def _commit_rectangle_polygon(self, scene_pos: QPointF) -> None:
        start = self._drag_start_scene_pos
        erase = self._drag_erases
        self._drag_kind = None
        self._drag_start_scene_pos = None
        self._drag_erases = False
        if start is None:
            self._editor_scene.clear_preview_rect()
            return
        self._editor_scene.add_rectangle_polygon(start, scene_pos, erase=erase)

    def _cancel_rectangle_polygon(self) -> None:
        if self._drag_kind != "rect_polygon":
            return
        self._drag_kind = None
        self._drag_start_scene_pos = None
        self._drag_erases = False
        self._editor_scene.clear_preview_rect()

    def _cycle_active_tool_mode(self) -> bool:
        if self._tool == EditorTool.ADD_POLYGON:
            if self._editor_scene.has_pending_polygon() or self._drag_kind == "rect_polygon":
                return False
            next_mode = (
                PolygonCreateMode.RECTANGLE
                if self._polygon_create_mode == PolygonCreateMode.POINTS
                else PolygonCreateMode.POINTS
            )
            self.set_polygon_create_mode(next_mode)
            return True
        if self._tool == EditorTool.BRUSH:
            order = [BrushMode.FREEFORM, BrushMode.ANGLED]
            index = order.index(self._brush_mode) if self._brush_mode in order else 0
            self.set_brush_mode(order[(index + 1) % len(order)])
            return True
        if self._tool == EditorTool.TRACE_PEN:
            order = [BrushMode.FREEFORM, BrushMode.ANGLED]
            index = order.index(self._trace_mode) if self._trace_mode in order else 0
            self.set_trace_mode(order[(index + 1) % len(order)])
            return True
        if self._tool == EditorTool.DELETE_VERTEX:
            next_mode = (
                DeleteVertexMode.AREA
                if self._delete_vertex_mode == DeleteVertexMode.SINGLE
                else DeleteVertexMode.SINGLE
            )
            self.set_delete_vertex_mode(next_mode)
            return True
        return False

    def _start_brush_drag(self, scene_pos: QPointF, *, erase: bool) -> None:
        self._drag_erases = bool(erase)
        self._drag_kind = "brush"
        self._drag_start_scene_pos = scene_pos
        self._editor_scene.start_pending_polygon(for_brush=True)
        self._editor_scene.set_pending_path_width(self._brush_thickness, cosmetic=False)
        self._append_brush_point(scene_pos)

    def _commit_brush_drag(self, release_pos: QPointF) -> None:
        if self._brush_mode == BrushMode.ANGLED and self._drag_start_scene_pos is not None:
            end_point = _snap_to_45(self._drag_start_scene_pos, release_pos)
            brush_points = [
                (self._drag_start_scene_pos.x(), self._drag_start_scene_pos.y()),
                (end_point.x(), end_point.y()),
            ]
        else:
            if not self._brush_pan_guard:
                self._append_brush_point(release_pos)
            brush_points = self._editor_scene.pending_points_snapshot()
        self._brush_pan_guard = False
        self._editor_scene.add_brush_stroke(brush_points, self._brush_thickness, erase=self._drag_erases)

    def _append_trace_point(self, scene_pos: QPointF, *, erase: bool, snap: bool = False) -> None:
        if self._editor_scene.has_pending_polygon():
            if self._pending_polygon_erases is None:
                self._pending_polygon_erases = bool(erase)
            elif bool(erase) != self._pending_polygon_erases:
                self._finish_pending_trace()
                return
        else:
            self._pending_polygon_erases = bool(erase)
            self._editor_scene.start_pending_polygon(for_brush=True)
            self._editor_scene.set_pending_path_width(self._trace_width, cosmetic=False)

        target = scene_pos
        last_point = self._editor_scene.pending_last_point()
        if (snap or self._trace_mode == BrushMode.ANGLED) and last_point is not None:
            target = _snap_to_45(last_point, scene_pos)
        if last_point is not None and hypot(target.x() - last_point.x(), target.y() - last_point.y()) < 1.0:
            return
        self._editor_scene.append_pending_point(target)

    def _finish_pending_trace(self) -> None:
        points = self._editor_scene.pending_points_snapshot()
        self._editor_scene.add_trace_stroke(points, self._trace_width, erase=bool(self._pending_polygon_erases))
        self._pending_polygon_erases = None

    def _update_tool_cursors(self) -> None:
        self._editor_scene.set_brush_cursor(
            self._last_pointer_scene_pos,
            self._trace_width if self._tool == EditorTool.TRACE_PEN else self._brush_thickness,
            self._tool in (EditorTool.BRUSH, EditorTool.TRACE_PEN),
        )
        self._editor_scene.set_via_cursor(
            self._last_pointer_scene_pos,
            self._via_width,
            self._via_height,
            self._tool == EditorTool.ADD_VIA,
        )

    def _finish_pending_polygon(self) -> None:
        if self._pending_polygon_erases:
            self._editor_scene.subtract_pending_polygon()
        else:
            self._editor_scene.finish_pending_polygon()
        self._pending_polygon_erases = None
        self._emit_effective_polygon_create_mode_changed()

    def _ruler_target(self, start: QPointF, target: QPointF, modifiers: Qt.KeyboardModifier) -> QPointF:
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            return _snap_to_45(start, target)
        return QPointF(target)

    def _format_ruler_measurement(self, start: QPointF, end: QPointF) -> str:
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        distance = hypot(dx, dy)
        return f"L={distance:.1f}px, dX={dx:.1f}, dY={dy:.1f}"

    def _start_contact_drag_profile(self, polygon_id: int | None) -> None:
        if (
            not contact_drag_profiling_enabled()
            or not self._editor_scene.polygon_is_contact(polygon_id)
            or polygon_id is None
        ):
            return
        if self._contact_drag_profile is not None:
            self._finish_contact_drag_profile("superseded", commit_ms=0.0)
        contact_count = self._editor_scene.contact_count()
        self._contact_drag_profile = ContactDragProfile.begin(
            polygon_id=polygon_id,
            contact_count=contact_count,
        )
        print(
            "[contour contact drag profiling] "
            f"started polygon_id={polygon_id} "
            f"contacts={contact_count}",
            flush=True,
        )

    def _finish_contact_drag_profile(
        self,
        status: str,
        *,
        commit_ms: float,
    ) -> None:
        profile = self._contact_drag_profile
        if profile is None:
            return
        self._contact_drag_profile = None
        profile.finish(commit_ms=commit_ms)
        print(profile.format_summary(status=status), flush=True)
        print(profile.format_stats(), flush=True)

    def _emit_vertex_move_profile(
        self,
        timings_ms: dict[str, float],
        *,
        polygon_count: int,
        vertex_count: int,
        profiler: cProfile.Profile | None,
    ) -> None:
        if not vertex_move_profiling_enabled():
            return
        total_ms = timings_ms.get("total_wall", sum(timings_ms.values()))
        detail = " ".join(
            f"{name}={elapsed:.3f}ms" for name, elapsed in timings_ms.items() if name != "total_wall"
        )
        message = (
            f"[contour vertex profiling] total={total_ms:.3f}ms polygons={polygon_count} "
            f"vertices={vertex_count} {detail}"
        )
        print(message, flush=True)
        if profiler is None:
            return
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        top_lines = vertex_move_top_lines()
        stats.print_stats(top_lines)
        report = stream.getvalue()
        print(f"[contour vertex profiling stats] top={top_lines}", flush=True)
        print(report, flush=True)

    def _cancel_move_vertex_tool_profile_finish(self) -> None:
        self._move_vertex_tool_profile_generation += 1
        self._move_vertex_tool_profile_paint_started_at = None

    def _schedule_move_vertex_tool_profile_finish(self) -> None:
        self._move_vertex_tool_profile_generation += 1
        generation = self._move_vertex_tool_profile_generation
        self._move_vertex_tool_profile_paint_started_at = perf_counter()
        QTimer.singleShot(
            0,
            lambda: self._finish_move_vertex_tool_profile_after_paint(generation),
        )

    def _finish_move_vertex_tool_profile_after_paint(self, generation: int) -> None:
        if generation != self._move_vertex_tool_profile_generation:
            return
        profile = self._move_vertex_tool_profile
        if profile is None:
            return
        paint_started_at = self._move_vertex_tool_profile_paint_started_at
        if paint_started_at is not None:
            profile.note_timing(
                "paint",
                (perf_counter() - paint_started_at) * 1000.0,
            )
        self._finish_move_vertex_tool_profile("displayed")

    def _finish_move_vertex_tool_profile(self, status: str) -> None:
        profile = self._move_vertex_tool_profile
        if profile is None:
            return
        self._move_vertex_tool_profile = None
        self._move_vertex_tool_profile_paint_started_at = None
        profile.note_timing("total_wall", profile.total_wall_ms())
        profile.finish()
        print(profile.format_summary(status=status), flush=True)
        print(profile.format_stats(), flush=True)

    def _commit_delete_vertices_in_area(self, rect: QRectF) -> None:
        profiling_enabled = delete_area_profiling_enabled()
        profile_timings: dict[str, float] = {}
        profile_total_start = perf_counter()
        profiler = cProfile.Profile() if profiling_enabled else None
        profiler_enabled = False
        if profiler is not None:
            profiler_enabled = try_enable_profiler(profiler)
        try:
            polygon_hits = vertex_hits = vertex_total = 0
            if profiling_enabled:
                phase_start = perf_counter()
                polygon_hits, vertex_hits, vertex_total = self._delete_area_profile_counts(rect)
                profile_timings["count"] = (perf_counter() - phase_start) * 1000.0
            phase_start = perf_counter()
            deleted = self._editor_scene.delete_vertices_in_rect(rect)
            if profiling_enabled:
                profile_timings["delete"] = (perf_counter() - phase_start) * 1000.0
            phase_start = perf_counter()
            self._editor_scene.clear_preview_rect()
            if profiling_enabled:
                profile_timings["clear_preview"] = (perf_counter() - phase_start) * 1000.0
                profile_timings["total_wall"] = (perf_counter() - profile_total_start) * 1000.0
                self._emit_delete_area_profile(
                    profile_timings,
                    polygon_hits=polygon_hits,
                    vertex_hits=vertex_hits,
                    vertex_total=vertex_total,
                    deleted=deleted,
                    profiler=profiler if profiler_enabled else None,
                )
        finally:
            if profiler_enabled and profiler is not None:
                try_disable_profiler(profiler)

    def _delete_area_profile_counts(self, rect: QRectF) -> tuple[int, int, int]:
        normalized = rect.normalized()
        polygon_hits = 0
        vertex_hits = 0
        vertex_total = 0
        for polygon in self._editor_scene.get_polygons():
            vertex_total += len(polygon.points)
            matched = sum(
                1
                for x_coord, y_coord in polygon.points
                if normalized.contains(QPointF(x_coord, y_coord))
            )
            if matched:
                polygon_hits += 1
                vertex_hits += matched
        return polygon_hits, vertex_hits, vertex_total

    def _emit_delete_area_profile(
        self,
        timings_ms: dict[str, float],
        *,
        polygon_hits: int,
        vertex_hits: int,
        vertex_total: int,
        deleted: int,
        profiler: cProfile.Profile | None,
    ) -> None:
        if not delete_area_profiling_enabled():
            return
        total_ms = timings_ms.get("total_wall", sum(timings_ms.values()))
        detail = " ".join(
            f"{name}={elapsed:.3f}ms" for name, elapsed in timings_ms.items() if name != "total_wall"
        )
        message = (
            f"[contour delete-area profiling] total={total_ms:.3f}ms "
            f"polygons_hit={polygon_hits} vertices_hit={vertex_hits} "
            f"vertices_total={vertex_total} deleted={deleted} {detail}"
        )
        print(message, flush=True)
        if profiler is None:
            return
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        top_lines = delete_area_top_lines()
        stats.print_stats(top_lines)
        report = stream.getvalue()
        print(f"[contour delete-area profiling stats] top={top_lines}", flush=True)
        print(report, flush=True)
