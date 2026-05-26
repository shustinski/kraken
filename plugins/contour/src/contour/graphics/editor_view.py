from __future__ import annotations

import cProfile
import io
import pstats
from collections import OrderedDict
from math import hypot, log2
from time import perf_counter

from typing import cast

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
    QColor,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QGuiApplication,
    QResizeEvent,
    QShortcut,
    QTabletEvent,
    QUndoStack,
    QWheelEvent,
)
from PyQt6.QtWidgets import QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsView, QWidget

from ..adapters.qt.pyramid import PyramidFrameLoadRunnable
from ..application.frame_lod import FixedGridFrameLayout, PyramidFrameStore
from ..application.processing import DisplaySettings
from ..application.vector_geometry_postprocess import (
    VectorGeometrySettings,
    apply_polygon_points_to_clone,
    apply_vertex_position_to_clone,
    postprocess_changed_polygon_only,
    resolve_focus_id_after_geometry_pass,
)
from ..commands import ReplacePolygonSetCommand
from ..domain import Point, PolygonData, integer_points
from ..domain.polygon_ring import is_valid_closed_polygon_vertex_move
from ..infrastructure.profiling import (
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
from .tool_mode_logic import effective_polygon_create_mode, normalize_editor_tool
from .tools import BrushMode, DeleteVertexMode, EditorTool, PolygonCreateMode
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
    deleteVertexModeChanged = pyqtSignal(object)
    zoomChanged = pyqtSignal(float)
    neighborFrameActivated = pyqtSignal(str)
    viaDebugRequested = pyqtSignal(object)
    metalOverlayDetailRequested = pyqtSignal(str, str)
    middlePreviewHoldChanged = pyqtSignal(bool)
    frameNavigationRequested = pyqtSignal(object)
    currentFrameChanged = pyqtSignal(object)
    editorViewportChanged = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        self._editor_scene = PolygonEditorScene()
        super().__init__(self._editor_scene, parent)
        self._opengl_viewport_enabled = self._configure_opengl_viewport()
        self._steady_render_hints = QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        self._zooming_render_hints = QPainter.RenderHint(0)
        self.setRenderHints(self._steady_render_hints)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
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

        self._tool = EditorTool.SELECT
        self._polygon_create_mode = PolygonCreateMode.POINTS
        self._brush_mode = BrushMode.FREEFORM
        self._brush_thickness = 12.0
        self._trace_width = 12.0
        self._via_width = 12.0
        self._via_height = 12.0
        self._antialias_grade = 1
        self._delete_vertex_mode = DeleteVertexMode.SINGLE
        self._select_press_polygon_id: int | None = None
        self._select_press_start: QPointF | None = None
        self._drag_kind: str | None = None
        self._drag_polygon_id: int | None = None
        self._drag_vertex_index: int | None = None
        self._drag_origin_points: list[tuple[float, float]] | None = None
        self._drag_start_scene_pos: QPointF | None = None
        self._last_pointer_scene_pos: QPointF | None = None
        self._drag_erases = False
        self._pending_polygon_erases: bool | None = None
        self._middle_pan_active = False
        self._middle_pan_last_viewport: QPointF | None = None
        self._polygon_overlays_visible_before_space_hold: bool | None = None
        self._last_pointer_viewport_pos: QPointF | None = None
        self._image_click_mode = False
        self._image_region_selection_mode = False
        self._via_debug_inspection_enabled = False
        self._clipboard_polygons: list[PolygonData] = []
        self._clipboard_anchor = QPointF(0.0, 0.0)
        self._paste_mode = False
        self._paste_preview_items: list[QGraphicsPathItem] = []
        self._vector_geometry_settings = VectorGeometrySettings()
        self._drag_polygons_snapshot: list[PolygonData] | None = None
        self._brush_pan_guard = False
        self._pending_wheel_zoom_factor = 1.0
        self._pending_wheel_zoom_viewport_pixel: QPoint | None = None
        self._wheel_zoom_timer = QTimer(self)
        self._wheel_zoom_timer.setSingleShot(True)
        self._wheel_zoom_timer.setInterval(_WHEEL_ZOOM_COALESCE_MS)
        self._wheel_zoom_timer.timeout.connect(self._flush_queued_wheel_zoom)
        self._zoom_animation_timer = QTimer(self)
        self._zoom_animation_timer.setInterval(_ZOOM_ANIMATION_FRAME_MS)
        self._zoom_animation_timer.timeout.connect(self._advance_zoom_animation)
        self._zoom_animation_viewport_pixel: QPoint | None = None
        self._zoom_animation_target_zoom = 1.0
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
        self._pyramid_selection_item = QGraphicsRectItem()
        selection_pen = QPen(QColor("#22D3EE"), 2.0)
        selection_pen.setCosmetic(True)
        self._pyramid_selection_item.setPen(selection_pen)
        self._pyramid_selection_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._pyramid_selection_item.setZValue(-17)
        self._editor_scene.addItem(self._pyramid_selection_item)
        self._pyramid_selection_item.hide()

        self._editor_scene.polygonsChanged.connect(self.polygonsEdited.emit)
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
            shortcut.activated.connect(lambda t=tool: self.set_tool(t))

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

    def set_tool(self, tool: EditorTool) -> None:
        tool = normalize_editor_tool(tool)
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
        if tool != EditorTool.DELETE_VERTEX:
            self._editor_scene.clear_preview_rect()
        if tool != EditorTool.RULER:
            self._editor_scene.clear_measurement()
            self.rulerMeasurementChanged.emit("")
        if tool != EditorTool.ANTIALIAS:
            self._editor_scene.clear_vertex_preview()
        self._update_tool_cursors()
        self.toolChanged.emit(tool)
        self._emit_effective_polygon_create_mode_changed()

    def set_polygon_create_mode(self, mode: PolygonCreateMode) -> None:
        mode = PolygonCreateMode(mode)
        changed = mode != self._polygon_create_mode
        self._polygon_create_mode = mode
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
        self._brush_thickness = max(1.0, float(thickness))
        if self._tool == EditorTool.BRUSH:
            self._editor_scene.set_pending_path_width(self._brush_thickness, cosmetic=False)
        self._update_tool_cursors()

    def set_trace_width(self, width: float) -> None:
        self._trace_width = max(1.0, float(width))
        if self._tool == EditorTool.TRACE_PEN:
            self._editor_scene.set_pending_path_width(self._trace_width, cosmetic=False)
        self._update_tool_cursors()

    def set_via_size(self, width: float, height: float) -> None:
        self._via_width = max(1.0, float(width))
        self._via_height = max(1.0, float(height))
        self._update_tool_cursors()

    def set_vector_geometry_settings(self, settings: VectorGeometrySettings | None) -> None:
        self._vector_geometry_settings = settings if settings is not None else VectorGeometrySettings()
        self._editor_scene.set_vector_geometry_settings(settings)

    def set_delete_vertex_mode(self, mode: DeleteVertexMode) -> None:
        mode = DeleteVertexMode(mode)
        changed = mode != self._delete_vertex_mode
        self._delete_vertex_mode = mode
        self._editor_scene.clear_preview_rect()
        if changed:
            self.deleteVertexModeChanged.emit(mode)

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

    def set_image(self, image) -> None:
        previous_rect = QRectF(self._editor_scene.sceneRect())
        self._editor_scene.set_image(image)
        previous_was_placeholder = previous_rect.width() <= 1.0 and previous_rect.height() <= 1.0
        if previous_was_placeholder:
            self.fit_to_view()
        self._update_navigation_scene_rect()

    def set_image_pixmap(self, pixmap: QPixmap) -> None:
        previous_rect = QRectF(self._editor_scene.sceneRect())
        self._editor_scene.set_image_pixmap(pixmap)
        previous_was_placeholder = previous_rect.width() <= 1.0 and previous_rect.height() <= 1.0
        if previous_was_placeholder:
            self.fit_to_view()
        self._update_navigation_scene_rect()

    def set_polygons(self, polygons: list[PolygonData], *, emit_signal: bool = True) -> None:
        self._editor_scene.set_polygons(polygons, emit_signal=emit_signal)

    def get_polygons(self) -> list[PolygonData]:
        return self._editor_scene.get_polygons()

    def antialias_selected_polygons(self, grade: int) -> bool:
        return self._editor_scene.antialias_selected_polygons(grade)

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
        should_enable = bool(store.has_zarr()) if enabled is None else bool(enabled and store.has_zarr())
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

    def set_random_object_colors_enabled(self, enabled: bool) -> None:
        self._editor_scene.set_random_object_colors_enabled(enabled)

    def set_extra_layers(self, layers: list[dict[str, object]]) -> None:
        self._editor_scene.set_extra_layers(layers)

    def set_gradient_overlay(self, image, opacity: float = 0.45) -> None:
        self._editor_scene.set_gradient_overlay(image, opacity)

    def clear_gradient_overlay(self) -> None:
        self._editor_scene.clear_gradient_overlay()

    def set_gradient_overlay_opacity(self, opacity: float) -> None:
        self._editor_scene.set_gradient_overlay_opacity(opacity)

    def set_polygon_category_visible(self, category: str, visible: bool) -> None:
        self._editor_scene.set_polygon_category_visible(category, visible)

    def set_ui_language(self, language: str | None) -> None:
        self._editor_scene.set_ui_language(language)

    def set_image_click_mode(self, enabled: bool) -> None:
        self._image_click_mode = bool(enabled)

    def set_image_region_selection_mode(self, enabled: bool) -> None:
        self._image_region_selection_mode = bool(enabled)
        if not enabled:
            self._editor_scene.clear_preview_rect()

    def fit_to_view(self) -> None:
        rect = self._editor_scene.main_image_rect()
        if rect.width() > 0 and rect.height() > 0:
            self._stop_zoom_animation()
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._clamp_current_zoom_at_viewport_pixel(self._require_viewport().rect().center())
            self._update_navigation_scene_rect()
            self.zoomChanged.emit(self.zoom_factor())

    def center_main_image(self) -> None:
        rect = self._editor_scene.main_image_rect()
        if rect.width() > 0 and rect.height() > 0:
            self.centerOn(rect.center())
            self._update_navigation_scene_rect()

    def zoom_in(self) -> None:
        self._start_zoom_animation(self._zoom_focus_viewport_pixel(), DEFAULT_ZOOM_STEP_FACTOR)

    def zoom_out(self) -> None:
        self._start_zoom_animation(self._zoom_focus_viewport_pixel(), 1.0 / DEFAULT_ZOOM_STEP_FACTOR)

    def undo(self) -> None:
        self.undo_stack.undo()

    def redo(self) -> None:
        self.undo_stack.redo()

    def copy_selected(self) -> None:
        polygons = self._editor_scene.selected_polygons()
        if not polygons:
            return
        self._clipboard_polygons = [polygon.clone() for polygon in polygons]
        self._clipboard_anchor = _polygons_center(self._clipboard_polygons)

    def cut_selected(self) -> None:
        self.copy_selected()
        if self._clipboard_polygons:
            self._editor_scene.delete_polygon()

    def start_paste_mode(self) -> None:
        if not self._clipboard_polygons:
            return
        self._paste_mode = True
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

    def _update_paste_preview(self, scene_pos: QPointF | None) -> None:
        self._clear_paste_preview()
        if not self._paste_mode or scene_pos is None:
            return
        dx = scene_pos.x() - self._clipboard_anchor.x()
        dy = scene_pos.y() - self._clipboard_anchor.y()
        pen = QPen(QColor("#38BDF8"), 1.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        brush = QColor("#38BDF8")
        brush.setAlpha(42)
        for polygon in self._clipboard_polygons:
            shifted = polygon.clone()
            shifted.points = cast(
                list[Point],
                integer_points([(x + dx, y + dy) for x, y in shifted.points]),
            )
            path = QPainterPath()
            if shifted.shape_hint == "box" or shifted.category == "via":
                x_values = [point[0] for point in shifted.points]
                y_values = [point[1] for point in shifted.points]
                path.addEllipse(
                    QRectF(min(x_values), min(y_values), max(x_values) - min(x_values), max(y_values) - min(y_values))
                )
            else:
                if shifted.points:
                    path.moveTo(shifted.points[0][0], shifted.points[0][1])
                    for x_coord, y_coord in shifted.points[1:]:
                        path.lineTo(x_coord, y_coord)
                    path.closeSubpath()
            item = QGraphicsPathItem(path)
            item.setZValue(40)
            item.setPen(pen)
            item.setBrush(QBrush(brush))
            self._editor_scene.addItem(item)
            self._paste_preview_items.append(item)

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
            self.middlePreviewHoldChanged.emit(True)
            event.accept()
            return

        if self._paste_mode and event.button() == Qt.MouseButton.LeftButton:
            self._editor_scene.add_cloned_polygons_at(self._clipboard_polygons, self._clipboard_anchor, scene_pos)
            self._update_paste_preview(scene_pos)
            event.accept()
            return

        if self._image_region_selection_mode and event.button() == Qt.MouseButton.LeftButton:
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
            if (
                event.button() == Qt.MouseButton.RightButton
                and not self._editor_scene.has_pending_polygon()
                and self._editor_scene.polygon_at(scene_pos) is not None
            ):
                self._editor_scene.delete_polygon_at(scene_pos)
                event.accept()
                return
            if create_mode == PolygonCreateMode.RECTANGLE and event.button() == Qt.MouseButton.LeftButton:
                self._drag_kind = "rect_polygon"
                self._drag_start_scene_pos = scene_pos
                self._drag_erases = False
                self._editor_scene.set_preview_rect(scene_pos, scene_pos)
                event.accept()
                return
            if create_mode == PolygonCreateMode.RECTANGLE and event.button() == Qt.MouseButton.RightButton:
                self._drag_kind = "rect_polygon"
                self._drag_start_scene_pos = scene_pos
                self._drag_erases = True
                self._editor_scene.set_preview_rect(scene_pos, scene_pos)
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

        if self._tool == EditorTool.ADD_VIA and event.button() == Qt.MouseButton.LeftButton:
            self._editor_scene.add_via_at(scene_pos, self._via_width, self._via_height)
            self._update_tool_cursors()
            event.accept()
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
            self._editor_scene.delete_polygon_at(scene_pos)
            event.accept()
            return

        if self._tool == EditorTool.ANTIALIAS:
            self._drag_kind = "antialias_area"
            self._drag_start_scene_pos = scene_pos
            self._editor_scene.set_preview_rect(scene_pos, scene_pos)
            event.accept()
            return

        if self._tool == EditorTool.SELECT:
            polygon_id = self._editor_scene.polygon_at(scene_pos)
            additive_selection = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            if polygon_id is None:
                target_frame_id = self._pyramid_frame_at_viewport_pos(viewport_pixel)
                if target_frame_id is not None and target_frame_id != self._pyramid_current_frame_id:
                    self.set_current_frame_id(target_frame_id, center=True, emit_signal=True)
                    self.frameNavigationRequested.emit(target_frame_id)
                    event.accept()
                    return
                self._drag_kind = "select_area"
                self._drag_start_scene_pos = scene_pos
                self._select_press_polygon_id = None
                self._select_press_start = None
                self._editor_scene.set_preview_rect(scene_pos, scene_pos)
                event.accept()
                return
            self._editor_scene.select_polygon(polygon_id, additive=additive_selection)
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
            clicked_polygon_id = self._editor_scene.polygon_at(scene_pos)
            selected_polygon_id = self._editor_scene.selected_polygon_id()
            polygon_id = selected_polygon_id or clicked_polygon_id
            if clicked_polygon_id is not None and clicked_polygon_id != selected_polygon_id:
                self._editor_scene.select_polygon(clicked_polygon_id)
                polygon_id = clicked_polygon_id
            if polygon_id is not None:
                self._editor_scene.add_vertex_at(polygon_id, scene_pos)
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
            target_polygon_id = selected_polygon_id or clicked_polygon_id
            if clicked_polygon_id is not None and clicked_polygon_id != selected_polygon_id:
                self._editor_scene.select_polygon(clicked_polygon_id)
                target_polygon_id = clicked_polygon_id
            hit = self._editor_scene.vertex_at(scene_pos, tolerance)
            if hit is None:
                # Practical fallback: users often click near the handle, not exactly on it.
                hit = self._editor_scene.vertex_at(scene_pos, self._scene_tolerance(12))
            if hit is None and target_polygon_id is not None:
                # If a polygon is clicked but no handle was hit, move the nearest vertex.
                hit = self._editor_scene.nearest_vertex_in_polygon(target_polygon_id, scene_pos)
            if hit is None:
                # Last-resort fallback in move mode: pick nearest vertex globally.
                hit = self._editor_scene.nearest_vertex(scene_pos)
            if hit is not None:
                polygon_id, vertex_index = hit
                self._editor_scene.select_polygon(polygon_id)
                self._drag_kind = "vertex"
                self._drag_polygon_id = polygon_id
                self._drag_vertex_index = vertex_index
                self._drag_origin_points = self._editor_scene.polygon_points(polygon_id)
                self._drag_polygons_snapshot = self._editor_scene.get_polygons()
                self._drag_start_scene_pos = scene_pos
            event.accept()
            return

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
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
        if self._tool in (EditorTool.BRUSH, EditorTool.TRACE_PEN) or brush_drag_active or trace_drag_active:
            self._editor_scene.clear_conductor_hover_highlight()
        else:
            self._editor_scene.sync_conductor_hover_highlight(scene_pos)
        if self._tool == EditorTool.ANTIALIAS and self._drag_kind is None:
            self._editor_scene.sync_vertex_preview(scene_pos)
        else:
            self._editor_scene.clear_vertex_preview()
        if (
            self._tool == EditorTool.SELECT
            and self._select_press_polygon_id is not None
            and self._drag_kind is None
            and self._select_press_start is not None
            and bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        ):
            dx = scene_pos.x() - self._select_press_start.x()
            dy = scene_pos.y() - self._select_press_start.y()
            if hypot(dx, dy) >= self._scene_tolerance(4.0):
                self._drag_kind = "polygon"
                self._drag_polygon_id = self._select_press_polygon_id
                self._drag_origin_points = self._editor_scene.polygon_points(self._select_press_polygon_id)
                self._drag_polygons_snapshot = self._editor_scene.get_polygons()
                self._drag_start_scene_pos = QPointF(self._select_press_start)
                self._select_press_polygon_id = None
                self._select_press_start = None
        if self._tool == EditorTool.PAN:
            super().mouseMoveEvent(event)
            return
        if self._tool == EditorTool.ADD_POLYGON and (
            self._effective_polygon_create_mode() == PolygonCreateMode.POINTS
            or self._editor_scene.has_pending_polygon()
        ):
            self._editor_scene.update_pending_cursor(scene_pos)
            event.accept()
            return
        if self._drag_kind == "rect_polygon" and self._drag_start_scene_pos is not None:
            self._editor_scene.set_preview_rect(self._drag_start_scene_pos, scene_pos)
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
                if last_point is not None and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                else scene_pos
            )
            self._editor_scene.update_pending_cursor(target)
            event.accept()
            return
        if self._drag_kind == "vertex" and self._drag_polygon_id is not None and self._drag_vertex_index is not None:
            self._editor_scene.preview_vertex_move(self._drag_polygon_id, self._drag_vertex_index, scene_pos)
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
            self._editor_scene.preview_polygon_move(self._drag_polygon_id, moved)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if event.button() == Qt.MouseButton.MiddleButton and self._middle_pan_active:
            self._middle_pan_active = False
            self._middle_pan_last_viewport = None
            self.middlePreviewHoldChanged.emit(False)
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
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton) and self._drag_kind is not None:
            if self._drag_kind == "brush":
                release_pos = self.mapToScene(event.position().toPoint())
                self._commit_brush_drag(release_pos)

            elif self._drag_kind == "rect_polygon" and self._drag_start_scene_pos is not None:
                self._editor_scene.add_rectangle_polygon(
                    self._drag_start_scene_pos,
                    self.mapToScene(event.position().toPoint()),
                    erase=self._drag_erases,
                )
            elif self._drag_kind == "ruler" and self._drag_start_scene_pos is not None:
                release_pos = self.mapToScene(event.position().toPoint())
                target_pos = self._ruler_target(self._drag_start_scene_pos, release_pos, event.modifiers())
                measurement_text = self._format_ruler_measurement(self._drag_start_scene_pos, target_pos)
                self._editor_scene.set_measurement(self._drag_start_scene_pos, target_pos, measurement_text)
                self.rulerMeasurementChanged.emit(measurement_text)
            elif self._drag_kind == "delete_area" and self._drag_start_scene_pos is not None:
                self._editor_scene.delete_vertices_in_rect(
                    QRectF(self._drag_start_scene_pos, self.mapToScene(event.position().toPoint()))
                )
                self._editor_scene.clear_preview_rect()
            elif self._drag_kind == "select_area" and self._drag_start_scene_pos is not None:
                release_pos = self.mapToScene(event.position().toPoint())
                rect = QRectF(self._drag_start_scene_pos, release_pos).normalized()
                self._editor_scene.clear_preview_rect()
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
                new_points = self._editor_scene.polygon_points(self._drag_polygon_id)
                old_point = self._drag_origin_points[self._drag_vertex_index]
                new_point = new_points[self._drag_vertex_index]
                if _points_different(old_point, new_point):
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
                        trial = apply_vertex_position_to_clone(
                            self._drag_polygons_snapshot,
                            self._drag_polygon_id,
                            self._drag_vertex_index,
                            new_point,
                        )
                        profile_timings["clone"] = (perf_counter() - phase_start) * 1000.0
                        phase_start = perf_counter()
                        processed, changed = postprocess_changed_polygon_only(
                            trial,
                            self._vector_geometry_settings,
                            polygon_id=self._drag_polygon_id,
                        )
                        profile_timings["postprocess"] = (perf_counter() - phase_start) * 1000.0
                        if not changed:
                            # Keep direct vertex move if local cleanup produced no effective topology update.
                            processed = trial
                        phase_start = perf_counter()
                        focus_id = resolve_focus_id_after_geometry_pass(
                            self._drag_polygons_snapshot,
                            self._drag_polygon_id,
                            processed,
                        )
                        profile_timings["focus"] = (perf_counter() - phase_start) * 1000.0
                        phase_start = perf_counter()
                        self.undo_stack.push(
                            ReplacePolygonSetCommand(
                                self._editor_scene,
                                self._drag_polygons_snapshot,
                                processed,
                                "Move vertex",
                            )
                        )
                        self._editor_scene.select_polygon(focus_id)
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
                self._drag_kind == "polygon"
                and self._drag_polygon_id is not None
                and self._drag_origin_points is not None
                and self._drag_polygons_snapshot is not None
            ):
                new_points = self._editor_scene.polygon_points(self._drag_polygon_id)
                if _polygon_points_different(self._drag_origin_points, new_points):
                    if not is_valid_closed_polygon_ring(new_points):
                        self._editor_scene.preview_polygon_move(self._drag_polygon_id, self._drag_origin_points)
                        self._editor_scene.warn_invalid_polygon_geometry()
                    else:
                        trial = apply_polygon_points_to_clone(
                            self._drag_polygons_snapshot,
                            self._drag_polygon_id,
                            new_points,
                        )
                        processed, _c = postprocess_changed_polygon_only(
                            trial,
                            self._vector_geometry_settings,
                            polygon_id=self._drag_polygon_id,
                        )
                        if not _c:
                            # Keep direct polygon move if local cleanup produced no effective topology update.
                            processed = trial
                        focus_id = resolve_focus_id_after_geometry_pass(
                            self._drag_polygons_snapshot,
                            self._drag_polygon_id,
                            processed,
                        )
                        self.undo_stack.push(
                            ReplacePolygonSetCommand(
                                self._editor_scene,
                                self._drag_polygons_snapshot,
                                processed,
                                "Move polygon",
                            )
                        )
                        self._editor_scene.select_polygon(focus_id)
            self._drag_kind = None
            self._drag_polygon_id = None
            self._drag_vertex_index = None
            self._drag_origin_points = None
            self._drag_start_scene_pos = None
            self._drag_polygons_snapshot = None
            self._drag_erases = False
            self._brush_pan_guard = False
            self._update_tool_cursors()
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
            self._drag_origin_points = None
            self._drag_start_scene_pos = None
            self._drag_polygons_snapshot = None
            self._drag_erases = False
            self._update_tool_cursors()
            event.accept()
            return
        super().tabletEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            neighbor_path = self._editor_scene.neighbor_frame_path_at(self.mapToScene(event.position().toPoint()))
            if neighbor_path:
                self.neighborFrameActivated.emit(neighbor_path)
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
            event.key() == Qt.Key.Key_Space
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
            and self.isEnabled()
            and (self.hasFocus() or self._require_viewport().hasFocus())
        ):
            if event.isAutoRepeat():
                event.accept()
                return
            if self._polygon_overlays_visible_before_space_hold is None:
                self._polygon_overlays_visible_before_space_hold = self._editor_scene.polygon_overlays_visible()
                self._editor_scene.set_polygon_overlays_visible(False)
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
            self._editor_scene.cancel_pending_polygon()
            self._editor_scene.clear_measurement()
            self._editor_scene.clear_preview_rect()
            self._exit_paste_mode()
            if self._tool == EditorTool.SELECT:
                self._editor_scene.select_polygon(None)
            self._select_press_polygon_id = None
            self._select_press_start = None
            if self._tool == EditorTool.RULER:
                self.rulerMeasurementChanged.emit("")
            self._drag_kind = None
            self._drag_erases = False
            self._pending_polygon_erases = None
            self._update_tool_cursors()
            self._emit_effective_polygon_create_mode_changed()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Delete:
            self._editor_scene.delete_polygon()
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

    def keyReleaseEvent(self, event: QKeyEvent | None) -> None:
        if event is None:
            return
        if (
            event.key() == Qt.Key.Key_Space
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
            and not event.isAutoRepeat()
            and self._polygon_overlays_visible_before_space_hold is not None
        ):
            self._editor_scene.set_polygon_overlays_visible(self._polygon_overlays_visible_before_space_hold)
            self._polygon_overlays_visible_before_space_hold = None
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

    def _advance_zoom_animation(self) -> None:
        viewport_pixel = self._zoom_animation_viewport_pixel
        if viewport_pixel is None:
            self._finish_zoom_animation()
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
            self._finish_zoom_animation()

    def _finish_zoom_animation(self) -> None:
        self._zoom_animation_timer.stop()
        self._zoom_animation_viewport_pixel = None
        self._leave_zoom_render_mode()
        self._update_navigation_scene_rect()
        self._schedule_pyramid_visible_update()
        self._update_tool_cursors()

    def _enter_zoom_render_mode(self) -> None:
        self.setRenderHints(self._zooming_render_hints)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)

    def _leave_zoom_render_mode(self) -> None:
        self.setRenderHints(self._steady_render_hints)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, False)

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

    def leaveEvent(self, event: QEvent | None) -> None:
        if event is None:
            return
        self._editor_scene.clear_conductor_hover_highlight()
        self._editor_scene.clear_vertex_preview()
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
            return
        zoom = self.zoom_factor() if zoom is None else max(1e-6, float(zoom))
        viewport_rect = self._require_viewport().rect()
        margin_x = float(viewport_rect.width()) / max(zoom, 1e-6) + 2.0
        margin_y = float(viewport_rect.height()) / max(zoom, 1e-6) + 2.0
        self.setSceneRect(base_rect.adjusted(-margin_x, -margin_y, margin_x, margin_y))

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

    def _cycle_active_tool_mode(self) -> bool:
        if self._tool == EditorTool.ADD_POLYGON:
            if self._editor_scene.has_pending_polygon():
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
        if snap and last_point is not None:
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
        print(message)
        self.logRequested.emit(message)
        if profiler is None:
            return
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        top_lines = vertex_move_top_lines()
        stats.print_stats(top_lines)
        report = stream.getvalue()
        print(f"[contour vertex profiling stats] top={top_lines}")
        print(report)
        self.logRequested.emit(f"[contour vertex profiling stats] top={top_lines}\n{report}")
