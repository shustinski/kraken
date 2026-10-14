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
    write_profile_report,
)
from ..infrastructure.runtime_config import config_bool, config_float, config_int
from .editor_hotkeys import event_matches_shortcut, shortcut_sequence, tool_shortcut_sequence
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

from .editor_view_parts.module_state import (
    _WHEEL_ZOOM_COALESCE_MS,
    _ZOOM_ANIMATION_FRAME_MS,
    _ZOOM_EASING_FRACTION,
    _ZOOM_SETTLE_RATIO,
    _ZOOM_VECTOR_BATCH_THRESHOLD,
    _MINIMAP_DRAWING_TOOLS,
    _OPENGL_VIEWPORT_ENABLED,
    _OPENGL_DISABLED_PLATFORMS,
    _PYRAMID_VISIBLE_UPDATE_MS,
    _PYRAMID_CACHE_LIMIT,
)
from .editor_view_parts.construction import ConstructionMixin
from .editor_view_parts.configure_opengl_viewport import ConfigureOpenglViewportMixin
from .editor_view_parts.tool import ToolMixin
from .editor_view_parts.polygons import PolygonsMixin
from .editor_view_parts.pyramid_frame_store import PyramidFrameStoreMixin
from .editor_view_parts.refresh_pyramid_visible_frames import RefreshPyramidVisibleFramesMixin
from .editor_view_parts.polygon_overlay_hide_hold import PolygonOverlayHideHoldMixin
from .editor_view_parts.wheel_event import WheelEventMixin
from .editor_view_parts.mouse_press_event import MousePressEventMixin
from .editor_view_parts.mouse_move_event import MouseMoveEventMixin
from .editor_view_parts.mouse_release_event import MouseReleaseEventMixin
from .editor_view_parts.mouse_double_click_event import MouseDoubleClickEventMixin
from .editor_view_parts.key_press_event import KeyPressEventMixin
from .editor_view_parts.key_release_event import KeyReleaseEventMixin
from .editor_view_parts.apply_zoom_at_viewport import ApplyZoomAtViewportMixin
from .editor_view_parts.cycle_active_tool_mode import CycleActiveToolModeMixin
from .editor_view_parts.commit_delete_vertices_in import CommitDeleteVerticesInMixin

class PolygonEditorView(
    ConstructionMixin,
    ConfigureOpenglViewportMixin,
    ToolMixin,
    PolygonsMixin,
    PyramidFrameStoreMixin,
    RefreshPyramidVisibleFramesMixin,
    PolygonOverlayHideHoldMixin,
    WheelEventMixin,
    MousePressEventMixin,
    MouseMoveEventMixin,
    MouseReleaseEventMixin,
    MouseDoubleClickEventMixin,
    KeyPressEventMixin,
    KeyReleaseEventMixin,
    ApplyZoomAtViewportMixin,
    CycleActiveToolModeMixin,
    CommitDeleteVerticesInMixin,
    QGraphicsView,
):
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

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _WHEEL_ZOOM_COALESCE_MS,
    _ZOOM_ANIMATION_FRAME_MS,
    _ZOOM_EASING_FRACTION,
    _ZOOM_SETTLE_RATIO,
    _ZOOM_VECTOR_BATCH_THRESHOLD,
    _MINIMAP_DRAWING_TOOLS,
    _OPENGL_VIEWPORT_ENABLED,
    _OPENGL_DISABLED_PLATFORMS,
    _PYRAMID_VISIBLE_UPDATE_MS,
    _PYRAMID_CACHE_LIMIT,
)
