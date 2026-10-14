"""matrix_view: construction."""
from __future__ import annotations
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)
from ...core.domain import FrameRecord
from ...core.domain import ComparisonMode
from ...core.subpixel_grid import (
    SubpixelGrid,
    SubpixelGridSpec,
)
from neuralimage.Validation_gradient_widget_lite.ui.i18n import Translator
from neuralimage.Validation_gradient_widget_lite.ui.ui_constants import (
    DEFAULT_CELL_SIZE,
    DEFAULT_ERROR_WINDOW,
    DEFAULT_GRADIENT_NAME,
    MATRIX_BACKGROUND,
    MATRIX_CELL_GAP,
    MATRIX_SCENE_PADDING,
    SELECTION_BLINK_INTERVAL_MS,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.module_state import (
    SUBPIXEL_GRID_WORKER_COUNT,
    SUBPIXEL_VISIBILITY_THRESHOLD,
    TILE_HOVER_PREFETCH_MS,
    TILE_VIEWPORT_DEBOUNCE_MS,
    VIEW_LOD_PIXEL,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.score_fill_color import MatrixLayoutConfig


class ConstructionMixin:
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._i18n = Translator()
        self._t = self._i18n.tr
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._records: list[FrameRecord] = []
        self._cell_size = DEFAULT_CELL_SIZE
        self._gap = MATRIX_CELL_GAP
        self._scene_padding = MATRIX_SCENE_PADDING
        self._selected_item: _MatrixCellItem | None = None
        self._hovered_item: _MatrixCellItem | None = None
        self._selected_subpixel_selection: MatrixTileSelection | None = None
        self._hovered_subpixel_selection: MatrixTileSelection | None = None
        self._processing_keys: set[str] = set()
        self._reference_key: str | None = None
        self._columns = 0
        self._rows = 0
        self._gradient_name = DEFAULT_GRADIENT_NAME
        self._error_window_low, self._error_window_high = DEFAULT_ERROR_WINDOW
        self._auto_color_window_low, self._auto_color_window_high = DEFAULT_ERROR_WINDOW
        self._score_view_mode = "relative"
        self._metric_key: str | None = None
        self._point_match_radius = 3.0
        self._bce_score_cap = 1.0
        self._overview_image: QImage | None = None
        self._selection_blink_on = False
        self._selection_blink_timer = QTimer(self)
        self._selection_blink_timer.setInterval(SELECTION_BLINK_INTERVAL_MS)
        self._selection_blink_timer.timeout.connect(self._toggle_selection_blink)
        self._selection_blink_timer.start()
        self._layout_config = MatrixLayoutConfig()
        self._subpixel_spec: SubpixelGridSpec | None = None
        self._subpixel_aggregation = "mean"
        self._subpixel_comparison_mode = ComparisonMode.DISAGREEMENT
        self._subpixel_grid_cache: OrderedDict[tuple[object, ...], SubpixelGrid] = OrderedDict()
        self._item_by_key: dict[str, _MatrixCellItem] = {}
        self._record_by_position: dict[tuple[int, int], FrameRecord] = {}
        self._record_positions: dict[str, tuple[int, int]] = {}
        self._record_index_by_key: dict[str, int] = {}
        self._overview_layer_item: QGraphicsPixmapItem | None = None
        self._matrix_frame_item: QGraphicsRectItem | None = None
        self._virtualized_items_enabled = False
        self._tile_zoom_threshold = SUBPIXEL_VISIBILITY_THRESHOLD
        self._tile_overlay_visible = False
        self._overview_layer_visible = False
        self._active_lod_band = VIEW_LOD_PIXEL
        self._tile_request_generation = 0
        self._tile_load_generation: int | None = None
        self._pending_tile_keys: deque[str] = deque()
        self._pending_tile_key_set: set[str] = set()
        self._visible_record_key_cache_signature: tuple[object, ...] | None = None
        self._visible_record_key_cache: tuple[str, ...] = tuple()
        self._focus_scene_pos: QPointF | None = None
        self._hover_scene_pos: QPointF | None = None
        self._zoom_anchor_scene_pos: QPointF | None = None
        self._last_viewport_center_scene_pos: QPointF | None = None
        self._pan_bias: tuple[float, float] = (0.0, 0.0)
        self._tile_viewport_timer = QTimer(self)
        self._tile_viewport_timer.setSingleShot(True)
        self._tile_viewport_timer.setInterval(TILE_VIEWPORT_DEBOUNCE_MS)
        self._tile_viewport_timer.timeout.connect(self._prepare_visible_tile_queue)
        self._hover_prefetch_timer = QTimer(self)
        self._hover_prefetch_timer.setSingleShot(True)
        self._hover_prefetch_timer.setInterval(TILE_HOVER_PREFETCH_MS)
        self._hover_prefetch_timer.timeout.connect(self._on_hover_prefetch_timeout)
        self._tile_load_timer = QTimer(self)
        self._tile_load_timer.setSingleShot(True)
        self._tile_load_timer.timeout.connect(self._process_pending_tile_queue)
        self._tile_result_timer = QTimer(self)
        self._tile_result_timer.setInterval(30)
        self._tile_result_timer.timeout.connect(self._poll_tile_futures)
        self._tile_executor = ThreadPoolExecutor(max_workers=max(1, int(SUBPIXEL_GRID_WORKER_COUNT)), thread_name_prefix="matrix-tile")
        self._tile_futures: dict[str, tuple[int, tuple[object, ...], Future]] = {}
        self._tile_cache_hits = 0
        self._tile_disk_cache_hits = 0
        self._tile_cache_misses = 0
        self._tile_jobs_submitted = 0
        self._tile_jobs_completed = 0
        self._tile_jobs_discarded = 0
        self._pan_active = False
        self._pan_start = None
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundBrush(MATRIX_BACKGROUND)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.viewport().setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.horizontalScrollBar().valueChanged.connect(self._on_viewport_scroll_changed)
        self.verticalScrollBar().valueChanged.connect(self._on_viewport_scroll_changed)
