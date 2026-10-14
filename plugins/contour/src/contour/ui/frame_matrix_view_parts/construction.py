"""frame_matrix_view: construction."""
from __future__ import annotations
from PyQt6.QtCore import QPoint, QSize, Qt, QThreadPool, QTimer
from PyQt6.QtGui import QBrush, QColor, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QListWidgetItem,
    QWidget,
)
from ...application.frame_lod import PyramidFrameStore

from contour.ui.frame_matrix_view_parts.module_state import _MATRIX_ZOOM_FRAME_MS


class ConstructionMixin:
    def __init__(self, parent: QWidget | None = None) -> None:
        self._scene = QGraphicsScene()
        super().__init__(self._scene, parent)
        self._items: list[QListWidgetItem] = []
        self._item_groups: dict[
            int,
            tuple[QGraphicsRectItem, QGraphicsRectItem, QGraphicsRectItem, QGraphicsPixmapItem, QGraphicsSimpleTextItem],
        ] = {}
        self._icon_size = QSize(64, 64)
        self._grid_size = QSize(64, 64)
        self._spacing = 0
        self._columns = 1
        self._current_row = -1
        self._signals_blocked = False
        self._middle_pan_active = False
        self._middle_pan_origin = QPoint()
        self._middle_pan_scroll_origin = QPoint()
        self._layout_dirty = False
        self._matrix_zoom = 1.0
        self._matrix_target_zoom = 1.0
        self._matrix_zoom_anchor: QPoint | None = None
        self._matrix_zoom_timer = QTimer(self)
        self._matrix_zoom_timer.setInterval(_MATRIX_ZOOM_FRAME_MS)
        self._matrix_zoom_timer.timeout.connect(self._advance_zoom_animation)
        self._layout_width = 1
        self._layout_height = 1
        self._overlap_pixels_x = 0
        self._overlap_pixels_y = 0
        self._emitted_thumbnail_source_size = QSize()
        self._pixmap_lod_cache: dict[tuple, QPixmap] = {}
        self._pyramid_thumbnail_generation = 0
        self._pyramid_thumbnail_pending: set[tuple[int, int, int, int]] = set()
        self._pyramid_thumbnail_thread_pool = QThreadPool(self)
        self._pyramid_thumbnail_thread_pool.setMaxThreadCount(2)
        self._pyramid_thumbnail_thread_pool.setExpiryTimeout(30000)
        self._pyramid_store: PyramidFrameStore | None = None
        self._navigator_lods: tuple[int, ...] = ()
        self._visible_index_range: tuple[int, int] = (0, -1)
        self._render_region_sync_allowed = True
        self._suppress_matrix_refresh = False
        self._render_region_timer = QTimer(self)
        self._render_region_timer.setSingleShot(True)
        self._render_region_timer.timeout.connect(self._sync_render_region)

        self._opengl_viewport_enabled = self._configure_opengl_viewport()
        self._steady_render_hints = QPainter.RenderHint.SmoothPixmapTransform
        self._zooming_render_hints = QPainter.RenderHint(0)
        self.setRenderHints(self._steady_render_hints)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setBackgroundBrush(QBrush(QColor("#111827")))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, False)
