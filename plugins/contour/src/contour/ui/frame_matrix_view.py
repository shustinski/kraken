from __future__ import annotations

from pathlib import Path
import re

from PyQt6.QtCore import QPoint, QRect, QRectF, QSize, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QGuiApplication, QPainter, QPen, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QListWidget,
    QListWidgetItem,
    QWidget,
)

from ..adapters.qt.pyramid import PyramidThumbnailLoadRunnable
from ..application.frame_lod import PyramidFrameStore
from ..graphics.viewport_navigation import (
    DEFAULT_ZOOM_STEP_FACTOR,
    MAX_ZOOM_FACTOR,
    MIN_ZOOM_FACTOR,
    clamp_zoom_factor,
    viewport_scroll_correction_after_scale_reanchor,
    zoom_factor_for_wheel_delta,
)
from .item_status_painting import FRAME_STATUS_ROLE
from .large_dataset import clamp_thumbnail_source_size

try:
    from shiboken6 import isValid as _shiboken_is_valid
except ImportError:
    _shiboken_is_valid = None

_OPENGL_VIEWPORT_ENABLED = True
_OPENGL_DISABLED_PLATFORMS = {"offscreen", "minimal"}
_MATRIX_ZOOM_FRAME_MS = 16
_MATRIX_ZOOM_EASING_FRACTION = 0.55
_MATRIX_ZOOM_SETTLE_RATIO = 0.001
_LOD_LABEL_MIN_ZOOM = 0.75
_LOD_STATUS_MIN_ZOOM = 0.22
_THUMBNAIL_LOD_LEVELS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
_THUMBNAIL_PIXMAP_ROLE = int(Qt.ItemDataRole.UserRole) + 1001
_FRAME_ID_ROLE = int(Qt.ItemDataRole.UserRole) + 1002
_RENDER_REGION_BUFFER_CELLS = 2
# Keep full rows around the current frame visible (matches thumbnail load window).
_FOCUS_NEIGHBOR_ROW_BUFFER = 0
_STATUS_MARKER_COLORS = {
    "viewed": QColor("#475569"),
    "modified": QColor("#B45309"),
    "saved": QColor("#047857"),
    "no_vector": QColor("#B91C1C"),
}


class FrameMatrixGraphicsView(QGraphicsView):
    """QGraphicsView-backed frame matrix with the QListWidget subset Contour uses."""

    itemClicked = pyqtSignal(QListWidgetItem)
    thumbnailLodChanged = pyqtSignal(int, int)
    frameNavigationRequested = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        self._scene = QGraphicsScene()
        super().__init__(self._scene, parent)
        self._items: list[QListWidgetItem] = []
        self._item_groups: dict[
            int,
            tuple[QGraphicsRectItem, QGraphicsRectItem, QGraphicsRectItem, QGraphicsPixmapItem, QGraphicsSimpleTextItem],
        ] = {}
        self._icon_size = QSize(64, 48)
        self._grid_size = QSize(64, 48)
        self._spacing = 0
        self._columns = 1
        self._current_row = -1
        self._signals_blocked = False
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

    def widget(self) -> "FrameMatrixGraphicsView":
        return self

    def setWidgetResizable(self, _enabled: bool) -> None:
        return

    def setViewMode(self, _mode: QListWidget.ViewMode) -> None:
        return

    def setResizeMode(self, _mode: QListWidget.ResizeMode) -> None:
        return

    def setMovement(self, _movement: QListWidget.Movement) -> None:
        return

    def setSelectionMode(self, _mode: QAbstractItemView.SelectionMode) -> None:
        return

    def setUniformItemSizes(self, _enabled: bool) -> None:
        return

    def setWrapping(self, _enabled: bool) -> None:
        return

    def setIconSize(self, size: QSize) -> None:
        if QSize(size) != self._icon_size:
            self._pyramid_thumbnail_generation += 1
            self._pyramid_thumbnail_pending.clear()
        self._icon_size = QSize(size)
        self._mark_layout_dirty()

    def setGridSize(self, size: QSize) -> None:
        if QSize(size) != self._grid_size:
            self._pyramid_thumbnail_generation += 1
            self._pyramid_thumbnail_pending.clear()
            self._pixmap_lod_cache.clear()
        self._grid_size = QSize(size)
        self._mark_layout_dirty()

    def setSpacing(self, spacing: int) -> None:
        self._spacing = max(0, int(spacing))
        self._mark_layout_dirty()

    def setFrameOverlapPixels(self, pixels: int, pixels_y: int | None = None) -> None:
        self._overlap_pixels_x = max(0, int(pixels))
        self._overlap_pixels_y = self._overlap_pixels_x if pixels_y is None else max(0, int(pixels_y))
        self._mark_layout_dirty()

    def frameWidth(self) -> int:
        return 0

    def count(self) -> int:
        return len(self._items)

    def item(self, index: int) -> QListWidgetItem | None:
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def addItem(self, item: QListWidgetItem) -> None:
        self._items.append(item)
        self._mark_layout_dirty()

    def clear(self) -> None:
        self._render_region_timer.stop()
        self._render_region_sync_allowed = False
        self._pyramid_thumbnail_generation += 1
        self._pyramid_thumbnail_pending.clear()
        try:
            self._pyramid_thumbnail_thread_pool.clear()
        except RuntimeError:
            pass
        self._items.clear()
        self._item_groups.clear()
        self._current_row = -1
        self._scene.clear()
        self._scene.setSceneRect(QRectF())
        self._layout_dirty = False
        self._pixmap_lod_cache.clear()
        self._visible_index_range = (0, -1)

    def shutdownPyramidLoading(self) -> None:
        self._pyramid_thumbnail_generation += 1
        self._pyramid_thumbnail_pending.clear()
        try:
            self._pyramid_thumbnail_thread_pool.clear()
            self._pyramid_thumbnail_thread_pool.waitForDone(3000)
        except RuntimeError:
            pass

    @staticmethod
    def _graphics_item_is_valid(graphics_item: object) -> bool:
        if _shiboken_is_valid is None:
            try:
                graphics_item.scene()
                return True
            except RuntimeError:
                return False
        return bool(_shiboken_is_valid(graphics_item))

    def _group_is_valid(self, group: tuple) -> bool:
        return all(self._graphics_item_is_valid(graphics_item) for graphics_item in group)

    def _attach_graphics_group_for_item(self, item: QListWidgetItem, index: int) -> None:
        background_item = QGraphicsRectItem()
        border_pen = QPen(QColor("#263241"), 1.0)
        border_pen.setCosmetic(True)
        background_item.setPen(border_pen)
        background_item.setBrush(QBrush(QColor("#111827")))
        status_item = QGraphicsRectItem()
        status_item.setPen(QPen(Qt.PenStyle.NoPen))
        status_item.hide()
        selection_item = QGraphicsRectItem()
        selection_pen = QPen(QColor("#60A5FA"), 2.0)
        selection_pen.setCosmetic(True)
        selection_item.setPen(selection_pen)
        selection_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        selection_item.hide()
        pixmap_item = QGraphicsPixmapItem()
        text_item = QGraphicsSimpleTextItem()
        text_item.setBrush(QBrush(QColor("#E5E7EB")))
        for z_value, graphics_item in enumerate((background_item, status_item, pixmap_item, text_item, selection_item)):
            graphics_item.setData(0, index)
            graphics_item.setZValue(float(z_value))
            self._scene.addItem(graphics_item)
        self._item_groups[id(item)] = (background_item, status_item, selection_item, pixmap_item, text_item)

    def _item_group_for(self, item: QListWidgetItem) -> tuple | None:
        group = self._item_groups.get(id(item))
        if group is not None and self._group_is_valid(group):
            return group
        if group is not None:
            self._item_groups.pop(id(item), None)
        try:
            index = self._items.index(item)
        except ValueError:
            return None
        self._attach_graphics_group_for_item(item, index)
        group = self._item_groups.get(id(item))
        return group if group is not None and self._group_is_valid(group) else None

    def currentRow(self) -> int:
        return self._current_row

    def currentItem(self) -> QListWidgetItem | None:
        return self.item(self._current_row)

    def setCurrentRow(self, row: int) -> None:
        new_row = row if 0 <= int(row) < len(self._items) else -1
        if new_row == self._current_row and not self._layout_dirty:
            return
        previous_row = self._current_row
        self._current_row = new_row
        if self._layout_dirty or not self._item_groups:
            self._refresh_items()
            return
        self._update_selection_rows(previous_row, new_row)
        self._sync_render_region()

    def clearSelection(self) -> None:
        if self._current_row == -1 and not self._layout_dirty:
            return
        previous_row = self._current_row
        self._current_row = -1
        if self._layout_dirty or not self._item_groups:
            self._refresh_items()
            return
        self._update_selection_rows(previous_row, -1)
        self._sync_render_region()

    def blockSignals(self, block: bool) -> bool:  # type: ignore[override]
        previous = self._signals_blocked
        self._signals_blocked = bool(block)
        super().blockSignals(block)
        return previous

    def setUpdatesEnabled(self, enabled: bool) -> None:  # type: ignore[override]
        super().setUpdatesEnabled(enabled)
        if enabled and not getattr(self, "_suppress_matrix_refresh", False):
            self._refresh_items()

    def setFixedSize(self, width: int, height: int) -> None:  # type: ignore[override]
        self._set_layout_extent(width, height)
        self.setMinimumSize(1, 1)
        self.setMaximumSize(16777215, 16777215)

    def minimumWidth(self) -> int:
        return self._layout_width

    def minimumHeight(self) -> int:
        return self._layout_height

    def width(self) -> int:
        return self._layout_width

    def height(self) -> int:
        return self._layout_height

    def doItemsLayout(self) -> None:
        self._refresh_items()

    def refreshItems(self) -> None:
        self._pixmap_lod_cache.clear()
        self._refresh_items()

    def refreshVisibleRegion(self) -> None:
        """Update only matrix cells intersecting the viewport (render region)."""
        self._sync_render_region()

    def refreshThumbnailIndexes(self, indexes: list[int]) -> None:
        """Repaint matrix cells for rows that received new thumbnail icons."""
        self.updateThumbnailPixmaps(indexes)

    def setPyramidFrameStore(self, store: PyramidFrameStore | None) -> None:
        self._pyramid_thumbnail_generation += 1
        self._pyramid_thumbnail_pending.clear()
        try:
            self._pyramid_thumbnail_thread_pool.clear()
        except RuntimeError:
            pass
        self._pyramid_store = store if store is not None and store.has_zarr() else None
        if self._pyramid_store is None:
            self._navigator_lods = ()
        else:
            lods = tuple(sorted(int(lod) for lod in self._pyramid_store.available_lods()))
            self._navigator_lods = lods[-3:]
        self._pixmap_lod_cache.clear()
        self.refreshVisibleRegion()

    def navigatorLods(self) -> tuple[int, ...]:
        return tuple(self._navigator_lods)

    def setCurrentFrameId(self, frame_id: int | None) -> None:
        if frame_id is None:
            self.clearSelection()
            return
        self.setCurrentRow(int(frame_id))

    def updateThumbnailPixmaps(self, indexes: list[int]) -> None:
        """Lightweight pixmap refresh for visible cells (avoids full cell relayout)."""
        if not self._items or not indexes:
            return
        region = self._effective_render_index_range()
        columns = max(1, self._columns)
        step_x = self._cell_step_x()
        step_y = self._cell_step_y()
        for index in sorted({int(value) for value in indexes if 0 <= int(value) < len(self._items)}):
            if not self._index_in_render_region(index, region):
                continue
            item = self._items[index]
            if item.isHidden():
                continue
            group = self._item_group_for(item)
            if group is None:
                continue
            _background_item, _status_item, _selection_item, pixmap_item, _text_item = group
            pixmap = self._pixmap_for_item_lod(item)
            if pixmap.isNull():
                continue
            pixmap_item.setPixmap(pixmap)
            col = index % columns
            row = index // columns
            self._position_pixmap_item(pixmap_item, pixmap, col * step_x, row * step_y)
            pixmap_item.setVisible(True)

    def _schedule_render_region_sync(self, *_args) -> None:
        if not self.updatesEnabled() or self._layout_dirty or not self._render_region_sync_allowed:
            return
        if getattr(self, "_suppress_matrix_refresh", False):
            return
        self._render_region_timer.stop()
        self._render_region_timer.start(32)

    def visualItemRect(self, item: QListWidgetItem) -> QRect:
        try:
            index = self._items.index(item)
        except ValueError:
            return QRect()
        col = index % max(1, self._columns)
        row = index // max(1, self._columns)
        return QRect(col * self._cell_step_x(), row * self._cell_step_y(), self._grid_size.width(), self._grid_size.height())

    def ensureVisible(self, x: int, y: int, w: int = 0, h: int = 0, xmargin: int = 0, ymargin: int = 0) -> None:  # type: ignore[override]
        super().ensureVisible(float(x), float(y), float(w), float(h), xmargin, ymargin)

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is None:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta:
                self._start_zoom_animation(event.position().toPoint(), zoom_factor_for_wheel_delta(delta))
            event.accept()
            return
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            delta = event.angleDelta()
            delta_value = delta.x() if delta.x() else delta.y()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta_value)
            event.accept()
            return
        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            delta = event.angleDelta().y()
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        item = self._item_at_viewport_pos(event.position().toPoint())
        if item is not None:
            try:
                self.setCurrentRow(self._items.index(item))
            except ValueError:
                pass
            if not self._signals_blocked:
                self.itemClicked.emit(item)
                self.frameNavigationRequested.emit(self._frame_id_for_item(item))
            event.accept()
            return
        super().mousePressEvent(event)

    def _item_at_viewport_pos(self, point: QPoint) -> QListWidgetItem | None:
        scene_pos = self.mapToScene(point)
        col = int(scene_pos.x() // self._cell_step_x())
        row = int(scene_pos.y() // self._cell_step_y())
        index = row * max(1, self._columns) + col
        item = self.item(index)
        if item is None or item.isHidden():
            return None
        return item

    def _apply_zoom_at_viewport_pixel(self, viewport_pixel: QPoint, factor: float) -> None:
        old_zoom = self._matrix_zoom
        new_zoom = clamp_zoom_factor(old_zoom * float(factor))
        if abs(new_zoom - old_zoom) <= 1e-9:
            return
        factor = new_zoom / old_zoom
        view_point = self.viewport().mapTo(self, viewport_pixel)
        scene_anchor = self.mapToScene(view_point)
        self.scale(factor, factor)
        self._matrix_zoom = new_zoom
        mapped = self.viewport().mapFrom(self, self.mapFromScene(scene_anchor))
        dh, dv = viewport_scroll_correction_after_scale_reanchor((viewport_pixel.x(), viewport_pixel.y()), (mapped.x(), mapped.y()))
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + dh)
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + dv)
        self._emit_thumbnail_lod_if_needed()
        region = self._effective_render_index_range()
        first_index, last_index = region
        if last_index >= first_index:
            self._apply_lod_for_indexes(range(first_index, last_index + 1), region=region)

    def _start_zoom_animation(self, viewport_pixel: QPoint, factor: float) -> None:
        base_zoom = self._matrix_target_zoom if self._matrix_zoom_timer.isActive() else self._matrix_zoom
        target_zoom = clamp_zoom_factor(base_zoom * float(factor))
        if abs(target_zoom - self._matrix_zoom) <= 1e-9:
            return
        self._matrix_target_zoom = target_zoom
        self._matrix_zoom_anchor = QPoint(viewport_pixel)
        if not self._matrix_zoom_timer.isActive():
            self._enter_zoom_render_mode()
            self._matrix_zoom_timer.start()

    def _advance_zoom_animation(self) -> None:
        anchor = self._matrix_zoom_anchor
        if anchor is None:
            self._finish_zoom_animation()
            return
        remaining = self._matrix_target_zoom - self._matrix_zoom
        if abs(remaining) <= max(_MATRIX_ZOOM_SETTLE_RATIO, abs(self._matrix_target_zoom) * _MATRIX_ZOOM_SETTLE_RATIO):
            next_zoom = self._matrix_target_zoom
            finish = True
        else:
            next_zoom = self._matrix_zoom + remaining * _MATRIX_ZOOM_EASING_FRACTION
            finish = False
        self._apply_zoom_at_viewport_pixel(anchor, next_zoom / self._matrix_zoom)
        if finish:
            self._finish_zoom_animation()

    def _finish_zoom_animation(self) -> None:
        self._matrix_zoom_timer.stop()
        self._matrix_zoom_anchor = None
        self._leave_zoom_render_mode()
        self._schedule_render_region_sync()

    def _enter_zoom_render_mode(self) -> None:
        self.setRenderHints(self._zooming_render_hints)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)

    def _leave_zoom_render_mode(self) -> None:
        self.setRenderHints(self._steady_render_hints)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, False)

    def _mark_layout_dirty(self) -> None:
        self._layout_dirty = True
        if self.updatesEnabled() and not getattr(self, "_suppress_matrix_refresh", False):
            self._refresh_items()

    def _set_layout_extent(self, width: int, height: int) -> None:
        self._layout_width = max(1, int(width))
        self._layout_height = max(1, int(height))
        step_x = self._cell_step_x()
        overlap_x = min(self._overlap_pixels_x, max(0, max(1, self._grid_size.width()) - 1))
        self._columns = max(1, int((self._layout_width - overlap_x - 1) // step_x))
        self._scene.setSceneRect(QRectF(0, 0, self._layout_width, self._layout_height))

    def _update_selection_rows(self, previous_row: int, new_row: int) -> None:
        region = self._effective_render_index_range()
        for index in (previous_row, new_row):
            if index < 0 or index >= len(self._items):
                continue
            item = self._items[index]
            group = self._item_group_for(item)
            if group is None:
                continue
            self._layout_item_geometry(index, item, group, region=region)

    def _visible_item_index_range(self) -> tuple[int, int]:
        if not self._items:
            return (0, -1)
        columns = max(1, self._columns)
        step_x = max(1, self._cell_step_x())
        step_y = max(1, self._cell_step_y())
        viewport_rect = self.viewport().rect()
        top_left = self.mapToScene(viewport_rect.topLeft())
        bottom_right = self.mapToScene(viewport_rect.bottomRight())
        min_x = min(top_left.x(), bottom_right.x())
        max_x = max(top_left.x(), bottom_right.x())
        min_y = min(top_left.y(), bottom_right.y())
        max_y = max(top_left.y(), bottom_right.y())
        buffer = max(0, int(_RENDER_REGION_BUFFER_CELLS))
        first_row = max(0, int(min_y // step_y) - buffer)
        last_row = max(first_row, int(max_y // step_y) + buffer)
        first_col = max(0, int(min_x // step_x) - buffer)
        last_col = max(first_col, int(max_x // step_x) + buffer)
        first_index = first_row * columns + first_col
        last_index = min(len(self._items) - 1, (last_row + 1) * columns + last_col - 1)
        return first_index, last_index

    def _effective_render_index_range(self) -> tuple[int, int]:
        """Viewport range unioned with full rows around the current frame (neighbor band)."""

        first_index, last_index = self._visible_item_index_range()
        if not self._items:
            return first_index, last_index
        columns = max(1, self._columns)
        if 0 <= self._current_row < len(self._items):
            center_row = self._current_row // columns
            buffer = max(0, int(_FOCUS_NEIGHBOR_ROW_BUFFER))
            for row in range(max(0, center_row - buffer), center_row + buffer + 1):
                row_start = row * columns
                row_end = min(len(self._items) - 1, row_start + columns - 1)
                if last_index < first_index:
                    first_index, last_index = row_start, row_end
                else:
                    first_index = min(first_index, row_start)
                    last_index = max(last_index, row_end)
        return first_index, last_index

    def _index_in_render_region(self, index: int, region: tuple[int, int]) -> bool:
        first_index, last_index = region
        return last_index >= first_index and first_index <= index <= last_index

    def _layout_item_geometry(
        self,
        index: int,
        item: QListWidgetItem,
        group: tuple,
        *,
        region: tuple[int, int],
    ) -> None:
        background_item, status_item, selection_item, pixmap_item, text_item = group
        hidden = item.isHidden()
        in_region = self._index_in_render_region(index, region)
        for graphics_item in group:
            graphics_item.setVisible(not hidden and in_region)
        if hidden or not in_region:
            pixmap_item.setPixmap(QPixmap())
            return
        columns = max(1, self._columns)
        cell_w = max(1, self._grid_size.width())
        cell_h = max(1, self._grid_size.height())
        step_x = self._cell_step_x()
        step_y = self._cell_step_y()
        col = index % columns
        row = index // columns
        x = col * step_x
        y = row * step_y
        selected = index == self._current_row
        background_item.setRect(QRectF(x + 0.5, y + 0.5, max(1, cell_w - 1), max(1, cell_h - 1)))
        background_item.setBrush(QBrush(QColor("#111827")))
        status_color = _STATUS_MARKER_COLORS.get(str(item.data(FRAME_STATUS_ROLE) or ""))
        if status_color is not None:
            marker_size = max(4, min(8, cell_w // 8, cell_h // 6))
            status_item.setRect(QRectF(x + 2, y + 2, marker_size, marker_size))
            status_item.setBrush(QBrush(status_color))
            status_item.setData(1, True)
            status_item.show()
        else:
            status_item.setData(1, False)
            status_item.hide()
        selection_item.setRect(QRectF(x + 1, y + 1, max(1, cell_w - 2), max(1, cell_h - 2)))
        selection_item.setVisible(selected)
        pixmap = self._pixmap_for_item_lod(item)
        pixmap_item.setPixmap(pixmap)
        self._position_pixmap_item(pixmap_item, pixmap, x, y)
        text = item.text()
        if not text:
            tip = item.toolTip()
            text = Path(str(tip)).stem if tip else ""
        text = _trailing_frame_number(text)
        text_item.setText(text)
        text_item.setVisible(bool(text) and self._matrix_zoom >= _LOD_LABEL_MIN_ZOOM)
        text_item.setPos(x + 3, y + max(0, cell_h - 15))

    def _refresh_items(self) -> None:
        if not self._items:
            return
        region = self._effective_render_index_range()
        for group in list(self._item_groups.values()):
            if not self._group_is_valid(group):
                continue
            for graphics_item in group:
                graphics_item.setVisible(False)
            group[3].setPixmap(QPixmap())
        first_index, last_index = region
        if last_index >= first_index:
            for index in range(max(0, first_index), min(len(self._items) - 1, last_index) + 1):
                item = self._items[index]
                group = self._item_group_for(item)
                if group is None:
                    continue
                if not self._index_in_render_region(index, region):
                    continue
                self._layout_item_geometry(index, item, group, region=region)
        self._layout_dirty = False
        self._visible_index_range = region
        self._render_region_sync_allowed = bool(self._items)
        self._prune_pixmap_lod_cache(region)

    def _sync_render_region(self) -> None:
        if not self._items or self._layout_dirty or not self._render_region_sync_allowed:
            if self._layout_dirty and self._items:
                self._refresh_items()
            return
        region = self._effective_render_index_range()
        if region == self._visible_index_range:
            return
        previous_region = self._visible_index_range
        self._visible_index_range = region
        affected: set[int] = set()
        first_index, last_index = region
        prev_first, prev_last = previous_region
        if last_index >= first_index:
            affected.update(range(first_index, last_index + 1))
        if prev_last >= prev_first:
            affected.update(range(prev_first, prev_last + 1))
        for index in sorted(affected):
            if index < 0 or index >= len(self._items):
                continue
            item = self._items[index]
            group = self._item_group_for(item)
            if group is None:
                continue
            self._layout_item_geometry(index, item, group, region=region)
        self._prune_pixmap_lod_cache(region)

    def _apply_lod(self, *, region: tuple[int, int] | None = None) -> None:
        self._emit_thumbnail_lod_if_needed()
        if region is None:
            region = self._effective_render_index_range()
            self._visible_index_range = region
        first_index, last_index = region
        if last_index < first_index:
            self._prune_pixmap_lod_cache(region)
            return
        self._apply_lod_for_indexes(range(first_index, last_index + 1), region=region)
        self._prune_pixmap_lod_cache(region)

    def _apply_lod_for_indexes(self, indexes, *, region: tuple[int, int]) -> None:
        show_labels = self._matrix_zoom >= _LOD_LABEL_MIN_ZOOM
        show_status = self._matrix_zoom >= _LOD_STATUS_MIN_ZOOM
        for index in indexes:
            if index < 0 or index >= len(self._items):
                continue
            item = self._items[index]
            group = self._item_group_for(item)
            if group is None:
                continue
            _background_item, status_item, _selection_item, pixmap_item, text_item = group
            hidden = item.isHidden()
            in_region = self._index_in_render_region(index, region)
            if not in_region:
                for graphics_item in group:
                    graphics_item.setVisible(False)
                pixmap_item.setPixmap(QPixmap())
                continue
            if not hidden:
                new_pixmap = self._pixmap_for_item_lod(item)
                if new_pixmap.cacheKey() != pixmap_item.pixmap().cacheKey():
                    pixmap_item.setPixmap(new_pixmap)
                col = index % max(1, self._columns)
                row = index // max(1, self._columns)
                self._position_pixmap_item(
                    pixmap_item,
                    new_pixmap,
                    col * self._cell_step_x(),
                    row * self._cell_step_y(),
                )
            pixmap_item.setVisible((not hidden) and not pixmap_item.pixmap().isNull())
            text_item.setVisible((not hidden) and show_labels and bool(text_item.text()))
            status_item.setVisible((not hidden) and show_status and bool(status_item.data(1)))
        self._prune_pixmap_lod_cache(region)

    def _prune_pixmap_lod_cache(self, region: tuple[int, int]) -> None:
        first_index, last_index = region
        if last_index < first_index:
            self._pixmap_lod_cache.clear()
            return
        visible_item_ids: set[int] = set()
        for index in range(max(0, first_index), min(len(self._items) - 1, last_index) + 1):
            if self._index_in_render_region(index, region):
                visible_item_ids.add(id(self._items[index]))
        for key in list(self._pixmap_lod_cache):
            if not key or key[0] not in visible_item_ids:
                self._pixmap_lod_cache.pop(key, None)

    def _pixmap_for_item_lod(self, item: QListWidgetItem) -> QPixmap:
        zarr_pixmap = self._zarr_pixmap_for_item_lod(item)
        if zarr_pixmap is not None:
            return zarr_pixmap
        return QPixmap()

    def _zarr_pixmap_for_item_lod(self, item: QListWidgetItem) -> QPixmap | None:
        store = self._pyramid_store
        if store is None or not self._navigator_lods:
            return None
        frame_id = self._frame_id_for_item(item)
        if frame_id < 0:
            return None
        lod = self._navigator_lod()
        key = (id(item), int(frame_id), int(lod), self._grid_size.width(), self._grid_size.height())
        cached = self._pixmap_lod_cache.get(key)
        if cached is not None:
            return cached
        self._queue_pyramid_thumbnail_load(int(frame_id), int(lod))
        return None

    def _queue_pyramid_thumbnail_load(self, frame_id: int, lod: int) -> None:
        store = self._pyramid_store
        if store is None:
            return
        target_width = max(1, int(self._grid_size.width()))
        target_height = max(1, int(self._grid_size.height()))
        pending_key = (int(frame_id), int(lod), target_width, target_height)
        if pending_key in self._pyramid_thumbnail_pending:
            return
        self._pyramid_thumbnail_pending.add(pending_key)
        generation = int(self._pyramid_thumbnail_generation)
        runnable = PyramidThumbnailLoadRunnable(
            generation,
            int(frame_id),
            int(lod),
            store,
            target_width,
            target_height,
        )
        runnable.signals.result.connect(self._on_pyramid_thumbnail_loaded)
        runnable.signals.error.connect(self._on_pyramid_thumbnail_error)
        self._pyramid_thumbnail_thread_pool.start(runnable)

    def _on_pyramid_thumbnail_loaded(
        self,
        generation: int,
        frame_id: int,
        lod: int,
        target_width: int,
        target_height: int,
        qimage: object,
    ) -> None:
        pending_key = (int(frame_id), int(lod), int(target_width), int(target_height))
        self._pyramid_thumbnail_pending.discard(pending_key)
        if int(generation) != int(self._pyramid_thumbnail_generation):
            return
        item = self.item(int(frame_id))
        if item is None:
            return
        try:
            pixmap = QPixmap.fromImage(qimage)
        except Exception:
            return
        if pixmap.isNull():
            return
        pixmap = self._cover_crop_pixmap_to_cell_aspect(pixmap)
        key = (id(item), int(frame_id), int(lod), int(target_width), int(target_height))
        self._pixmap_lod_cache[key] = pixmap
        region = self._effective_render_index_range()
        self._prune_pixmap_lod_cache(region)
        if int(lod) != int(self._navigator_lod()):
            return
        if not self._index_in_render_region(int(frame_id), region):
            return
        group = self._item_group_for(item)
        if group is None or item.isHidden():
            return
        _background_item, _status_item, _selection_item, pixmap_item, _text_item = group
        pixmap_item.setPixmap(pixmap)
        col = int(frame_id) % max(1, self._columns)
        row = int(frame_id) // max(1, self._columns)
        self._position_pixmap_item(pixmap_item, pixmap, col * self._cell_step_x(), row * self._cell_step_y())
        pixmap_item.setVisible(True)

    def _on_pyramid_thumbnail_error(
        self,
        generation: int,
        frame_id: int,
        lod: int,
        target_width: int,
        target_height: int,
        _message: str,
    ) -> None:
        self._pyramid_thumbnail_pending.discard(
            (int(frame_id), int(lod), int(target_width), int(target_height))
        )

    def _frame_id_for_item(self, item: QListWidgetItem) -> int:
        value = item.data(_FRAME_ID_ROLE)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
        try:
            return self._items.index(item)
        except ValueError:
            return -1

    def _navigator_lod(self) -> int:
        if not self._navigator_lods:
            return 0
        if len(self._navigator_lods) == 1:
            return self._navigator_lods[0]
        zoom = max(MIN_ZOOM_FACTOR, min(MAX_ZOOM_FACTOR, float(self._matrix_zoom)))
        if zoom >= 1.0:
            return self._navigator_lods[0]
        if zoom >= 0.35 or len(self._navigator_lods) == 2:
            return self._navigator_lods[min(1, len(self._navigator_lods) - 1)]
        return self._navigator_lods[-1]

    def _thumbnail_lod(self) -> float:
        effective = max(MIN_ZOOM_FACTOR, min(MAX_ZOOM_FACTOR, self._matrix_zoom))
        for level in _THUMBNAIL_LOD_LEVELS:
            if effective <= level:
                return level
        return _THUMBNAIL_LOD_LEVELS[-1]

    def _cover_crop_pixmap_to_cell_aspect(self, pixmap: QPixmap) -> QPixmap:
        if pixmap.isNull():
            return pixmap
        target_aspect = max(1, self._grid_size.width()) / float(max(1, self._grid_size.height()))
        source_w = max(1, pixmap.width())
        source_h = max(1, pixmap.height())
        source_aspect = source_w / float(source_h)
        if abs(source_aspect - target_aspect) <= 1e-6:
            return pixmap
        if source_aspect > target_aspect:
            crop_w = max(1, int(round(source_h * target_aspect)))
            crop_x = max(0, (source_w - crop_w) // 2)
            return pixmap.copy(QRect(crop_x, 0, crop_w, source_h))
        crop_h = max(1, int(round(source_w / target_aspect)))
        crop_y = max(0, (source_h - crop_h) // 2)
        return pixmap.copy(QRect(0, crop_y, source_w, crop_h))

    def _pixmap_scene_scale(self, pixmap: QPixmap) -> float:
        if pixmap.isNull():
            return 1.0
        return max(
            max(1, self._grid_size.width()) / float(max(1, pixmap.width())),
            max(1, self._grid_size.height()) / float(max(1, pixmap.height())),
        )

    def _position_pixmap_item(self, pixmap_item: QGraphicsPixmapItem, pixmap: QPixmap, x: int, y: int) -> None:
        pixmap_item.setScale(self._pixmap_scene_scale(pixmap))
        display_w = pixmap.width() * pixmap_item.scale()
        display_h = pixmap.height() * pixmap_item.scale()
        pixmap_item.setPos(
            x + max(0, (self._grid_size.width() - display_w) / 2.0),
            y + max(0, (self._grid_size.height() - display_h) / 2.0),
        )

    def thumbnailSourceSize(self) -> QSize:
        lod = self._thumbnail_lod()
        width, height = clamp_thumbnail_source_size(
            int(round(self._icon_size.width() * lod)),
            int(round(self._icon_size.height() * lod)),
        )
        return QSize(width, height)

    def _emit_thumbnail_lod_if_needed(self) -> None:
        source_size = self.thumbnailSourceSize()
        if source_size == self._emitted_thumbnail_source_size:
            return
        self._emitted_thumbnail_source_size = QSize(source_size)
        self.thumbnailLodChanged.emit(source_size.width(), source_size.height())

    def _cell_step_x(self) -> int:
        cell_w = max(1, self._grid_size.width())
        return max(1, cell_w - min(self._overlap_pixels_x, max(0, cell_w - 1)))

    def _cell_step_y(self) -> int:
        cell_h = max(1, self._grid_size.height())
        return max(1, cell_h - min(self._overlap_pixels_y, max(0, cell_h - 1)))


def _trailing_frame_number(text: str) -> str:
    matches = re.findall(r"\d+", text)
    if not matches:
        return text
    return matches[-1]
