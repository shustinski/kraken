"""frame_matrix_view: attach graphics group item."""
from __future__ import annotations
from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QBrush, QColor, QGuiApplication, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QListWidget,
    QListWidgetItem,
)

from contour.ui.frame_matrix_view_parts.module_state import (
    _OPENGL_DISABLED_PLATFORMS,
    _OPENGL_VIEWPORT_ENABLED,
)


class AttachGraphicsGroupItemMixin:
    def _configure_opengl_viewport(self) -> bool:
        if not _OPENGL_VIEWPORT_ENABLED:
            return False
        app = QGuiApplication.instance()
        platform = QGuiApplication.platformName().lower() if app is not None else ""
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

    def widget(self) -> FrameMatrixGraphicsView:
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
    def _graphics_item_is_valid(graphics_item: QGraphicsItem) -> bool:
        if shiboken6 is None:
            try:
                graphics_item.scene()
                return True
            except RuntimeError:
                return False
        return bool(shiboken6.isValid(graphics_item))

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

    def blockSignals(self, block: bool) -> bool:
        previous = self._signals_blocked
        self._signals_blocked = bool(block)
        super().blockSignals(block)
        return previous

    def setUpdatesEnabled(self, enabled: bool) -> None:
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
