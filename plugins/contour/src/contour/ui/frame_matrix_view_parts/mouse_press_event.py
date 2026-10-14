"""frame_matrix_view: mouse press event."""
from __future__ import annotations
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import (
    QListWidgetItem,
)
from ...application.frame_lod import PyramidFrameStore
from ...graphics.viewport_navigation import (
    zoom_factor_for_wheel_delta,
)


class MousePressEventMixin:
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
        if store is self._pyramid_store:
            self.refreshVisibleRegion()
            return
        self._pyramid_thumbnail_generation += 1
        self._pyramid_thumbnail_pending.clear()
        try:
            self._pyramid_thumbnail_thread_pool.clear()
        except RuntimeError:
            pass
        self._pyramid_store = store if store is not None and store.has_lod() else None
        self._navigator_lods = ()
        if self._pyramid_store is not None:
            try:
                lod_source = (
                    self._pyramid_store.available_lods_hint()
                    if self._pyramid_store.__class__ is PyramidFrameStore
                    else self._pyramid_store.available_lods()
                )
                lods = tuple(sorted(int(lod) for lod in lod_source))
            except Exception:
                self._pyramid_store = None
            else:
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
        horizontal_scrollbar = self.horizontalScrollBar()
        assert horizontal_scrollbar is not None
        vertical_scrollbar = self.verticalScrollBar()
        assert vertical_scrollbar is not None
        if event is None:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta:
                self._start_zoom_animation(event.position().toPoint(), zoom_factor_for_wheel_delta(delta))
            event.accept()
            return
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            angle_delta = event.angleDelta()
            delta_value = angle_delta.x() if angle_delta.x() else angle_delta.y()
            horizontal_scrollbar.setValue(horizontal_scrollbar.value() - delta_value)
            event.accept()
            return
        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            delta = event.angleDelta().y()
            vertical_scrollbar.setValue(vertical_scrollbar.value() - delta)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        viewport = self.viewport()
        assert viewport is not None
        horizontal_scrollbar = self.horizontalScrollBar()
        assert horizontal_scrollbar is not None
        vertical_scrollbar = self.verticalScrollBar()
        assert vertical_scrollbar is not None
        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle_pan_active = True
            self._middle_pan_origin = event.position().toPoint()
            self._middle_pan_scroll_origin = QPoint(
                horizontal_scrollbar.value(),
                vertical_scrollbar.value(),
            )
            viewport.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        item = self._item_at_viewport_pos(event.position().toPoint())
        if item is not None:
            try:
                self.setCurrentRow(self._items.index(item))
            except ValueError:
                pass
            if event.button() == Qt.MouseButton.RightButton:
                event.accept()
                return
            if not self._signals_blocked:
                self.itemClicked.emit(item)
                self.frameNavigationRequested.emit(self._frame_id_for_item(item))
            event.accept()
            return
        super().mousePressEvent(event)

    def contextMenuEvent(self, event) -> None:
        viewport = self.viewport()
        assert viewport is not None
        viewport_pos = viewport.mapFrom(self, event.pos())
        item = self._item_at_viewport_pos(viewport_pos)
        if item is None:
            event.ignore()
            return
        try:
            self.setCurrentRow(self._items.index(item))
        except ValueError:
            pass
        self.frameContextMenuRequested.emit(self._frame_id_for_item(item), event.globalPos())
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        horizontal_scrollbar = self.horizontalScrollBar()
        assert horizontal_scrollbar is not None
        vertical_scrollbar = self.verticalScrollBar()
        assert vertical_scrollbar is not None
        if self._middle_pan_active:
            if not (event.buttons() & Qt.MouseButton.MiddleButton):
                self._finish_middle_pan()
                return
            delta = event.position().toPoint() - self._middle_pan_origin
            horizontal_scrollbar.setValue(self._middle_pan_scroll_origin.x() - delta.x())
            vertical_scrollbar.setValue(self._middle_pan_scroll_origin.y() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)
