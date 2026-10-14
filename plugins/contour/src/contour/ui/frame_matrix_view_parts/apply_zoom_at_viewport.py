"""frame_matrix_view: apply zoom at viewport."""
from __future__ import annotations
from PyQt6.QtCore import QPoint, QRectF, Qt
from PyQt6.QtWidgets import (
    QGraphicsView,
    QListWidgetItem,
)
from ...graphics.viewport_navigation import (
    clamp_zoom_factor,
    viewport_scroll_correction_after_scale_reanchor,
)

from contour.ui.frame_matrix_view_parts.module_state import (
    _FOCUS_NEIGHBOR_ROW_BUFFER,
    _MATRIX_ZOOM_EASING_FRACTION,
    _MATRIX_ZOOM_SETTLE_RATIO,
    _RENDER_REGION_BUFFER_CELLS,
)


class ApplyZoomAtViewportMixin:
    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._middle_pan_active:
            self._finish_middle_pan()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _finish_middle_pan(self) -> None:
        viewport = self.viewport()
        assert viewport is not None
        self._middle_pan_active = False
        viewport.unsetCursor()

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
        viewport = self.viewport()
        assert viewport is not None
        horizontal_scrollbar = self.horizontalScrollBar()
        assert horizontal_scrollbar is not None
        vertical_scrollbar = self.verticalScrollBar()
        assert vertical_scrollbar is not None
        old_zoom = self._matrix_zoom
        new_zoom = clamp_zoom_factor(old_zoom * float(factor))
        if abs(new_zoom - old_zoom) <= 1e-9:
            return
        factor = new_zoom / old_zoom
        view_point = viewport.mapTo(self, viewport_pixel)
        scene_anchor = self.mapToScene(view_point)
        self.scale(factor, factor)
        self._matrix_zoom = new_zoom
        mapped = viewport.mapFrom(self, self.mapFromScene(scene_anchor))
        dh, dv = viewport_scroll_correction_after_scale_reanchor((viewport_pixel.x(), viewport_pixel.y()), (mapped.x(), mapped.y()))
        horizontal_scrollbar.setValue(horizontal_scrollbar.value() + dh)
        vertical_scrollbar.setValue(vertical_scrollbar.value() + dv)
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
        viewport = self.viewport()
        assert viewport is not None
        if not self._items:
            return (0, -1)
        columns = max(1, self._columns)
        step_x = max(1, self._cell_step_x())
        step_y = max(1, self._cell_step_y())
        viewport_rect = viewport.rect()
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
