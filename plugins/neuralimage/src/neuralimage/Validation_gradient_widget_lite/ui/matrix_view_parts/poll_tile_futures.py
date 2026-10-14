"""matrix_view: poll tile futures."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QToolTip,
)
from ...core.domain import FrameRecord
from ...core.subpixel_grid import (
    SubpixelGrid,
)
from ...core.repository import natural_sort_key
from neuralimage.Validation_gradient_widget_lite.ui.ui_constants import (
    MATRIX_MAX_SCALE,
    MATRIX_MIN_SCALE,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.store_subpixel_grid_disk import _store_subpixel_grid_to_disk


class PollTileFuturesMixin:
    def _poll_tile_futures(self) -> None:
        if not self._tile_futures:
            self._tile_result_timer.stop()
            return
        for key, (generation, cache_key, future) in list(self._tile_futures.items()):
            if not future.done():
                continue
            self._tile_futures.pop(key, None)
            if generation != self._tile_request_generation or generation != self._tile_load_generation:
                self._tile_jobs_discarded += 1
                continue
            try:
                grid = future.result()
            except Exception:
                self._tile_jobs_discarded += 1
                continue
            if not isinstance(grid, SubpixelGrid):
                self._tile_jobs_discarded += 1
                continue
            self._subpixel_cache_put(cache_key, grid)
            _store_subpixel_grid_to_disk(cache_key, grid)
            item = self._item_by_key.get(key)
            if item is None:
                self._tile_jobs_discarded += 1
                continue
            self._apply_subpixel_grid_to_item(item, grid)
            self._tile_jobs_completed += 1
        if not self._tile_futures:
            self._tile_result_timer.stop()
        if self._pending_tile_keys and self._tile_load_generation == self._tile_request_generation:
            self._tile_load_timer.start(0)

    def tile_debug_stats(self) -> dict[str, int | str]:
        return {
            "lod_band": str(self._active_lod_band),
            "materialized_items": int(len(self._item_by_key)),
            "records": int(len(self._records)),
            "ram_cache_items": int(len(self._subpixel_grid_cache)),
            "ram_cache_hits": int(self._tile_cache_hits),
            "disk_cache_hits": int(self._tile_disk_cache_hits),
            "cache_misses": int(self._tile_cache_misses),
            "pending_tile_keys": int(len(self._pending_tile_keys)),
            "in_flight_tile_jobs": int(len(self._tile_futures)),
            "tile_jobs_submitted": int(self._tile_jobs_submitted),
            "tile_jobs_completed": int(self._tile_jobs_completed),
            "tile_jobs_discarded": int(self._tile_jobs_discarded),
            "focus_row": int(round(self._active_focus_grid_position()[0])),
            "focus_column": int(round(self._active_focus_grid_position()[1])),
        }

    def _sort_records(self, records: list[FrameRecord], sort_mode: str) -> list[FrameRecord]:
        mode = str(sort_mode or "name")
        if mode == "score_desc":
            return sorted(records, key=lambda item: float(item.score), reverse=True)
        if mode == "score_asc":
            return sorted(records, key=lambda item: float(item.score))
        if mode == "input_order":
            return list(records)
        return sorted(records, key=lambda item: natural_sort_key(item.display_name or item.key))

    def _handle_zoom_wheel(self, event) -> bool:
        modifiers = event.modifiers() | QApplication.keyboardModifiers()
        if not bool(modifiers & Qt.KeyboardModifier.ControlModifier):
            return False
        delta_y = event.angleDelta().y() or event.pixelDelta().y()
        if delta_y == 0:
            event.accept()
            return True
        factor = 1.15 if delta_y > 0 else (1.0 / 1.15)
        next_scale = self.transform().m11() * factor
        if MATRIX_MIN_SCALE <= next_scale <= MATRIX_MAX_SCALE:
            try:
                self._zoom_anchor_scene_pos = self.mapToScene(event.position().toPoint())
                self._set_focus_scene_pos(self._zoom_anchor_scene_pos, schedule=False)
            except Exception:
                self._zoom_anchor_scene_pos = self._viewport_center_scene_pos()
                self._set_focus_scene_pos(self._zoom_anchor_scene_pos, schedule=False)
            self.scale(factor, factor)
            self._emit_overview_state()
        event.accept()
        return True

    def wheelEvent(self, event) -> None:
        if self._handle_zoom_wheel(event):
            self._update_tile_lod()
            return
        super().wheelEvent(event)

    def closeEvent(self, event) -> None:
        self._invalidate_tile_requests()
        try:
            self._tile_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        super().closeEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_active = True
            self._pan_start = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            item = self._item_for_view_pos(event.pos())
            if item is not None:
                selection = self._tile_selection_for_cell(item, event.pos(), allow_build=True)
                if selection is not None:
                    self._select_tile_selection(selection)
                    self.recordSelected.emit(item.record)
                    self.tileSelected.emit(selection)
                else:
                    self._clear_tile_selection()
                    self._select_item(item)
                    self.recordSelected.emit(item.record)
            else:
                self._clear_tile_selection()
                self._clear_selection()
                self.recordSelected.emit(None)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            item = self._item_for_view_pos(event.pos())
            if item is not None:
                selection = self._tile_selection_for_cell(item, event.pos(), allow_build=True)
                if selection is not None:
                    self._select_tile_selection(selection)
                    self.recordSelected.emit(item.record)
                    self.tileActivated.emit(selection)
                else:
                    self._clear_tile_selection()
                    self._select_item(item)
                    self.recordSelected.emit(item.record)
                    self.recordActivated.emit(item.record)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event) -> None:
        item = self._item_for_view_pos(event.pos())
        record = None
        if item is not None:
            selection = self._tile_selection_for_cell(item, event.pos(), allow_build=True)
            if selection is not None:
                self._select_tile_selection(selection)
                self.recordSelected.emit(item.record)
                self.tileSelected.emit(selection)
            else:
                self._clear_tile_selection()
                self._select_item(item)
                self.recordSelected.emit(item.record)
            record = item.record
        self.contextMenuRequested.emit(record, event.globalPos())
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._pan_active and self._pan_start is not None:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self._pan_bias = (-float(delta.x()) / max(1.0, float(self.transform().m11())), -float(delta.y()) / max(1.0, float(self.transform().m11())))
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            self._emit_overview_state()
            event.accept()
            return
        item = self._item_for_view_pos(event.pos())
        if item is not None:
            self._hover_scene_pos = self.mapToScene(event.pos())
            self._hover_prefetch_timer.start()
            selection = self._tile_selection_for_cell(item, event.pos())
            if selection is not None:
                self._set_hover_tile_selection(selection)
                self._set_hover_item(item)
                QToolTip.showText(event.globalPosition().toPoint(), self._tile_hover_text(selection), self.viewport())
            else:
                self._set_hover_tile_selection(None)
                self._set_hover_item(item)
                QToolTip.showText(event.globalPosition().toPoint(), self._hover_text(item.record), self.viewport())
        else:
            self._hover_scene_pos = None
            self._hover_prefetch_timer.stop()
            self._set_hover_tile_selection(None)
            self._set_hover_item(None)
            QToolTip.hideText()
        super().mouseMoveEvent(event)
