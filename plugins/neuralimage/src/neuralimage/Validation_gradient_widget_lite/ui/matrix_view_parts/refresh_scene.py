"""matrix_view: refresh scene."""
from __future__ import annotations
from typing import TYPE_CHECKING
from ...core.domain import FrameRecord

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.module_state import (
    TILE_PREFETCH_MARGIN_CELLS,
    VIEW_LOD_OVERVIEW,
    VIEW_LOD_PIXEL,
    VIEW_LOD_SUBPIXEL,
)


class RefreshSceneMixin:
    def set_processing_keys(self, processing_keys) -> None:
        previous = set(self._processing_keys)
        self._processing_keys = {str(key) for key in processing_keys}
        for key in previous | self._processing_keys:
            item = self._ensure_item_for_key(key) if key in self._processing_keys else self._item_by_key.get(key)
            if item is not None:
                self._apply_item_style(item)
        self._sync_visible_matrix_items()
        self._sync_low_zoom_visibility_for_keys(previous | self._processing_keys)
        self._emit_overview_state()

    def set_reference_key(self, reference_key: str | None) -> None:
        previous = self._reference_key
        self._reference_key = str(reference_key) if reference_key else None
        for key in {previous, self._reference_key}:
            item = self._ensure_item_for_key(key) if key == self._reference_key else self._item_by_key.get(str(key)) if key else None
            if item is not None:
                self._apply_item_style(item)
        self._sync_visible_matrix_items()
        self._sync_low_zoom_visibility_for_keys({previous, self._reference_key})
        self._emit_overview_state()

    def set_records(self, records: list[FrameRecord], *, sort_mode: str = "name", reset_view: bool = False) -> None:
        if sort_mode == "name" and str(getattr(self._layout_config, "mode", "indexed_grid") or "indexed_grid") == "indexed_grid":
            ordered = list(records)
        else:
            ordered = self._sort_records(list(records), sort_mode)
        if reset_view:
            self.resetTransform()
        self._rebuild_scene(ordered)

    def refresh_scene(self) -> None:
        if not self._records:
            return
        for item in sorted(self._item_by_key.values(), key=lambda item: item.index):
            item.setToolTip(self._tooltip_for_record(item.record))
            self._apply_item_style(item, sync_tile_state=False)
        self._sync_tile_state_for_keys(self._item_by_key.keys())
        placements = [
            (record, position[0], position[1])
            for record in self._records
            for position in [self._record_positions.get(str(record.key))]
            if position is not None
        ]
        self._overview_image = self._build_overview_image(placements)
        self._refresh_overview_layer_pixmap()
        self._sync_visible_matrix_items()
        self._scene.update()
        self.viewport().update()
        self._emit_overview_state()
        self._update_tile_lod(force=True)

    def current_record(self) -> FrameRecord | None:
        return self._selected_item.record if self._selected_item is not None else None

    def selected_tile_selection(self) -> MatrixTileSelection | None:
        return self._selected_subpixel_selection

    def select_record_by_key(self, key: str, *, ensure_visible: bool = True) -> FrameRecord | None:
        item = self._ensure_item_for_key(str(key))
        if item is None:
            return None
        self._select_item(item)
        self.recordSelected.emit(item.record)
        if ensure_visible:
            self.centerOn(item)
            self._emit_overview_state()
        return item.record

    def neighbor_record(self, record: FrameRecord | str, direction: str) -> FrameRecord | None:
        key = record.key if isinstance(record, FrameRecord) else str(record)
        position = self._record_positions.get(key)
        if position is None:
            return None
        row, column = position
        direction_name = str(direction).lower()
        if direction_name == "right":
            return self._next_horizontal_record(row, column, step=1)
        if direction_name == "left":
            return self._next_horizontal_record(row, column, step=-1)
        if direction_name == "down":
            return self._next_vertical_record(row, column, step=1)
        if direction_name == "up":
            return self._next_vertical_record(row, column, step=-1)
        return None

    def _next_horizontal_record(self, row: int, column: int, *, step: int) -> FrameRecord | None:
        row_columns = sorted(col for (item_row, col) in self._record_by_position if item_row == row)
        if step > 0:
            for candidate_col in row_columns:
                if candidate_col > column:
                    return self._record_by_position.get((row, candidate_col))
            for candidate_row in sorted({item_row for (item_row, _col) in self._record_by_position if item_row > row}):
                candidate_cols = sorted(col for (item_row, col) in self._record_by_position if item_row == candidate_row)
                if candidate_cols:
                    return self._record_by_position.get((candidate_row, candidate_cols[0]))
            return None
        for candidate_col in reversed(row_columns):
            if candidate_col < column:
                return self._record_by_position.get((row, candidate_col))
        for candidate_row in sorted({item_row for (item_row, _col) in self._record_by_position if item_row < row}, reverse=True):
            candidate_cols = sorted(col for (item_row, col) in self._record_by_position if item_row == candidate_row)
            if candidate_cols:
                return self._record_by_position.get((candidate_row, candidate_cols[-1]))
        return None

    def _next_vertical_record(self, row: int, column: int, *, step: int) -> FrameRecord | None:
        rows = sorted({item_row for (item_row, _col) in self._record_by_position if item_row != row})
        if step < 0:
            rows = list(reversed(rows))
        for candidate_row in rows:
            if step > 0 and candidate_row <= row:
                continue
            if step < 0 and candidate_row >= row:
                continue
            if (candidate_row, column) in self._record_by_position:
                return self._record_by_position[(candidate_row, column)]
            candidate_cols = sorted(col for (item_row, col) in self._record_by_position if item_row == candidate_row)
            if candidate_cols:
                nearest_col = min(candidate_cols, key=lambda value: abs(value - column))
                return self._record_by_position.get((candidate_row, nearest_col))
        return None

    def _invalidate_tile_requests(self) -> None:
        self._tile_request_generation += 1
        self._tile_load_generation = None
        self._pending_tile_keys.clear()
        self._pending_tile_key_set.clear()
        for _key, (_generation, _cache_key, future) in list(self._tile_futures.items()):
            future.cancel()
        self._tile_futures.clear()
        self._tile_viewport_timer.stop()
        self._hover_prefetch_timer.stop()
        self._tile_load_timer.stop()
        self._tile_result_timer.stop()

    def _lod_band_for_current_view(self) -> str:
        if self._overview_layer_should_be_active():
            return VIEW_LOD_OVERVIEW
        if self._subpixel_overlay_visible():
            return VIEW_LOD_SUBPIXEL
        return VIEW_LOD_PIXEL

    def _schedule_visible_tile_request(self, *, immediate: bool = False, preserve_generation: bool = False) -> None:
        if self._subpixel_spec is None or not self._tile_overlay_visible:
            self._invalidate_tile_requests()
            return
        if not preserve_generation:
            self._tile_request_generation += 1
            self._tile_load_generation = None
            self._pending_tile_keys.clear()
            self._pending_tile_key_set.clear()
            self._tile_load_timer.stop()
        if immediate:
            self._tile_viewport_timer.stop()
            self._prepare_visible_tile_queue()
        else:
            self._tile_viewport_timer.start()

    def _visible_record_keys(self, *, margin_cells: int = TILE_PREFETCH_MARGIN_CELLS) -> tuple[str, ...]:
        signature = self._visible_record_key_signature(margin_cells=margin_cells)
        if signature == self._visible_record_key_cache_signature:
            return self._visible_record_key_cache
        prioritized = self._prioritized_record_keys(margin_cells=margin_cells)
        keys = tuple(key for _priority, key in prioritized)
        self._visible_record_key_cache_signature = signature
        self._visible_record_key_cache = keys
        return keys

    def _visible_record_key_signature(self, *, margin_cells: int) -> tuple[object, ...]:
        focus_row, focus_column = self._active_focus_grid_position()
        return (
            int(margin_cells),
            int(self.horizontalScrollBar().value()),
            int(self.verticalScrollBar().value()),
            int(self.viewport().width()),
            int(self.viewport().height()),
            round(float(self.transform().m11()), 4),
            int(self._rows),
            int(self._columns),
            int(len(self._record_positions)),
            int(self._cell_size),
            int(self._gap),
            int(self._scene_padding),
            round(float(focus_row), 3),
            round(float(focus_column), 3),
            round(float(self._pan_bias[0]), 3),
            round(float(self._pan_bias[1]), 3),
        )

if TYPE_CHECKING:
    from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.score_fill_color import MatrixTileSelection
