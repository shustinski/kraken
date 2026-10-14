"""matrix_view: prioritized record keys."""
from __future__ import annotations
import math
from PyQt6.QtCore import QPointF, QRectF
from ...core.domain import FrameRecord
from ...core.subpixel_grid import (
    SubpixelGrid,
)
from ...core.repository import natural_sort_key

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.module_state import (
    MATRIX_ITEM_KEEP_MARGIN_CELLS,
    MATRIX_MAX_MATERIALIZED_ITEMS,
    TILE_PREFETCH_MARGIN_CELLS,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.matrix_cell_item import _MatrixCellItem


class PrioritizedRecordKeysMixin:
    def _grid_position_for_scene_pos(self, scene_pos: QPointF | None) -> tuple[float, float] | None:
        if scene_pos is None:
            return None
        span = float(self._cell_size + self._gap)
        if span <= 0.0:
            return None
        column = (float(scene_pos.x()) - float(self._scene_padding)) / span
        row = (float(scene_pos.y()) - float(self._scene_padding)) / span
        if row < -1.0 or column < -1.0 or row > float(self._rows) or column > float(self._columns):
            return None
        return row, column

    def _active_focus_grid_position(self) -> tuple[float, float]:
        for candidate in (self._hover_scene_pos, self._zoom_anchor_scene_pos, self._focus_scene_pos):
            resolved = self._grid_position_for_scene_pos(candidate)
            if resolved is not None:
                return resolved
        resolved = self._grid_position_for_scene_pos(self._viewport_center_scene_pos())
        if resolved is not None:
            return resolved
        return max(0.0, (self._rows - 1) * 0.5), max(0.0, (self._columns - 1) * 0.5)

    def _prioritized_record_keys(self, *, margin_cells: int = TILE_PREFETCH_MARGIN_CELLS) -> tuple[tuple[tuple[float, float, float, float, tuple[object, ...]], str], ...]:
        if not self._record_by_position or self._columns <= 0 or self._rows <= 0:
            return tuple()
        visible_scene_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        if visible_scene_rect.width() <= 0.0 or visible_scene_rect.height() <= 0.0:
            return tuple()
        span = float(self._cell_size + self._gap)
        if span <= 0.0:
            return tuple()
        margin = max(0, int(margin_cells))
        left = float(visible_scene_rect.left()) - float(self._scene_padding)
        right = float(visible_scene_rect.right()) - float(self._scene_padding)
        top = float(visible_scene_rect.top()) - float(self._scene_padding)
        bottom = float(visible_scene_rect.bottom()) - float(self._scene_padding)
        min_column = max(0, int(math.floor(left / span)) - margin)
        max_column = min(self._columns - 1, int(math.floor(right / span)) + margin)
        min_row = max(0, int(math.floor(top / span)) - margin)
        max_row = min(self._rows - 1, int(math.floor(bottom / span)) + margin)
        if min_column > max_column or min_row > max_row:
            return tuple()
        visible_min_column = max(0, int(math.floor(left / span)))
        visible_max_column = min(self._columns - 1, int(math.floor(right / span)))
        visible_min_row = max(0, int(math.floor(top / span)))
        visible_max_row = min(self._rows - 1, int(math.floor(bottom / span)))
        focus_row, focus_column = self._active_focus_grid_position()
        pan_dx, pan_dy = self._pan_bias
        candidates: list[tuple[tuple[float, float, float, float, tuple[object, ...]], str]] = []
        for row in range(min_row, max_row + 1):
            for column in range(min_column, max_column + 1):
                record = self._record_by_position.get((row, column))
                if record is None:
                    continue
                inside_viewport = visible_min_row <= row <= visible_max_row and visible_min_column <= column <= visible_max_column
                d_row = float(row) - float(focus_row)
                d_col = float(column) - float(focus_column)
                ring = max(abs(d_row), abs(d_col))
                distance = abs(d_row) + abs(d_col)
                direction_bias = 0.0
                if abs(pan_dx) > 0.01 or abs(pan_dy) > 0.01:
                    direction_bias = -0.01 * ((float(column) - float(focus_column)) * pan_dx + (float(row) - float(focus_row)) * pan_dy)
                priority = (
                    0.0 if inside_viewport else 1.0,
                    float(ring),
                    float(distance),
                    float(direction_bias),
                    natural_sort_key(str(record.key)),
                )
                candidates.append((priority, str(record.key)))
        candidates.sort(key=lambda item: item[0])
        return tuple(candidates)

    def _assign_cached_subpixel_grid(self, item: _MatrixCellItem) -> bool:
        spec = self._subpixel_spec
        if spec is None:
            return False
        cached = self._subpixel_cache_get(self._subpixel_cache_key_for_record(item.record, spec))
        if cached is None:
            return False
        self._apply_subpixel_grid_to_item(item, cached, update=False)
        return True

    @staticmethod
    def _apply_subpixel_grid_to_item(item: _MatrixCellItem, grid: SubpixelGrid, *, update: bool = True) -> None:
        item.subpixel_grid = grid
        try:
            item.record.subpixel_grid = grid
        except Exception:
            pass
        if update:
            item.update()

    def _overlay_record_keys(self) -> set[str]:
        keys = set(self._overview_overlay_keys())
        if self._selected_subpixel_selection is not None:
            keys.add(str(self._selected_subpixel_selection.record.key))
        if self._hovered_subpixel_selection is not None:
            keys.add(str(self._hovered_subpixel_selection.record.key))
        return {key for key in keys if key}

    def _create_matrix_item(self, record: FrameRecord, row: int, column: int, index: int) -> _MatrixCellItem:
        x = self._scene_padding + int(column) * (self._cell_size + self._gap)
        y = self._scene_padding + int(row) * (self._cell_size + self._gap)
        item = _MatrixCellItem(QRectF(x, y, self._cell_size, self._cell_size), record, row, column, index)
        item.subpixel_spec = self._subpixel_spec
        item.subpixel_overlay_enabled = self._subpixel_spec is not None
        item.subpixel_grid_provider = self._subpixel_grid_for_record if self._subpixel_spec is not None else None
        grid = None
        if self._subpixel_spec is not None and self._tile_overlay_visible:
            grid = self._subpixel_cache_get(self._subpixel_cache_key_for_record(record, self._subpixel_spec))
        if grid is None:
            grid = getattr(record, "subpixel_grid", None)
        if not isinstance(grid, SubpixelGrid) or self._subpixel_spec is None:
            grid = None
        elif self._subpixel_spec.mode == "tile":
            if (
                grid.spec.mode != "tile"
                or int(grid.spec.tile_width) != int(self._subpixel_spec.tile_width)
                or int(grid.spec.tile_height) != int(self._subpixel_spec.tile_height)
                or int(grid.spec.overlap) != int(self._subpixel_spec.overlap)
            ):
                grid = None
        elif grid.spec.normalized() != self._subpixel_spec:
            grid = None
        item.subpixel_grid = grid
        item.subpixel_color_fn = self._subpixel_color_for_value
        item.subpixel_metric_key = self._metric_key
        item.setToolTip(self._tooltip_for_record(record))
        self._scene.addItem(item)
        self._item_by_key[str(record.key)] = item
        self._apply_item_style(item, sync_tile_state=False)
        self._sync_tile_state_for_keys({record.key})
        return item

    def _ensure_item_for_key(self, key: str | None) -> _MatrixCellItem | None:
        if key is None:
            return None
        normalized_key = str(key)
        item = self._item_by_key.get(normalized_key)
        if item is not None:
            return item
        position = self._record_positions.get(normalized_key)
        record = None if position is None else self._record_by_position.get(position)
        if record is None or position is None:
            return None
        return self._create_matrix_item(record, position[0], position[1], self._record_index_by_key.get(normalized_key, 0))

    def _remove_matrix_item(self, key: str) -> None:
        item = self._item_by_key.pop(str(key), None)
        if item is None:
            return
        if item is self._selected_item:
            self._selected_item = None
        if item is self._hovered_item:
            self._hovered_item = None
        self._scene.removeItem(item)

    def _keys_to_materialize(self) -> tuple[str, ...]:
        if not self._virtualized_items_enabled:
            return tuple(self._record_positions)
        visible_keys = self._visible_record_keys(margin_cells=MATRIX_ITEM_KEEP_MARGIN_CELLS)
        max_items = max(1, int(MATRIX_MAX_MATERIALIZED_ITEMS))
        ordered_keys = list(visible_keys[:max_items])
        key_set = set(ordered_keys)
        for key in self._overlay_record_keys():
            if key not in key_set:
                ordered_keys.append(key)
                key_set.add(key)
        return tuple(ordered_keys)

    def _sync_visible_matrix_items(self, *, force: bool = False) -> None:
        if not self._virtualized_items_enabled:
            return
        keep_keys = self._keys_to_materialize()
        keep_key_set = set(keep_keys)
        for key in keep_keys:
            self._ensure_item_for_key(key)
        overlay_keys = self._overlay_record_keys()
        for key in list(self._item_by_key):
            if key not in keep_key_set and key not in overlay_keys:
                self._remove_matrix_item(key)
        self._sync_overview_layer_visibility(force=True)
