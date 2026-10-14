"""matrix_view: tile selection cell."""
from __future__ import annotations
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor, QPen
from ...core.analysis_modes import metric_visual_ratio
from ...core.domain import FrameRecord
from ...core.subpixel_grid import (
    SubpixelGridSpec,
)
from ...core.repository import metric_higher_is_better
from neuralimage.Validation_gradient_widget_lite.ui.ui_constants import (
    DEFAULT_BORDER,
    HOVER_BORDER,
    MATRIX_DEFAULT_PEN_WIDTH,
    MATRIX_HOVER_PEN_WIDTH,
    MATRIX_PROCESSING_PEN_WIDTH,
    MATRIX_REFERENCE_PEN_WIDTH,
    MATRIX_SELECTED_BLEND_RATIO,
    PROCESSING_BORDER,
    REFERENCE_BORDER,
    SELECTED_BLINK_COLOR,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.module_state import SUBPIXEL_VISIBILITY_EXIT_THRESHOLD

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.compute_auto_color_window import (
    blend_colors,
    enhance_palette_position,
    interpolate_gradient_color,
    map_score_to_palette_position,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.score_fill_color import (
    MatrixTileSelection,
    _display_tile_index_for_cell,
    _subpixel_overlay_visible_for_rect,
)


class TileSelectionCellMixin:
    def _subpixel_overlay_visible(self, spec: SubpixelGridSpec | None = None) -> bool:
        active_spec = self._subpixel_spec if spec is None else spec
        if active_spec is None:
            return False
        zoom_level = max(0.01, abs(float(self.transform().m11())))
        threshold = SUBPIXEL_VISIBILITY_EXIT_THRESHOLD if self._tile_overlay_visible else self._tile_zoom_threshold
        probe_rect = QRectF(0.0, 0.0, float(self._cell_size), float(self._cell_size))
        return _subpixel_overlay_visible_for_rect(
            probe_rect,
            active_spec,
            zoom_level,
            zoom_threshold=threshold,
        )

    def _update_tile_lod(self, *, force: bool = False) -> None:
        show_tiles = self._subpixel_overlay_visible()
        previous_visibility = self._tile_overlay_visible
        previous_lod_band = self._active_lod_band
        if show_tiles and self._focus_scene_pos is None:
            self._set_focus_scene_pos(self._zoom_anchor_scene_pos or self._viewport_center_scene_pos(), schedule=False)
        self._tile_overlay_visible = show_tiles
        if not show_tiles:
            self._set_hover_tile_selection(None)
        overview_changed = self._sync_overview_layer_visibility(force=force)
        self._active_lod_band = self._lod_band_for_current_view()
        lod_changed = self._active_lod_band != previous_lod_band
        if force or show_tiles != previous_visibility or overview_changed or lod_changed:
            self.viewport().update()
        self._sync_visible_matrix_items()
        if show_tiles:
            self._schedule_visible_tile_request(
                immediate=force or show_tiles != previous_visibility or lod_changed,
                preserve_generation=show_tiles == previous_visibility and not lod_changed and not force,
            )
        else:
            self._invalidate_tile_requests()

    def _clear_selection(self) -> None:
        if self._selected_item is not None:
            previous = self._selected_item
            self._selected_item = None
            self._selection_blink_on = False
            self._apply_item_style(previous)
            self._sync_low_zoom_visibility_for_keys({previous.record.key})
        self._emit_overview_state()

    def _set_hover_item(self, item: _MatrixCellItem | None) -> None:
        if self._hovered_item is item:
            return
        previous = self._hovered_item
        self._hovered_item = item
        if previous is not None:
            self._apply_item_style(previous)
        if item is not None:
            self._apply_item_style(item)
        self._sync_low_zoom_visibility_for_keys(
            {
                None if previous is None else previous.record.key,
                None if item is None else item.record.key,
            }
        )

    def _apply_item_style(self, item: _MatrixCellItem, *, sync_tile_state: bool = True) -> None:
        if item is self._hovered_item:
            pen = QPen(HOVER_BORDER, MATRIX_HOVER_PEN_WIDTH)
        elif item.record.key in self._processing_keys:
            pen = QPen(PROCESSING_BORDER, MATRIX_PROCESSING_PEN_WIDTH)
        elif self._reference_key is not None and item.record.key == self._reference_key:
            pen = QPen(REFERENCE_BORDER, MATRIX_REFERENCE_PEN_WIDTH)
        else:
            pen = QPen(DEFAULT_BORDER, MATRIX_DEFAULT_PEN_WIDTH)
        brush_color = self._background_color_for_record(item.record)
        if item is self._selected_item and self._selection_blink_on:
            brush_color = blend_colors(brush_color, SELECTED_BLINK_COLOR, MATRIX_SELECTED_BLEND_RATIO)
        item.setPen(pen)
        item.setBrush(brush_color)
        if sync_tile_state:
            self._sync_tile_state_for_keys({item.record.key})

    def _sync_tile_state_for_keys(self, record_keys) -> None:
        keys = {str(key) for key in record_keys}
        if not keys:
            return
        for key in keys:
            item = self._item_by_key.get(key)
            if item is None:
                continue
            selected_tile = self._selected_subpixel_selection if self._selected_subpixel_selection is not None and self._selected_subpixel_selection.record.key == key else None
            hovered_tile = self._hovered_subpixel_selection if self._hovered_subpixel_selection is not None and self._hovered_subpixel_selection.record.key == key else None
            item.set_tile_state(selected_tile, hovered_tile)

    def _tile_overlay_active(self) -> bool:
        return self._tile_overlay_visible

    def _tile_selection_for_cell(self, item: _MatrixCellItem, view_pos, *, allow_build: bool = False) -> MatrixTileSelection | None:
        spec = self._subpixel_spec
        if spec is None or not self._tile_overlay_active():
            return None
        rect = item.rect()
        if rect.width() <= 0.0 or rect.height() <= 0.0:
            return None
        scene_pos = self.mapToScene(view_pos)
        local_pos = item.mapFromScene(scene_pos)
        local_x = float(local_pos.x()) - float(rect.left())
        local_y = float(local_pos.y()) - float(rect.top())
        if local_x < 0.0 or local_y < 0.0 or local_x > rect.width() or local_y > rect.height():
            return None
        subpixel_grid = item.subpixel_grid
        if subpixel_grid is None and self._assign_cached_subpixel_grid(item):
            subpixel_grid = item.subpixel_grid
        if subpixel_grid is None and allow_build:
            subpixel_grid = self._subpixel_grid_for_record(item.record)
            item.subpixel_grid = subpixel_grid
        if subpixel_grid is None:
            if str(item.record.key) not in self._pending_tile_key_set:
                self._schedule_visible_tile_request()
            return None
        actual_spec = subpixel_grid.spec if subpixel_grid is not None else spec
        resolved = _display_tile_index_for_cell(local_x, local_y, rect, actual_spec)
        if resolved is None:
            return None
        tile_row, tile_column = resolved
        if subpixel_grid is not None:
            value = subpixel_grid.value_at(tile_row, tile_column)
            confidence = subpixel_grid.confidence_at(tile_row, tile_column)
            parent_value = subpixel_grid.aggregate_value(self._subpixel_aggregation)
        else:
            value = float(item.record.score if bool(getattr(item.record, "score_ready", False)) else 0.0)
            confidence = None
            parent_value = value
        return MatrixTileSelection(
            record=item.record,
            matrix_row=item.row,
            matrix_column=item.column,
            sub_row=tile_row,
            sub_column=tile_column,
            spec=actual_spec,
            parent_value=float(parent_value),
            subpixel_value=float(value),
            subpixel_confidence=confidence,
            aggregation=self._subpixel_aggregation,
            metric_key=str(self._metric_key or "overall_frame_score"),
        )

    def _toggle_selection_blink(self) -> None:
        if self._selected_item is None:
            if self._selection_blink_on:
                self._selection_blink_on = False
                self._emit_overview_state()
            return
        self._selection_blink_on = not self._selection_blink_on
        self._apply_item_style(self._selected_item)
        self._emit_overview_state()

    def _display_score(self, record: FrameRecord) -> float | None:
        if not bool(getattr(record, "score_ready", False)):
            return None
        if self._score_view_mode == "absolute":
            absolute_value = getattr(record, "absolute_score", None)
            if absolute_value is None:
                return None
            ratio = metric_visual_ratio(
                self._metric_key,
                float(absolute_value),
                point_match_radius=float(self._point_match_radius),
                bce_score_cap=float(self._bce_score_cap),
            )
            if ratio is None:
                return None
            higher_is_better = metric_higher_is_better(str(self._metric_key or ""))
            goodness = float(ratio) if higher_is_better else (1.0 - float(ratio))
            return max(0.0, min(goodness, 1.0))
        return float(record.score)

    @staticmethod
    def _format_metric_value(value: float) -> str:
        numeric = float(value)
        if 0.0 <= numeric <= 1.0:
            return f"{numeric * 100.0:.3f}%"
        return f"{numeric:.3f}"

    def _background_color(self, score: float) -> QColor:
        if self._score_view_mode == "absolute":
            position = max(0.0, min(float(score), 1.0))
        else:
            position = map_score_to_palette_position(score, self._auto_color_window_low, self._auto_color_window_high)
        position = enhance_palette_position(position)
        return interpolate_gradient_color(self._gradient_name, position)
