"""matrix_view: rebuild scene."""
from __future__ import annotations
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QPen
from ...core.domain import FrameRecord
from neuralimage.Validation_gradient_widget_lite.ui.ui_constants import (
    SUBDUED_TEXT_COLOR,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.module_state import MATRIX_VIRTUALIZE_RECORD_THRESHOLD

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.build_matrix_layout import build_matrix_layout

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.compute_auto_color_window import compute_auto_color_window


class RebuildSceneMixin:
    def _rebuild_scene(self, records: list[FrameRecord]) -> None:
        self._invalidate_tile_requests()
        self._clear_visible_record_key_cache()
        self._records = list(records)
        ready_scores = [float(record.score) for record in self._records if bool(getattr(record, "score_ready", False))]
        self._auto_color_window_low, self._auto_color_window_high = compute_auto_color_window(ready_scores)
        selected_key = self._selected_item.record.key if self._selected_item is not None else None
        hovered_key = self._hovered_item.record.key if self._hovered_item is not None else None
        self._selected_item = None
        self._hovered_item = None
        self._selected_subpixel_selection = None
        self._hovered_subpixel_selection = None
        self._item_by_key.clear()
        self._record_by_position.clear()
        self._record_positions.clear()
        self._record_index_by_key.clear()
        self._scene.clear()
        self._overview_layer_item = None
        self._matrix_frame_item = None
        self._overview_layer_visible = False
        self._virtualized_items_enabled = False
        if not self._records:
            self._columns = 0
            self._rows = 0
            self._overview_image = None
            self.overviewChanged.emit(None, QRectF(), None, False, tuple(), None)
            return
        placements, self._columns, self._rows = build_matrix_layout(self._records, self._layout_config)
        matrix_rect = self._matrix_scene_rect()
        matrix_width = float(matrix_rect.width())
        matrix_height = float(matrix_rect.height())
        self._scene.setSceneRect(0, 0, matrix_width + self._scene_padding * 2, matrix_height + self._scene_padding * 2)
        self._matrix_frame_item = self._scene.addRect(matrix_rect, QPen(SUBDUED_TEXT_COLOR, 1.0))
        self._matrix_frame_item.setZValue(-5.0)
        self._virtualized_items_enabled = len(placements) >= MATRIX_VIRTUALIZE_RECORD_THRESHOLD
        for index, (record, row, column) in enumerate(placements):
            self._record_positions[record.key] = (row, column)
            self._record_by_position[(row, column)] = record
            self._record_index_by_key[record.key] = index
            if not self._virtualized_items_enabled:
                item = self._create_matrix_item(record, row, column, index)
                if selected_key == record.key:
                    self._selected_item = item
                if hovered_key == record.key:
                    self._hovered_item = item
        if self._virtualized_items_enabled:
            self._selected_item = self._ensure_item_for_key(selected_key)
            self._hovered_item = self._ensure_item_for_key(hovered_key)
        if self._selected_item is not None:
            self._apply_item_style(self._selected_item, sync_tile_state=False)
        if self._hovered_item is not None:
            self._apply_item_style(self._hovered_item, sync_tile_state=False)
        if not self._virtualized_items_enabled:
            self._sync_tile_state_for_keys(self._item_by_key.keys())
        self._overview_image = self._build_overview_image(placements)
        self._refresh_overview_layer_pixmap()
        self._sync_visible_matrix_items(force=True)
        self._emit_overview_state()
        self._update_tile_lod(force=True)
