"""details_dialog: refresh scene."""
from __future__ import annotations
from PyQt6.QtGui import QPixmap
from ...core.domain import ComparisonMode


class RefreshSceneMixin:
    def _refresh_scene(self, *, reset_view: bool) -> None:
        if self._payload_loading and not self._payload:
            self._tile_plan = None
            self._tile_selection = None
            self.original_item.setPixmap(QPixmap())
            self.first_source_item.setPixmap(QPixmap())
            self.second_source_item.setPixmap(QPixmap())
            self.result_item.setPixmap(QPixmap())
            self.tile_preview_item.setPixmap(QPixmap())
            self.tile_grid_item.setPixmap(QPixmap())
            self.tile_selection_item.setVisible(False)
            self._update_layer_states()
            return
        base_array = self._base_array()
        height, width = base_array.shape
        scene = self.overlay_view.scene()
        assert scene is not None
        scene.setSceneRect(0.0, 0.0, float(width), float(height))
        base_pixmap_key = (
            "base_pixmap",
            str(self._payload.get("selected_model_id") or ""),
            self._tile_selection,
            None
            if self._subpixel_selection is None
            else (
                int(self._subpixel_selection.parent_row),
                int(self._subpixel_selection.parent_column),
                int(self._subpixel_selection.sub_row),
                int(self._subpixel_selection.sub_column),
                int(getattr(self._subpixel_selection.spec, "rows", 0) or 0),
                int(getattr(self._subpixel_selection.spec, "columns", 0) or 0),
                str(getattr(self._subpixel_selection.spec, "mode", "grid") or "grid"),
                int(getattr(self._subpixel_selection.spec, "tile_width", 0) or 0),
                int(getattr(self._subpixel_selection.spec, "tile_height", 0) or 0),
                int(getattr(self._subpixel_selection.spec, "overlap", 0) or 0),
            ),
        )
        base_pixmap = self._overlay_cache.get(base_pixmap_key)
        if base_pixmap is None:
            base_pixmap = self._grayscale_to_pixmap(base_array)
            self._overlay_cache[base_pixmap_key] = base_pixmap
        self.original_item.setPixmap(base_pixmap)
        self._refresh_tile_overlays(base_array)
        _preset_key, first_key, second_key = self._current_comparison_tuple()
        mode = self._current_operation_mode()
        prefer_grayscale = self._selected_result_kind() == "diff" and mode == ComparisonMode.GRAYSCALE_DIFF
        self.first_source_item.setPixmap(self._source_pixmap(first_key, self._named_color("first_mask"), prefer_grayscale=prefer_grayscale))
        self.second_source_item.setPixmap(self._source_pixmap(second_key, self._named_color("second_mask"), prefer_grayscale=prefer_grayscale))
        self._refresh_result_layer()
        self._refresh_dynamic_labels()
        self._update_result_controls()
        self._update_layer_states()
        if reset_view:
            self._schedule_reset_view()
