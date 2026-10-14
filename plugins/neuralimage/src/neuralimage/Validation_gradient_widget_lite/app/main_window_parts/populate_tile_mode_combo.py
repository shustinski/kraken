"""main_window: populate tile mode combo."""
from __future__ import annotations
from ...ui.ui_constants import (
    DEFAULT_SUBPIXEL_AGGREGATION,
    DEFAULT_SUBPIXEL_VIEW_MODE,
    DEFAULT_TILE_MODE,
    DEFAULT_TILE_OVERLAP_MODE,
    SUBPIXEL_AGGREGATION_OPTIONS,
    SUBPIXEL_VIEW_MODE_OPTIONS,
    TILE_MODE_OPTIONS,
    TILE_OVERLAP_MODE_OPTIONS,
)


class PopulateTileModeComboMixin:
    def _populate_tile_mode_combo(self, selected_value: str | None) -> None:
        current = str(selected_value or DEFAULT_TILE_MODE)
        if current == "subpixel":
            current = "tile"
        self.tile_mode_combo.blockSignals(True)
        self.tile_mode_combo.clear()
        for label_key, key in TILE_MODE_OPTIONS:
            self.tile_mode_combo.addItem(self._t(label_key), key)
        index = self.tile_mode_combo.findData(current)
        self.tile_mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.tile_mode_combo.blockSignals(False)

    def _populate_subpixel_view_mode_combo(self, selected_value: str | None) -> None:
        current = str(selected_value or DEFAULT_SUBPIXEL_VIEW_MODE)
        if current == "subpixel":
            current = "tile"
        self.subpixel_view_mode_combo.blockSignals(True)
        self.subpixel_view_mode_combo.clear()
        for label_key, key in SUBPIXEL_VIEW_MODE_OPTIONS:
            self.subpixel_view_mode_combo.addItem(self._t(label_key), key)
        index = self.subpixel_view_mode_combo.findData(current)
        self.subpixel_view_mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.subpixel_view_mode_combo.blockSignals(False)

    def _populate_subpixel_aggregation_combo(self, selected_value: str | None) -> None:
        current = str(selected_value or DEFAULT_SUBPIXEL_AGGREGATION)
        self.subpixel_aggregation_combo.blockSignals(True)
        self.subpixel_aggregation_combo.clear()
        for label_key, key in SUBPIXEL_AGGREGATION_OPTIONS:
            self.subpixel_aggregation_combo.addItem(self._t(label_key), key)
        index = self.subpixel_aggregation_combo.findData(current)
        self.subpixel_aggregation_combo.setCurrentIndex(index if index >= 0 else 0)
        self.subpixel_aggregation_combo.blockSignals(False)

    def _populate_tile_overlap_mode_combo(self, selected_value: str | None) -> None:
        current = str(selected_value or DEFAULT_TILE_OVERLAP_MODE)
        self.tile_overlap_mode_combo.blockSignals(True)
        self.tile_overlap_mode_combo.clear()
        for label_key, key in TILE_OVERLAP_MODE_OPTIONS:
            self.tile_overlap_mode_combo.addItem(self._t(label_key), key)
        index = self.tile_overlap_mode_combo.findData(current)
        self.tile_overlap_mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.tile_overlap_mode_combo.blockSignals(False)
