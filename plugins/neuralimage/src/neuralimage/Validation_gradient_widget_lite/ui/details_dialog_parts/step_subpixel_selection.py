"""details_dialog: step subpixel selection."""
from __future__ import annotations
from ...core.subpixel_grid import (
    SubpixelSelection,
)
from ...ui.ui_constants import DEFAULT_SUBPIXEL_AGGREGATION


class StepSubpixelSelectionMixin:
    def _step_subpixel_selection(self, delta_row: int, delta_column: int) -> bool:
        selection = self._subpixel_selection
        if selection is None:
            return False
        spec = selection.spec or self._default_subpixel_spec()
        rows = max(1, int(spec.rows))
        columns = max(1, int(spec.columns))
        next_row = min(max(int(selection.sub_row) + int(delta_row), 0), rows - 1)
        next_column = min(max(int(selection.sub_column) + int(delta_column), 0), columns - 1)
        if next_row == int(selection.sub_row) and next_column == int(selection.sub_column):
            return False
        grid = self._subpixel_grid_for_view()
        if grid is not None and grid.spec.normalized() == spec.normalized():
            subpixel_value = float(grid.value_at(next_row, next_column))
            subpixel_confidence = grid.confidence_at(next_row, next_column)
            parent_value = float(grid.aggregate_value(selection.aggregation or DEFAULT_SUBPIXEL_AGGREGATION))
        else:
            subpixel_value = float(selection.subpixel_value)
            subpixel_confidence = selection.subpixel_confidence
            parent_value = float(selection.parent_value)
        self._subpixel_selection = SubpixelSelection(
            parent_row=int(selection.parent_row),
            parent_column=int(selection.parent_column),
            sub_row=int(next_row),
            sub_column=int(next_column),
            parent_value=float(parent_value),
            subpixel_value=float(subpixel_value),
            subpixel_confidence=None if subpixel_confidence is None else float(subpixel_confidence),
            aggregation=str(selection.aggregation or DEFAULT_SUBPIXEL_AGGREGATION),
            metric_key=str(selection.metric_key or self._comparison_score_metric_key()),
            spec=spec.normalized(),
        )
        self._session_view_state["subpixel_selection"] = {
            "parent_row": int(self._subpixel_selection.parent_row),
            "parent_column": int(self._subpixel_selection.parent_column),
            "sub_row": int(self._subpixel_selection.sub_row),
            "sub_column": int(self._subpixel_selection.sub_column),
            "parent_value": float(self._subpixel_selection.parent_value),
            "subpixel_value": float(self._subpixel_selection.subpixel_value),
            "subpixel_confidence": None if self._subpixel_selection.subpixel_confidence is None else float(self._subpixel_selection.subpixel_confidence),
            "aggregation": str(self._subpixel_selection.aggregation),
            "metric_key": str(self._subpixel_selection.metric_key),
            "spec_rows": int(self._subpixel_selection.spec.rows if self._subpixel_selection.spec is not None else spec.rows),
            "spec_columns": int(self._subpixel_selection.spec.columns if self._subpixel_selection.spec is not None else spec.columns),
            "spec_mode": str(self._subpixel_selection.spec.mode if self._subpixel_selection.spec is not None else spec.mode),
            "spec_tile_width": int(self._subpixel_selection.spec.tile_width if self._subpixel_selection.spec is not None else spec.tile_width),
            "spec_tile_height": int(self._subpixel_selection.spec.tile_height if self._subpixel_selection.spec is not None else spec.tile_height),
            "spec_overlap": int(self._subpixel_selection.spec.overlap if self._subpixel_selection.spec is not None else spec.overlap),
        }
        self._overlay_cache.clear()
        self._derived_cache.clear()
        self._refresh_scene(reset_view=False)
        self._refresh_info()
        self._update_tile_lod()
        self._store_view_settings()
        return True
