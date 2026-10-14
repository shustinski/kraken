"""matrix_view: subpixel grid record."""
from __future__ import annotations
from PyQt6.QtCore import QPointF
from ...core.domain import FrameRecord
from ...core.domain import ComparisonMode
from ...core.subpixel_grid import (
    SubpixelGrid,
    SubpixelGridSpec,
)
from neuralimage.Validation_gradient_widget_lite.ui.ui_constants import (
    DEFAULT_GRADIENT_NAME,
    GRADIENT_PRESETS,
    MATRIX_MIN_CELL_SIZE,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.module_state import SUBPIXEL_GRID_CACHE_MAX_ITEMS

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.build_subpixel_grid_record import _build_subpixel_grid_for_record

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.store_subpixel_grid_disk import (
    _load_subpixel_grid_from_disk,
    _store_subpixel_grid_to_disk,
    _subpixel_cache_key_for_record,
)


class SubpixelGridRecordMixin:
    def _clear_visible_record_key_cache(self) -> None:
        self._visible_record_key_cache_signature = None
        self._visible_record_key_cache = tuple()

    def _clear_subpixel_grid_cache(self) -> None:
        self._subpixel_grid_cache.clear()
        self._tile_cache_hits = 0
        self._tile_disk_cache_hits = 0
        self._tile_cache_misses = 0

    def _clear_record_subpixel_grids(self) -> None:
        for record in self._records:
            try:
                record.subpixel_grid = None
            except Exception:
                pass

    def _subpixel_cache_get(self, cache_key: tuple[object, ...]) -> SubpixelGrid | None:
        cached = self._subpixel_grid_cache.get(cache_key)
        if cached is not None:
            self._subpixel_grid_cache.move_to_end(cache_key)
            self._tile_cache_hits += 1
        return cached

    def _subpixel_cache_put(self, cache_key: tuple[object, ...], grid: SubpixelGrid) -> None:
        self._subpixel_grid_cache[cache_key] = grid
        self._subpixel_grid_cache.move_to_end(cache_key)
        while len(self._subpixel_grid_cache) > SUBPIXEL_GRID_CACHE_MAX_ITEMS:
            self._subpixel_grid_cache.popitem(last=False)

    def _subpixel_cache_key_for_record(self, record: FrameRecord, spec: SubpixelGridSpec) -> tuple[object, ...]:
        return _subpixel_cache_key_for_record(
            record,
            spec,
            aggregation=self._subpixel_aggregation,
            metric_key=self._metric_key,
            comparison_mode=self._subpixel_comparison_mode,
        )

    def _on_viewport_scroll_changed(self, *_args) -> None:
        self._update_pan_bias()
        self._clear_visible_record_key_cache()
        self._emit_overview_state()
        self._sync_visible_matrix_items()
        self._schedule_visible_tile_request(preserve_generation=True)

    def _viewport_center_scene_pos(self) -> QPointF:
        return self.mapToScene(self.viewport().rect().center())

    def _update_pan_bias(self) -> None:
        center = self._viewport_center_scene_pos()
        previous = self._last_viewport_center_scene_pos
        if previous is not None:
            dx = float(center.x() - previous.x())
            dy = float(center.y() - previous.y())
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                self._pan_bias = (dx, dy)
        self._last_viewport_center_scene_pos = center

    def _set_focus_scene_pos(self, scene_pos: QPointF | None, *, schedule: bool = True) -> None:
        self._focus_scene_pos = scene_pos
        self._clear_visible_record_key_cache()
        if schedule and self._tile_overlay_visible:
            self._schedule_visible_tile_request(immediate=False, preserve_generation=True)

    def _on_hover_prefetch_timeout(self) -> None:
        if self._hover_scene_pos is not None:
            self._set_focus_scene_pos(self._hover_scene_pos, schedule=True)

    def set_cell_size(self, cell_size: int) -> None:
        self._cell_size = max(MATRIX_MIN_CELL_SIZE, int(cell_size))
        self._clear_visible_record_key_cache()

    def set_gradient_preset(self, gradient_name: str) -> None:
        self._gradient_name = gradient_name if gradient_name in GRADIENT_PRESETS else DEFAULT_GRADIENT_NAME

    def set_error_window(self, low_bound: float, high_bound: float) -> None:
        self._error_window_low = max(0.0, min(float(low_bound), 1.0))
        self._error_window_high = max(0.0, min(float(high_bound), 1.0))

    def set_score_view_mode(self, mode: str | None) -> None:
        normalized = str(mode or "relative").strip().lower()
        self._score_view_mode = "absolute" if normalized == "absolute" else "relative"

    def set_metric_context(self, metric_key: str | None, *, point_match_radius: float, bce_score_cap: float) -> None:
        previous_metric_key = self._metric_key
        self._metric_key = None if metric_key is None else str(metric_key)
        self._point_match_radius = float(point_match_radius)
        self._bce_score_cap = float(bce_score_cap)
        if self._metric_key != previous_metric_key:
            self._clear_subpixel_grid_cache()
            self._clear_record_subpixel_grids()
            self._invalidate_tile_requests()
        for item in self._item_by_key.values():
            item.subpixel_metric_key = self._metric_key
            if self._metric_key != previous_metric_key:
                item.subpixel_grid = None
                item.update()

    def set_layout_config(self, layout_config: MatrixLayoutConfig) -> None:
        self._layout_config = layout_config
        self._clear_visible_record_key_cache()

    def set_subpixel_grid_spec(self, spec: SubpixelGridSpec | None, *, aggregation: str = "mean") -> None:
        previous_spec = self._subpixel_spec
        previous_aggregation = self._subpixel_aggregation
        self._subpixel_spec = None if spec is None else spec.normalized()
        self._subpixel_aggregation = str(aggregation or "mean")
        if self._subpixel_spec == previous_spec and self._subpixel_aggregation == previous_aggregation:
            return
        cache_invalidated = self._subpixel_spec != previous_spec or self._subpixel_aggregation != previous_aggregation
        if self._subpixel_spec != previous_spec:
            self._selected_subpixel_selection = None
            self._hovered_subpixel_selection = None
        if cache_invalidated:
            self._clear_subpixel_grid_cache()
            self._clear_record_subpixel_grids()
            self._invalidate_tile_requests()
        for item in self._item_by_key.values():
            item.subpixel_spec = self._subpixel_spec
            item.subpixel_overlay_enabled = self._subpixel_spec is not None
            item.subpixel_grid_provider = self._subpixel_grid_for_record if self._subpixel_spec is not None else None
            item.subpixel_color_fn = self._subpixel_color_for_value
            item.subpixel_metric_key = self._metric_key
            item.subpixel_grid = None
            item.update()
        if self._subpixel_spec != previous_spec:
            self._sync_tile_state_for_keys(self._item_by_key.keys())
        self._update_tile_lod(force=True)

    def set_subpixel_comparison_mode(self, comparison_mode) -> None:
        previous_mode = self._subpixel_comparison_mode
        if isinstance(comparison_mode, ComparisonMode):
            self._subpixel_comparison_mode = comparison_mode
        else:
            try:
                self._subpixel_comparison_mode = ComparisonMode(str(comparison_mode))
            except Exception:
                self._subpixel_comparison_mode = ComparisonMode.DISAGREEMENT
        if self._subpixel_comparison_mode != previous_mode:
            self._clear_subpixel_grid_cache()
            self._clear_record_subpixel_grids()
            self._invalidate_tile_requests()
            for item in self._item_by_key.values():
                item.subpixel_grid = None
                item.update()

    def set_tile_grid_plan(self, plan) -> None:  # pragma: no cover - compatibility shim
        if plan is None:
            self.set_subpixel_grid_spec(None)
            return
        rows = int(getattr(plan, "rows", 0) or 0)
        columns = int(getattr(plan, "columns", 0) or 0)
        self.set_subpixel_grid_spec(
            SubpixelGridSpec.from_tile_plan(
                tile_width=int(getattr(plan, "tile_width", 1) or 1),
                tile_height=int(getattr(plan, "tile_height", 1) or 1),
                overlap=int(getattr(plan, "overlap", 0) or 0),
                rows=rows,
                columns=columns,
            )
        )

    def _subpixel_grid_for_record(self, record: FrameRecord) -> SubpixelGrid | None:
        spec = self._subpixel_spec
        if spec is None:
            return getattr(record, "subpixel_grid", None)
        cache_key = self._subpixel_cache_key_for_record(record, spec)
        cached = self._subpixel_cache_get(cache_key)
        if cached is not None:
            return cached
        disk_cached = _load_subpixel_grid_from_disk(cache_key)
        if disk_cached is not None:
            self._tile_disk_cache_hits += 1
            self._subpixel_cache_put(cache_key, disk_cached)
            try:
                record.subpixel_grid = disk_cached
            except Exception:
                pass
            return disk_cached
        self._tile_cache_misses += 1
        grid = _build_subpixel_grid_for_record(
            record,
            spec,
            aggregation=self._subpixel_aggregation,
            metric_key=self._metric_key,
            comparison_mode=self._subpixel_comparison_mode,
        )
        self._subpixel_cache_put(cache_key, grid)
        _store_subpixel_grid_to_disk(cache_key, grid)
        try:
            record.subpixel_grid = grid
        except Exception:
            pass
        return grid
