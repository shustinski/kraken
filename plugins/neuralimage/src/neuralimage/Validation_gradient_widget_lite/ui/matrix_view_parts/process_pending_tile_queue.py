"""matrix_view: process pending tile queue."""
from __future__ import annotations
import time

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.module_state import (
    SUBPIXEL_GRID_MAX_IN_FLIGHT,
    TILE_LOAD_MAX_KEYS_PER_SLICE,
    TILE_LOAD_MAX_PER_SLICE,
    TILE_LOAD_SLICE_BUDGET_MS,
    TILE_PREFETCH_MARGIN_CELLS,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.build_subpixel_grid_record import _build_subpixel_grid_for_record

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.store_subpixel_grid_disk import _load_subpixel_grid_from_disk


class ProcessPendingTileQueueMixin:
    def _process_pending_tile_queue(self) -> None:
        if (
            self._subpixel_spec is None
            or not self._tile_overlay_visible
            or self._tile_load_generation != self._tile_request_generation
        ):
            self._pending_tile_keys.clear()
            self._pending_tile_key_set.clear()
            return
        visible_keys = set(self._visible_record_keys(margin_cells=TILE_PREFETCH_MARGIN_CELLS))
        submitted = 0
        processed = 0
        slice_started = time.perf_counter()
        while (
            self._pending_tile_keys
            and len(self._tile_futures) < SUBPIXEL_GRID_MAX_IN_FLIGHT
            and submitted < TILE_LOAD_MAX_PER_SLICE
            and processed < TILE_LOAD_MAX_KEYS_PER_SLICE
            and (time.perf_counter() - slice_started) * 1000.0 < TILE_LOAD_SLICE_BUDGET_MS
        ):
            key = self._pending_tile_keys.popleft()
            self._pending_tile_key_set.discard(key)
            processed += 1
            if key not in visible_keys:
                continue
            item = self._item_by_key.get(key)
            if item is None or item.subpixel_grid is not None or self._assign_cached_subpixel_grid(item):
                continue
            spec = self._subpixel_spec
            if spec is None:
                continue
            cache_key = self._subpixel_cache_key_for_record(item.record, spec)
            disk_cached = _load_subpixel_grid_from_disk(cache_key)
            if disk_cached is not None:
                self._tile_disk_cache_hits += 1
                self._subpixel_cache_put(cache_key, disk_cached)
                self._apply_subpixel_grid_to_item(item, disk_cached)
                continue
            future = self._tile_executor.submit(
                _build_subpixel_grid_for_record,
                item.record,
                spec,
                aggregation=self._subpixel_aggregation,
                metric_key=self._metric_key,
                comparison_mode=self._subpixel_comparison_mode,
            )
            self._tile_futures[key] = (int(self._tile_load_generation or 0), cache_key, future)
            self._tile_jobs_submitted += 1
            submitted += 1
        if self._tile_futures and not self._tile_result_timer.isActive():
            self._tile_result_timer.start()
        if self._pending_tile_keys:
            self._tile_load_timer.start(0)
