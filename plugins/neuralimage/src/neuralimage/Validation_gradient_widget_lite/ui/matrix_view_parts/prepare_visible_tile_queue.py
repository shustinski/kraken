"""matrix_view: prepare visible tile queue."""
from __future__ import annotations
from collections import deque

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.module_state import TILE_PREFETCH_MARGIN_CELLS


class PrepareVisibleTileQueueMixin:
    def _prepare_visible_tile_queue(self) -> None:
        if self._subpixel_spec is None or not self._tile_overlay_visible:
            return
        generation = self._tile_request_generation
        pending: list[str] = []
        pending_set: set[str] = set()
        for key in self._visible_record_keys(margin_cells=TILE_PREFETCH_MARGIN_CELLS):
            item = self._item_by_key.get(key)
            if item is None:
                continue
            if item.subpixel_grid is not None or self._assign_cached_subpixel_grid(item):
                continue
            if key in self._tile_futures:
                continue
            pending.append(key)
            pending_set.add(key)
        existing_pending = [key for key in self._pending_tile_keys if key not in pending_set and key in self._item_by_key]
        self._pending_tile_keys = deque(pending + existing_pending)
        self._pending_tile_key_set = set(self._pending_tile_keys)
        self._tile_load_generation = generation
        if self._pending_tile_keys:
            self._tile_load_timer.start(0)
