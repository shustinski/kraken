"""frame_matrix_view: apply lod indexes."""
from __future__ import annotations
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QListWidgetItem,
)
from ...adapters.qt.pyramid import PyramidThumbnailLoadRunnable

from contour.ui.frame_matrix_view_parts.module_state import (
    _LOD_LABEL_MIN_ZOOM,
    _LOD_STATUS_MIN_ZOOM,
    _THUMBNAIL_PIXMAP_ROLE,
)


class ApplyLodIndexesMixin:
    def _refresh_items(self) -> None:
        if not self._items:
            return
        region = self._effective_render_index_range()
        for existing_group in list(self._item_groups.values()):
            if not self._group_is_valid(existing_group):
                continue
            for graphics_item in existing_group:
                graphics_item.setVisible(False)
            existing_group[3].setPixmap(QPixmap())
        first_index, last_index = region
        if last_index >= first_index:
            for index in range(max(0, first_index), min(len(self._items) - 1, last_index) + 1):
                item = self._items[index]
                group = self._item_group_for(item)
                if group is None:
                    continue
                if not self._index_in_render_region(index, region):
                    continue
                self._layout_item_geometry(index, item, group, region=region)
        self._layout_dirty = False
        self._visible_index_range = region
        self._render_region_sync_allowed = bool(self._items)
        self._prune_pixmap_lod_cache(region)

    def _sync_render_region(self) -> None:
        if not self._items or self._layout_dirty or not self._render_region_sync_allowed:
            if self._layout_dirty and self._items:
                self._refresh_items()
            return
        region = self._effective_render_index_range()
        if region == self._visible_index_range:
            return
        previous_region = self._visible_index_range
        self._visible_index_range = region
        affected: set[int] = set()
        first_index, last_index = region
        prev_first, prev_last = previous_region
        if last_index >= first_index:
            affected.update(range(first_index, last_index + 1))
        if prev_last >= prev_first:
            affected.update(range(prev_first, prev_last + 1))
        for index in sorted(affected):
            if index < 0 or index >= len(self._items):
                continue
            item = self._items[index]
            group = self._item_group_for(item)
            if group is None:
                continue
            self._layout_item_geometry(index, item, group, region=region)
        self._prune_pixmap_lod_cache(region)

    def _apply_lod(self, *, region: tuple[int, int] | None = None) -> None:
        self._emit_thumbnail_lod_if_needed()
        if region is None:
            region = self._effective_render_index_range()
            self._visible_index_range = region
        first_index, last_index = region
        if last_index < first_index:
            self._prune_pixmap_lod_cache(region)
            return
        self._apply_lod_for_indexes(range(first_index, last_index + 1), region=region)
        self._prune_pixmap_lod_cache(region)

    def _apply_lod_for_indexes(self, indexes, *, region: tuple[int, int]) -> None:
        show_labels = self._matrix_zoom >= _LOD_LABEL_MIN_ZOOM
        show_status = self._matrix_zoom >= _LOD_STATUS_MIN_ZOOM
        for index in indexes:
            if index < 0 or index >= len(self._items):
                continue
            item = self._items[index]
            group = self._item_group_for(item)
            if group is None:
                continue
            _background_item, status_item, _selection_item, pixmap_item, text_item = group
            hidden = item.isHidden()
            in_region = self._index_in_render_region(index, region)
            if not in_region:
                for graphics_item in group:
                    graphics_item.setVisible(False)
                pixmap_item.setPixmap(QPixmap())
                continue
            if not hidden:
                new_pixmap = self._pixmap_for_item_lod(item)
                if new_pixmap.cacheKey() != pixmap_item.pixmap().cacheKey():
                    pixmap_item.setPixmap(new_pixmap)
                col = index % max(1, self._columns)
                row = index // max(1, self._columns)
                self._position_pixmap_item(
                    pixmap_item,
                    new_pixmap,
                    col * self._cell_step_x(),
                    row * self._cell_step_y(),
                )
            pixmap_item.setVisible((not hidden) and not pixmap_item.pixmap().isNull())
            text_item.setVisible((not hidden) and show_labels and bool(text_item.text()))
            status_item.setVisible((not hidden) and show_status and bool(status_item.data(1)))
        self._prune_pixmap_lod_cache(region)

    def _prune_pixmap_lod_cache(self, region: tuple[int, int]) -> None:
        first_index, last_index = region
        if last_index < first_index:
            self._pixmap_lod_cache.clear()
            return
        visible_item_ids: set[int] = set()
        for index in range(max(0, first_index), min(len(self._items) - 1, last_index) + 1):
            if self._index_in_render_region(index, region):
                visible_item_ids.add(id(self._items[index]))
        for key in list(self._pixmap_lod_cache):
            if not key or key[0] not in visible_item_ids:
                self._pixmap_lod_cache.pop(key, None)

    def _pixmap_for_item_lod(self, item: QListWidgetItem) -> QPixmap:
        pyramid_pixmap = self._pyramid_pixmap_for_item_lod(item)
        if pyramid_pixmap is not None:
            return pyramid_pixmap
        thumbnail_pixmap = item.data(_THUMBNAIL_PIXMAP_ROLE)
        if isinstance(thumbnail_pixmap, QPixmap) and not thumbnail_pixmap.isNull():
            return thumbnail_pixmap
        return QPixmap()

    def _pyramid_pixmap_for_item_lod(self, item: QListWidgetItem) -> QPixmap | None:
        store = self._pyramid_store
        if store is None or not self._navigator_lods:
            return None
        frame_id = self._frame_id_for_item(item)
        if frame_id < 0:
            return None
        lod = self._navigator_lod()
        key = (id(item), int(frame_id), int(lod), self._grid_size.width(), self._grid_size.height())
        cached = self._pixmap_lod_cache.get(key)
        if cached is not None:
            return cached
        self._queue_pyramid_thumbnail_load(int(frame_id), int(lod))
        return None

    def _queue_pyramid_thumbnail_load(self, frame_id: int, lod: int) -> None:
        store = self._pyramid_store
        if store is None:
            return
        target_width = max(1, int(self._grid_size.width()))
        target_height = max(1, int(self._grid_size.height()))
        pending_key = (int(frame_id), int(lod), target_width, target_height)
        if pending_key in self._pyramid_thumbnail_pending:
            return
        self._pyramid_thumbnail_pending.add(pending_key)
        generation = int(self._pyramid_thumbnail_generation)
        runnable = PyramidThumbnailLoadRunnable(
            generation,
            int(frame_id),
            int(lod),
            store,
            target_width,
            target_height,
        )
        runnable.signals.result.connect(self._on_pyramid_thumbnail_loaded)
        runnable.signals.error.connect(self._on_pyramid_thumbnail_error)
        self._pyramid_thumbnail_thread_pool.start(runnable)
