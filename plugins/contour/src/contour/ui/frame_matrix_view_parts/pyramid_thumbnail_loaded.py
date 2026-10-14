"""frame_matrix_view: pyramid thumbnail loaded."""
from __future__ import annotations
from PyQt6.QtGui import QImage, QPixmap


class PyramidThumbnailLoadedMixin:
    def _on_pyramid_thumbnail_loaded(
        self,
        generation: int,
        frame_id: int,
        lod: int,
        target_width: int,
        target_height: int,
        qimage: QImage,
    ) -> None:
        pending_key = (int(frame_id), int(lod), int(target_width), int(target_height))
        self._pyramid_thumbnail_pending.discard(pending_key)
        if int(generation) != int(self._pyramid_thumbnail_generation):
            return
        item = self.item(int(frame_id))
        if item is None:
            return
        try:
            pixmap = QPixmap.fromImage(qimage)
        except Exception:
            return
        if pixmap.isNull():
            return
        pixmap = self._cover_crop_pixmap_to_cell_aspect(pixmap)
        key = (id(item), int(frame_id), int(lod), int(target_width), int(target_height))
        self._pixmap_lod_cache[key] = pixmap
        region = self._effective_render_index_range()
        self._prune_pixmap_lod_cache(region)
        if int(lod) != int(self._navigator_lod()):
            return
        if not self._index_in_render_region(int(frame_id), region):
            return
        group = self._item_group_for(item)
        if group is None or item.isHidden():
            return
        _background_item, _status_item, _selection_item, pixmap_item, _text_item = group
        pixmap_item.setPixmap(pixmap)
        col = int(frame_id) % max(1, self._columns)
        row = int(frame_id) // max(1, self._columns)
        self._position_pixmap_item(pixmap_item, pixmap, col * self._cell_step_x(), row * self._cell_step_y())
        pixmap_item.setVisible(True)

    def _on_pyramid_thumbnail_error(
        self,
        generation: int,
        frame_id: int,
        lod: int,
        target_width: int,
        target_height: int,
        _message: str,
    ) -> None:
        self._pyramid_thumbnail_pending.discard(
            (int(frame_id), int(lod), int(target_width), int(target_height))
        )
