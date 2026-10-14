"""navigation_mixin: flush thumbnail icon batch."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    THUMBNAIL_APPLY_INTERVAL_MS,
    THUMBNAIL_ICONS_APPLY_PER_TICK,
    QIcon,
    QPixmap,
    QSize,
    Qt,
)


class FlushThumbnailIconBatchMixin:
    def _flush_thumbnail_icon_batch(self: Any) -> None:
        if not self._frame_matrix_thumbnails_enabled() or not hasattr(self, "thumbnail_grid"):
            getattr(self, "_thumbnail_pending_apply", {}).clear()
            return
        pending: dict[str, object] = getattr(self, "_thumbnail_pending_apply", {})
        if not pending:
            return
        per_tick = max(1, int(THUMBNAIL_ICONS_APPLY_PER_TICK))
        batch_paths = list(pending.keys())[:per_tick]
        placeholder = self._thumbnail_placeholder()
        icon_cache = getattr(self, "_thumbnail_icon_cache", {})
        loaded_generation = getattr(self, "_thumbnail_loaded_generation", {})
        loaded_sizes = getattr(self, "_thumbnail_loaded_sizes", {})
        current_generation = self._thumbnail_generation
        applied_rows: list[int] = []
        stalled = 0
        for path in batch_paths:
            payload = pending.get(path)
            if payload is None:
                continue
            if isinstance(payload, tuple):
                if len(payload) == 3:
                    qimage, requested_size, payload_generation = payload
                elif len(payload) == 2:
                    qimage, requested_size = payload
                    payload_generation = current_generation
                else:
                    pending.pop(path, None)
                    continue
            else:
                qimage = payload
                requested_size = self._thumbnail_request_size()
                payload_generation = current_generation
            if int(payload_generation) != int(current_generation):
                pending.pop(path, None)
                continue
            row = self._thumbnail_row_for_path(path)
            if row is None:
                stalled += 1
                continue
            item = self.thumbnail_grid.item(row)
            if item is None:
                continue
            icon = icon_cache.get((path, requested_size))
            if icon is None:
                if qimage is None or (hasattr(qimage, "isNull") and qimage.isNull()):
                    icon = placeholder
                    item.setData(int(Qt.ItemDataRole.UserRole) + 1001, None)
                else:
                    target_w, target_h = max(1, int(requested_size[0])), max(1, int(requested_size[1]))
                    pixmap_image = qimage
                    if (
                        hasattr(qimage, "width")
                        and hasattr(qimage, "height")
                        and (int(qimage.width()) != target_w or int(qimage.height()) != target_h)
                    ):
                        scaled = qimage.scaled(
                            QSize(target_w, target_h),
                            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                        crop_x = max(0, (int(scaled.width()) - target_w) // 2)
                        crop_y = max(0, (int(scaled.height()) - target_h) // 2)
                        pixmap_image = scaled.copy(crop_x, crop_y, target_w, target_h)
                    pixmap = QPixmap.fromImage(pixmap_image)
                    icon = QIcon(pixmap) if not pixmap.isNull() else placeholder
                    item.setData(int(Qt.ItemDataRole.UserRole) + 1001, pixmap if not pixmap.isNull() else None)
                icon_cache[(path, requested_size)] = icon
            item.setIcon(icon)
            loaded_generation[path] = current_generation
            loaded_sizes[path] = requested_size
            pending.pop(path, None)
            applied_rows.append(int(row))
        if applied_rows and hasattr(self.thumbnail_grid, "updateThumbnailPixmaps"):
            self.thumbnail_grid.updateThumbnailPixmaps(applied_rows)
        if pending:
            delay_ms = 120 if stalled and not applied_rows else max(16, int(THUMBNAIL_APPLY_INTERVAL_MS))
            self._thumbnail_apply_timer.stop()
            self._thumbnail_apply_timer.start(delay_ms)
