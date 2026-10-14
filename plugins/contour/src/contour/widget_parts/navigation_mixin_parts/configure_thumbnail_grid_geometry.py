"""navigation_mixin: configure thumbnail grid geometry."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    THUMBNAIL_SCROLL_SETTLE_MS,
    Path,
    PyramidFrameStore,
    QColor,
    QIcon,
    QPixmap,
    QSize,
    np,
)


class ConfigureThumbnailGridGeometryMixin:
    def _schedule_thumbnail_scroll_settle(self: Any) -> None:
        timer = getattr(self, "_thumbnail_scroll_settle_timer", None)
        if timer is None:
            return
        timer.stop()
        timer.start(max(16, int(THUMBNAIL_SCROLL_SETTLE_MS)))

    def _on_thumbnail_scroll_settled(self: Any) -> None:
        if bool(getattr(self, "_thumbnail_rebuild_in_progress", False)):
            return
        if hasattr(self, "thumbnail_grid") and hasattr(self.thumbnail_grid, "refreshVisibleRegion"):
            self.thumbnail_grid.refreshVisibleRegion()

    def _on_thumbnail_lod_changed(self: Any, *_args) -> None:
        self._thumbnail_generation += 1
        self._thumbnail_queued_paths.clear()
        self._thumbnail_queued_sizes.clear()
        getattr(self, "_thumbnail_pending_apply", {}).clear()
        try:
            self._thumbnail_thread_pool.clear()
        except AttributeError:
            pass
        self._prime_visible_thumbnail_reload()
        self._schedule_visible_thumbnail_loads()

    def _configure_thumbnail_grid_geometry(self: Any) -> None:
        if not self._frame_matrix_enabled() or not hasattr(self, "thumbnail_grid"):
            return
        columns = self._thumbnail_columns()
        icon_size = self._thumbnail_icon_size if hasattr(self, "_thumbnail_icon_size") else QSize(64, 48)
        cell_w = int(icon_size.width())
        cell_h = int(icon_size.height())
        self.thumbnail_grid.setIconSize(icon_size)
        self.thumbnail_grid.setGridSize(QSize(cell_w, cell_h))
        self.thumbnail_grid.setSpacing(0)
        full_overlap = 0
        if hasattr(self, "neighbor_overlap_spin"):
            try:
                full_overlap = max(0, int(self.neighbor_overlap_spin.value()))
            except (TypeError, ValueError):
                full_overlap = 0
        overlap_x, overlap_y = self._thumbnail_overlap_pixels_for_full_frame_overlap(full_overlap, icon_size)
        if hasattr(self.thumbnail_grid, "setFrameOverlapPixels"):
            self.thumbnail_grid.setFrameOverlapPixels(overlap_x, overlap_y)
        overlap_x = min(overlap_x, max(0, cell_w - 1))
        overlap_y = min(overlap_y, max(0, cell_h - 1))
        frame = 2 * int(self.thumbnail_grid.frameWidth())
        item_count = max(1, self._thumbnail_layout_slot_count())
        rows = max(1, int(np.ceil(item_count / float(columns))))
        # QListView icon wrap treats width==columns*cell_w as one column too few.
        grid_width = columns * cell_w - max(0, columns - 1) * overlap_x + 1 + frame
        grid_height = max(cell_h + frame, rows * cell_h - max(0, rows - 1) * overlap_y + frame)
        self.thumbnail_grid.setFixedSize(grid_width, grid_height)
        self.thumbnail_grid.doItemsLayout()

    def _thumbnail_overlap_pixels_for_full_frame_overlap(self: Any, full_overlap: int, icon_size: QSize) -> tuple[int, int]:
        full_overlap = max(0, int(full_overlap))
        if full_overlap <= 0:
            return (0, 0)
        full_w, full_h = self._frame_matrix_reference_image_size()
        if full_w <= 0 or full_h <= 0:
            return (0, 0)
        icon_w = max(1, int(icon_size.width()))
        icon_h = max(1, int(icon_size.height()))
        overlap_x = round(full_overlap * icon_w / float(full_w))
        overlap_y = round(full_overlap * icon_h / float(full_h))
        # An overlap equal to (or larger than) a cell would make the matrix
        # extent negative and all following frames collapse onto one point.
        return (
            min(max(0, overlap_x), icon_w - 1),
            min(max(0, overlap_y), icon_h - 1),
        )

    def _frame_matrix_reference_image_size(self: Any) -> tuple[int, int]:
        if hasattr(self, "_display_image_dimensions_for_vectors"):
            width, height = self._display_image_dimensions_for_vectors()
            if width > 0 and height > 0:
                return (width, height)
        current = getattr(self._workspace, "current_image_path", None)
        candidates = [current] if current else []
        candidates.extend(list(getattr(self._workspace, "image_paths", []))[:1])
        for path in candidates:
            if not path:
                continue
            try:
                from PyQt6.QtGui import QImageReader

                size = QImageReader(str(path)).size()
            except Exception:
                continue
            if size.isValid() and size.width() > 0 and size.height() > 0:
                return (int(size.width()), int(size.height()))
        return (0, 0)

    def _thumbnail_layout_slot_count(self: Any) -> int:
        if not hasattr(self, "thumbnail_grid"):
            return 0
        if not bool(getattr(self, "_asset_filter_match_only", False)):
            return self.thumbnail_grid.count()
        highest_visible_index = -1
        for index in range(self.thumbnail_grid.count()):
            item = self.thumbnail_grid.item(index)
            if item is not None and not item.isHidden():
                highest_visible_index = index
        return highest_visible_index + 1

    def _thumbnail_columns(self: Any) -> int:
        if not hasattr(self, "neighbor_columns_spin"):
            return 3
        try:
            return max(1, int(self.neighbor_columns_spin.value()))
        except (TypeError, ValueError):
            return 1

    def _thumbnail_placeholder(self: Any) -> QIcon:
        if not getattr(self, "_thumbnail_placeholder_icon", QIcon()).isNull():
            return self._thumbnail_placeholder_icon
        size = self._thumbnail_icon_size if hasattr(self, "_thumbnail_icon_size") else QSize(64, 48)
        pixmap = QPixmap(size)
        pixmap.fill(QColor("#1F2937"))
        self._thumbnail_placeholder_icon = QIcon(pixmap)
        return self._thumbnail_placeholder_icon

    def _thumbnail_paths_for_matrix(self: Any) -> list[str]:
        if not self._frame_matrix_enabled():
            return []
        return [str(Path(path)) for path in self._workspace.image_paths]

    def _configure_pyramid_frame_store(self: Any, image_paths: list[str]) -> None:
        normalized_paths = [str(Path(path)) for path in image_paths]
        existing_store = getattr(self, "_pyramid_frame_store", None)
        store: PyramidFrameStore | None
        if (
            isinstance(existing_store, PyramidFrameStore)
            and existing_store.image_paths == normalized_paths
        ):
            store = existing_store
        else:
            store = (
                PyramidFrameStore(normalized_paths, image_loader=getattr(self, "_source_image_loader", None))
                if normalized_paths else None
            )
        self._pyramid_frame_store = store
        current_path = str(Path(getattr(self._workspace, "current_image_path", "") or ""))
        current_frame_id = self._image_path_index(current_path) if current_path else None
        columns = self._thumbnail_columns()
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_pyramid_frame_store(
                store,
                frame_count=len(image_paths),
                columns=columns,
                current_frame_id=current_frame_id,
                enabled=False,
            )
        if hasattr(self, "thumbnail_grid"):
            self.thumbnail_grid.setPyramidFrameStore(store)

    def _ensure_neighbor_pyramid_or_prompt(self: Any) -> bool:
        if not self._neighbor_frames_enabled():
            return True
        image_paths = [str(Path(path)) for path in getattr(self._workspace, "image_paths", [])]
        if not image_paths:
            return True
        if not isinstance(getattr(self, "_pyramid_frame_store", None), PyramidFrameStore):
            self._configure_pyramid_frame_store(image_paths)
        return True
