"""processing_mixin: apply cached neighbor frames."""
from __future__ import annotations
from typing import Any
from PyQt6.QtGui import QImage, QImageReader
from contour.widget_parts._imports import (
    Path,
    PolygonData,
    QTimer,
    load_polygons_vector,
)


class ApplyCachedNeighborFramesMixin:
    def _neighbor_source_size(self: Any, image_path: str, image: object) -> tuple[int, int]:
        normalized = str(Path(image_path))
        cached = getattr(self, "_neighbor_image_dimensions", {}).get(normalized)
        if cached is not None:
            return max(1, int(cached[0])), max(1, int(cached[1]))
        state = getattr(self._workspace, "_state_cache", {}).get(normalized)
        source = getattr(state, "source_image", None) if state is not None else None
        if source is not None:
            height, width = source.shape[:2]
            size = (max(1, int(width)), max(1, int(height)))
            self._neighbor_image_dimensions[normalized] = size
            return size
        reader_size = QImageReader(normalized).size()
        if reader_size.isValid():
            size = (max(1, int(reader_size.width())), max(1, int(reader_size.height())))
            self._neighbor_image_dimensions[normalized] = size
            return size
        if isinstance(image, QImage) and not image.isNull():
            return max(1, image.width()), max(1, image.height())
        width = int(getattr(image, "shape", (1, 1))[1]) if hasattr(image, "shape") else 1
        height = int(getattr(image, "shape", (1, 1))[0]) if hasattr(image, "shape") else 1
        return max(1, width), max(1, height)

    def _neighbor_frame_vectors(self: Any, image_path: str) -> tuple[list[PolygonData], tuple[int, int] | None]:
        if not self._neighbor_vectors_enabled():
            return [], None
        normalized = str(Path(image_path))
        cache = getattr(self, "_neighbor_vector_cache", {})
        cached = cache.get(normalized)
        if cached is not None:
            return cached
        cif_path = self._find_matching_cif_path(normalized)
        if not cif_path:
            cache[normalized] = ([], None)
            self._neighbor_vector_cache = cache
            return [], None
        try:
            _referenced_image, image_size, polygons = load_polygons_vector(cif_path)
        except Exception:
            polygons = []
            image_size = None
        source_size = None
        if image_size is not None:
            source_size = (max(1, int(image_size[0])), max(1, int(image_size[1])))
        cached = ([polygon.clone() for polygon in polygons], source_size)
        cache[normalized] = cached
        if len(cache) > 256:
            cache.pop(next(iter(cache)))
        self._neighbor_vector_cache = cache
        return cached

    def _apply_cached_neighbor_frames(self: Any) -> None:
        if not self._neighbor_frames_enabled():
            self._neighbor_sync_image_path = None
            self._neighbor_frame_specs = []
            self._neighbor_queued_paths.clear()
            self.polygon_editor.set_neighbor_frames([], 0.0, 0, False)
            return
        if not self._neighbor_sync_is_current():
            return
        frames: list[tuple[int, int, object, str, list[PolygonData], tuple[int, int]]] = []
        include_vectors = self._neighbor_vectors_enabled()
        for column_offset, row_offset, image_path in getattr(self, "_neighbor_frame_specs", []):
            image = self._try_load_neighbor_preview_image(image_path)
            if image is None:
                continue
            source_size = self._neighbor_source_size(image_path, image)
            polygons: list[PolygonData] = []
            if include_vectors:
                polygons, vector_source_size = self._neighbor_frame_vectors(image_path)
                if vector_source_size is not None:
                    source_size = vector_source_size
            frames.append((column_offset, row_offset, image, image_path, polygons, source_size))
        if not frames and bool(getattr(self, "_neighbor_queued_paths", set())):
            return
        self.polygon_editor.set_neighbor_frames(
            frames,
            float(self.neighbor_opacity_spin.value()),
            int(self.neighbor_overlap_spin.value()),
            True,
        )
        if getattr(self, "_neighbor_queued_paths", set()):
            self._schedule_neighbor_frame_apply(delay_ms=100)
        else:
            current_path = self._workspace.current_image_path
            session = (
                self._frame_switch_profile_for_path(str(current_path))
                if current_path
                else None
            )
            if session is not None:
                session.complete_pending("neighbor_sync", suffix="_applied")
            QTimer.singleShot(0, self._center_editor_on_current_main_image)

    def _center_editor_on_current_main_image(self: Any, *, force: bool = False) -> None:
        if not hasattr(self, "polygon_editor") or not self._workspace.current_image_path:
            return
        if self._should_preserve_editor_view_position(self._workspace.current_image_path):
            return
        if not self.polygon_editor.should_auto_reposition_view(force=force):
            return
        if self.polygon_editor.pyramid_mode_enabled():
            frame_id = self._image_path_index(self._workspace.current_image_path)
            self.polygon_editor.set_current_frame_id(frame_id, center=True, emit_signal=False)
            return
        self.polygon_editor.center_main_image()

    def _should_preserve_editor_view_position(self: Any, image_path: str | None) -> bool:
        preserve_path = str(getattr(self, "_preserve_editor_view_position_path", "") or "")
        return bool(
            preserve_path
            and image_path
            and str(Path(image_path)) == str(Path(preserve_path))
        )

    def _odd_neighbor_grid_size(self: Any, value: int) -> int:
        size = max(3, min(7, int(value)))
        return size if size % 2 else size - 1

    def _neighbor_grid_size(self: Any) -> int:
        return self._odd_neighbor_grid_size(self.neighbor_max_grid_spin.value())
