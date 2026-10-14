"""processing_mixin: apply image paths workspace."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
    build_base_frame_number_map,
    build_base_frame_records,
    is_image_path,
)


class ApplyImagePathsWorkspaceMixin:
    def _apply_image_paths_to_workspace(
        self: Any,
        paths: list[str],
        *,
        clear_extra_layers: bool,
        select_path: str | None,
        fallback_to_first: bool,
        clear_state_cache: bool = True,
    ) -> None:
        source_numbers = getattr(self, "_source_frame_numbers", None)
        if source_numbers is not None:
            from ...application.frame_layers import BaseFrameRecord

            frame_records = [
                BaseFrameRecord(str(Path(path)), Path(path).stem, source_numbers[str(Path(path))])
                for path in paths
            ]
            warnings = []
        else:
            frame_records, warnings = build_base_frame_records(paths)
        for message in warnings:
            self._append_log(message)
        ordered_paths = [record.path for record in frame_records]
        self._base_frame_number_by_path = build_base_frame_number_map(frame_records)
        self._base_frame_numbers = set(self._base_frame_number_by_path.values())
        previous_paths = [str(Path(path)) for path in self._workspace.image_paths]
        preserve_cached_states = previous_paths == ordered_paths
        normalized_paths = self._workspace.replace_image_selection(
            ordered_paths,
            is_supported_image=is_image_path,
            clear_state_cache=bool(clear_state_cache and not preserve_cached_states),
        )
        self._configure_pyramid_frame_store(normalized_paths)
        self._reset_thumbnail_disk_cache_for_base_paths(normalized_paths)
        if not normalized_paths:
            self._save_persisted_current_image_path(None)
            self._save_persisted_session_paths()
        if not preserve_cached_states:
            with self._scene_source_image_cache_lock:
                self._scene_source_image_cache.clear()
                self._scene_source_image_cache_bytes = 0
            self._neighbor_image_cache.clear()
            self._neighbor_image_dimensions.clear()
            self._neighbor_vector_cache.clear()
            self._editor_pixmap_cache.clear()
            self._editor_polygons_signature = None
            getattr(self, "_thumbnail_icon_cache", {}).clear()
        getattr(self, "_thumbnail_pending_apply", {}).clear()
        self._thumbnail_loaded_generation.clear()
        self._thumbnail_loaded_sizes.clear()
        self._thumbnail_queued_paths.clear()
        self._thumbnail_queued_sizes.clear()
        self._prune_tagged_sets_for_images(normalized_paths)
        self._abort_in_flight_interactive_processing(preview=True, prepared=True)
        if clear_extra_layers:
            self._clear_extra_layers()
        self._update_extra_layers_enabled_state()
        self._pending_image_list_post_build = {
            "select_path": select_path,
            "fallback_to_first": fallback_to_first,
        }
        self._rebuild_image_list_items(normalized_paths)
        return
