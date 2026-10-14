"""processing_mixin: reset project."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
    is_image_path,
    is_visible_image_path,
)


class ResetProjectMixin:
    def load_images(
        self: Any,
        paths: list[str],
        *,
        preferred_current_image_path: str | None = None,
        image_list_mode: str | None = None,
    ) -> None:
        if image_list_mode is not None:
            self._set_image_list_mode(image_list_mode)
        if self._workspace.current_state is not None and not self._try_leave_current_frame():
            return
        self._directory_scanner.invalidate_pending_results()
        self._stop_work_simulation()
        normalized_paths = [str(Path(path)) for path in paths if is_image_path(path)]
        preferred = str(Path(preferred_current_image_path)) if preferred_current_image_path else None
        if preferred not in normalized_paths:
            preferred = None
        self._apply_image_paths_to_workspace(
            normalized_paths,
            clear_extra_layers=True,
            select_path=preferred,
            fallback_to_first=True,
            clear_state_cache=True,
        )
        return

    def append_images(self: Any, paths: list[str], *, select_first_new: bool = True) -> None:
        existing_paths = [str(Path(path)) for path in self._workspace.image_paths]
        existing_set = set(existing_paths)
        additions: list[str] = []
        seen = set(existing_set)
        for path in paths:
            normalized = str(Path(path))
            if normalized in seen or not is_visible_image_path(normalized):
                continue
            seen.add(normalized)
            additions.append(normalized)
        if not additions:
            return
        if self._workspace.current_state is not None and not self._try_leave_current_frame():
            return
        from ...infrastructure.settings_store import IMAGE_LIST_MODE_EXPLICIT

        self._set_image_list_mode(IMAGE_LIST_MODE_EXPLICIT)
        self._directory_scanner.invalidate_pending_results()
        self._stop_work_simulation()
        select_path = additions[0] if select_first_new else self._workspace.current_image_path
        self._apply_image_paths_to_workspace(
            [*existing_paths, *additions],
            clear_extra_layers=False,
            select_path=select_path,
            fallback_to_first=not bool(self._workspace.current_image_path),
            clear_state_cache=False,
        )

    def remove_images(self: Any, paths: list[str]) -> None:
        to_remove = {str(Path(path)) for path in paths}
        existing_paths = [str(Path(path)) for path in self._workspace.image_paths]
        remaining = [path for path in existing_paths if path not in to_remove]
        if remaining == existing_paths:
            return
        from ...infrastructure.settings_store import IMAGE_LIST_MODE_EXPLICIT

        current = self._workspace.current_image_path
        current_normalized = str(Path(current)) if current else None
        current_removed = current_normalized is not None and current_normalized in to_remove
        if current_removed:
            self._directory_scanner.invalidate_pending_results()
            self._stop_work_simulation()
            self._abort_in_flight_interactive_processing(preview=True, prepared=True)
        elif self._workspace.current_state is not None and not self._try_leave_current_frame():
            return
        else:
            self._directory_scanner.invalidate_pending_results()
            self._stop_work_simulation()
        self._set_image_list_mode(IMAGE_LIST_MODE_EXPLICIT)
        neighbor: str | None = None
        if current_removed:
            removed_index = existing_paths.index(current_normalized) if current_normalized in existing_paths else 0
            if remaining:
                neighbor = remaining[min(removed_index, len(remaining) - 1)]
        select_path = current_normalized if not current_removed else neighbor
        self._apply_image_paths_to_workspace(
            remaining,
            clear_extra_layers=False,
            select_path=select_path,
            fallback_to_first=select_path is None,
            clear_state_cache=False,
        )
        self._save_persisted_session_paths()

    def clear_frame_vectors(self: Any, image_path: str) -> None:
        normalized = str(Path(image_path))
        if normalized not in {str(Path(path)) for path in self._workspace.image_paths}:
            return
        self._workspace.mark_vectors_cleared(normalized)
        current = self._workspace.current_image_path
        if current is not None and str(Path(current)) == normalized:
            if hasattr(self, "polygon_editor"):
                self.polygon_editor.set_polygons([])
            self._sync_editor_polygons_to_current_workspace()
        self._refresh_image_list_item_states()
        self._refresh_vector_rows_for_workspace()
        if hasattr(self, "_refresh_image_list_item_states"):
            self._image_list_model.invalidate_path(normalized)

    def clear_all_loaded_vectors(self: Any) -> None:
        self._workspace.clear_cif_index()
        if hasattr(self, "cif_dir_edit"):
            self.cif_dir_edit.setText("")
        for image_path in list(self._workspace.image_paths):
            self._workspace.mark_vectors_cleared(image_path)
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_polygons([])
        if hasattr(self, "_sync_editor_polygons_to_current_workspace"):
            self._sync_editor_polygons_to_current_workspace()
        self._refresh_image_list_item_states()
        self._refresh_vector_rows_for_workspace()
        self._rebuild_vector_list()
        if hasattr(self, "_save_persisted_paths"):
            self._save_persisted_paths()

    def reset_project(self: Any) -> None:
        if self._workspace.current_state is not None and not self._try_leave_current_frame():
            return
        self._directory_scanner.invalidate_pending_results()
        self._vector_indexer.invalidate_pending_results()
        self._stop_work_simulation()
        self._abort_in_flight_interactive_processing(preview=True, prepared=True)
        self._workspace.clear_project()
        self._reset_thumbnail_disk_cache_for_base_paths([], force=True)
        self._base_frame_number_by_path = {}
        self._base_frame_numbers = set()
        self._neighbor_image_cache.clear()
        self._neighbor_image_dimensions.clear()
        self._neighbor_vector_cache.clear()
        with self._scene_source_image_cache_lock:
            self._scene_source_image_cache.clear()
            self._scene_source_image_cache_bytes = 0
        self._persisted_highlight_paths.clear()
        self._viewed_image_paths.clear()
        self._cif_load_failure_stems.clear()
        self._editor_pixmap_cache.clear()
        self._editor_polygons_signature = None
        self._thumbnail_path_to_row.clear()
        self._configure_pyramid_frame_store([])
        if hasattr(self, "_cancel_thumbnail_loading"):
            self._cancel_thumbnail_loading()
        getattr(self, "_thumbnail_icon_cache", {}).clear()
        self.input_dir_edit.setText("")
        self.cif_dir_edit.setText("")
        from ...infrastructure.settings_store import IMAGE_LIST_MODE_DIRECTORY

        self._set_image_list_mode(IMAGE_LIST_MODE_DIRECTORY)
        self._save_persisted_paths()
        self._save_persisted_current_image_path(None)
        self._save_persisted_session_paths()
        self._set_image_list_paths([])
        self._rebuild_thumbnail_grid()
        self._clear_extra_layers()
        self._update_extra_layers_enabled_state()
        self._rebuild_vector_list()
        self._refresh_image_list_item_states()
        self._sync_current_state_views()

    def _find_matching_cif_path(self: Any, image_path: str) -> str | None:
        return self._workspace.resolve_cif_path(image_path)

    def _should_defer_vector_load(self: Any) -> bool:
        return bool(getattr(self, "_defer_vector_load_until_cif_index", False))
