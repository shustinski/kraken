"""processing_mixin: prefetch frame neighborhood."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
    neighborhood_indices,
)


class PrefetchFrameNeighborhoodMixin:
    def _schedule_frame_prefetch(self: Any) -> None:
        if bool(getattr(self, "_closing", False)):
            return
        if getattr(self, "_frame_switch_profile", None) is not None:
            self._prefetch_deferred_for_profile = True
            timer = getattr(self, "_prefetch_timer", None)
            if timer is not None:
                timer.stop()
            return
        self._prefetch_deferred_for_profile = False
        timer = getattr(self, "_prefetch_timer", None)
        if timer is None:
            return
        timer.stop()
        timer.start(80)

    def _prefetch_neighborhood_paths(self: Any) -> tuple[str, ...]:
        paths = self._ordered_workspace_image_paths()
        current = self._workspace.current_image_path
        if not paths or current is None:
            return ()
        mapping = getattr(self, "_image_path_to_index", {})
        current_index = mapping.get(current) if mapping else None
        if current_index is None:
            current_normalized = current if current in paths else str(Path(current))
            try:
                current_index = paths.index(current_normalized)
            except ValueError:
                return ()
        matrix_enabled = bool(self._frame_matrix_enabled()) if hasattr(self, "_frame_matrix_enabled") else False
        columns = int(self._thumbnail_columns()) if hasattr(self, "_thumbnail_columns") else 1
        indices = neighborhood_indices(
            current_index,
            len(paths),
            matrix_enabled=matrix_enabled,
            columns=columns,
        )
        return tuple(paths[index] for index in indices)

    def _prefetch_frame_neighborhood(self: Any) -> None:
        if bool(getattr(self, "_closing", False)):
            return
        if getattr(self, "_frame_switch_profile", None) is not None:
            self._prefetch_deferred_for_profile = True
            return
        if getattr(self, "_frame_load_running_path", None) is not None:
            self._schedule_frame_prefetch()
            return
        neighborhood = self._prefetch_neighborhood_paths()
        if not neighborhood:
            return
        neighborhood_set = frozenset(neighborhood)
        self._prefetch_neighborhood_set = neighborhood_set
        self._workspace.retain_cached_states(neighborhood)
        self._prefetch_generation = int(getattr(self, "_prefetch_generation", 0)) + 1
        generation = self._prefetch_generation
        current = self._workspace.current_image_path or ""
        if current and current not in getattr(self, "_image_path_to_index", {}):
            current = str(Path(current))
        running = getattr(self, "_frame_load_running_path", "") or ""
        if running and running not in getattr(self, "_image_path_to_index", {}):
            running = str(Path(running))
        queued: set[str] = getattr(self, "_prefetch_queued_paths", set())
        for path in neighborhood:
            if path in {current, running} or path in queued:
                continue
            needs_source = not self._workspace.has_cached_source(path)
            needs_vectors = self._workspace.needs_vector_overlay(path)
            if not needs_source and not needs_vectors:
                continue
            self._enqueue_prefetch_frame(
                path,
                generation,
                vectors_only=bool(not needs_source and needs_vectors),
                neighborhood_set=neighborhood_set,
            )
