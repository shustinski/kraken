"""processing_mixin: queue neighbor frame load."""
from __future__ import annotations
from typing import Any
from ...adapters.qt.signal_connection import connect_queued
from contour.widget_parts._imports import (
    Path,
    QTimer,
    ThumbnailLoadRunnable,
    WorkspaceLoadResult,
)


class QueueNeighborFrameLoadMixin:
    def _sync_current_state_views(
        self: Any,
        *,
        defer_neighbors: bool = True,
        preserve_view: bool = False,
        sync_neighbors: bool = True,
    ) -> None:
        self._updating_views = True
        try:
            self._apply_frame_to_editor(
                defer_neighbors=defer_neighbors,
                defer_heavy_overlays=False,
                preserve_view=preserve_view,
                sync_neighbors=sync_neighbors,
            )
        finally:
            self._updating_views = False

    def _display_image_for_current_state(self: Any):
        current_state = self._workspace.current_state
        if self._source_preview_active() and current_state is not None and current_state.source_image is not None:
            return current_state.source_image
        return self._workspace.current_display_image()

    def _preprocessed_matches_current_pipeline(self: Any, state) -> bool:
        return (
            state is not None
            and state.preprocessed_image is not None
            and state.pipeline_config == self.get_pipeline()
        )

    def _clear_stale_preprocessed_image_for_current_pipeline(self: Any, state) -> None:
        if state is None or state.preprocessed_image is None:
            return
        if state.pipeline_config == self.get_pipeline():
            return
        state.preprocessed_image = None
        state.pipeline_config = None
        state.mask_image = None

    def _prepared_image_update_required(self: Any, image_result: WorkspaceLoadResult) -> bool:
        state = image_result.state
        return (
            state is not None
            and state.source_image is not None
            and not image_result.vectors_only
            and not self._preprocessed_matches_current_pipeline(state)
        )

    def _via_debug_inspection_enabled(self: Any) -> bool:
        return bool(
            hasattr(self, "recognition_mode_combo")
            and str(self.recognition_mode_combo.currentData() or "") in {"via", "conductors"}
        )

    def _neighbor_preview_max_dimension(self: Any) -> int:
        state = self._workspace.current_state
        if state is not None and state.source_image is not None:
            height, width = state.source_image.shape[:2]
            return max(128, min(512, int(min(width, height))))
        return 256

    def _load_scene_source_image_cached(self: Any, image_path: str):
        from contour.widget_parts.processing_mixin import load_image_color

        normalized = str(Path(image_path))
        cache = self._scene_source_image_cache
        with self._scene_source_image_cache_lock:
            cached = cache.get(normalized)
            if cached is not None:
                cache.move_to_end(normalized)
                return cached
            loader = getattr(self, "_source_image_loader", None) or load_image_color
            image = loader(normalized)
            previous = cache.pop(normalized, None)
            if previous is not None:
                self._scene_source_image_cache_bytes -= int(previous.nbytes)
            cache[normalized] = image
            self._scene_source_image_cache_bytes += int(image.nbytes)
            cache_limit = max(
                1,
                int(getattr(self, "_scene_source_image_cache_max_bytes", 512 * 1024 * 1024)),
            )
            while (
                self._scene_source_image_cache_bytes > cache_limit
                and len(cache) > 1
            ):
                _evicted_path, evicted = cache.popitem(last=False)
                self._scene_source_image_cache_bytes -= int(evicted.nbytes)
            return image

    def _neighbor_frame_image(self: Any, image_path: str):
        normalized = str(Path(image_path))
        state = getattr(self._workspace, "_state_cache", {}).get(normalized)
        if state is not None:
            return state.preprocessed_image if state.preprocessed_image is not None else state.source_image
        cached = self._neighbor_image_cache.get(normalized)
        if cached is not None:
            if hasattr(self._neighbor_image_cache, "move_to_end"):
                self._neighbor_image_cache.move_to_end(normalized)
            return cached
        return None

    def _queue_neighbor_frame_load(self: Any, image_path: str) -> None:
        normalized = str(Path(image_path))
        if normalized in getattr(self, "_neighbor_queued_paths", set()):
            return
        self._neighbor_queued_paths.add(normalized)
        try:
            max_dim = self._neighbor_preview_max_dimension()
            runnable = ThumbnailLoadRunnable(
                0,
                normalized,
                max_dim,
                max_dim,
                source_image_loader=self._load_scene_source_image_cached,
            )
            connect_queued(
                runnable.signals.result,
                lambda generation, image_path, width, height, qimage: self._on_neighbor_frame_loaded(
                    generation,
                    image_path,
                    width,
                    height,
                    qimage,
                ),
            )
            connect_queued(
                runnable.signals.finished,
                self._on_neighbor_frame_load_finished,
            )
            self._neighbor_thread_pool.start(runnable)
        except RuntimeError:
            self._neighbor_queued_paths.discard(normalized)

    def _on_neighbor_frame_loaded(
        self: Any,
        _generation: int,
        image_path: str,
        _width: int,
        _height: int,
        qimage: object,
    ) -> None:
        normalized = str(Path(image_path))
        self._neighbor_queued_paths.discard(normalized)
        if not self._neighbor_sync_is_current():
            return
        if not any(str(Path(path)) == normalized for *_offsets, path in getattr(self, "_neighbor_frame_specs", [])):
            return
        if qimage is None:
            return
        self._neighbor_image_cache[normalized] = qimage
        if hasattr(self._neighbor_image_cache, "move_to_end"):
            self._neighbor_image_cache.move_to_end(normalized)
        cache_limit = max(1, int(getattr(self, "_neighbor_image_cache_limit", 96)))
        while len(self._neighbor_image_cache) > cache_limit:
            if hasattr(self._neighbor_image_cache, "move_to_end"):
                evicted_path, _evicted_image = self._neighbor_image_cache.popitem(last=False)
            else:
                evicted_path = next(iter(self._neighbor_image_cache))
                self._neighbor_image_cache.pop(evicted_path, None)
            self._neighbor_image_dimensions.pop(str(evicted_path), None)
        self._schedule_neighbor_frame_apply(delay_ms=0)

    def _on_neighbor_frame_load_finished(self: Any, _generation: int, image_path: str) -> None:
        self._neighbor_queued_paths.discard(str(Path(image_path)))
        if self._neighbor_sync_is_current() and not getattr(self, "_neighbor_queued_paths", set()):
            self._schedule_neighbor_frame_apply(delay_ms=0)

    def _schedule_neighbor_frame_apply(self: Any, *, delay_ms: int = 0) -> None:
        if not self._neighbor_sync_is_current():
            return
        timer = getattr(self, "_neighbor_apply_timer", None)
        if timer is None:
            QTimer.singleShot(max(0, int(delay_ms)), self._apply_cached_neighbor_frames)
            return
        timer.stop()
        timer.start(max(0, int(delay_ms)))

    def _try_load_neighbor_preview_image(self: Any, image_path: str):
        return self._neighbor_frame_image(str(image_path))
