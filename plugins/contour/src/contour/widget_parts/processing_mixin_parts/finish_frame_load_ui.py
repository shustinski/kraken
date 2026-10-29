"""processing_mixin: finish frame load ui."""
from __future__ import annotations
import cProfile
from time import perf_counter
from typing import Any
from contour.widget_parts._imports import (
    Path,
    RewardEventType,
    WorkspaceLoadResult,
)


class FinishFrameLoadUiMixin:
    def _finish_frame_load_ui(
        self: Any,
        image_result: WorkspaceLoadResult,
        *,
        load_vectors: bool,
        profile_enabled: bool = False,
        profile_timings: dict[str, float] | None = None,
        profile_total_start: float = 0.0,
        profiler: cProfile.Profile | None = None,
        cif_path_for_profile: str | None = None,
        phase_start: float = 0.0,
    ) -> None:
        profile_timings = profile_timings or {}
        phase_start = phase_start or perf_counter()
        self._show_produced_recognition_for_current_file(image_result.image_path)
        preserve_editor_view_position = self._should_preserve_editor_view_position(
            image_result.image_path
        )
        self._save_persisted_current_image_path(image_result.image_path)
        if not self._is_extraction_mode_enabled():
            self._viewed_image_paths.add(str(Path(image_result.image_path)))
            self._handle_gamification_ui_event(RewardEventType.IMAGE_VIEWED)
        if (
            image_result.state is not None
            and not image_result.cache_hit
            and not image_result.reused_current_state
            and not image_result.vectors_only
        ) or (image_result.state is not None and image_result.vectors_only):
            image_result.state.loaded_cif_path = self._find_matching_cif_path(image_result.image_path)
            image_result.state.reference_polygons = [polygon.clone() for polygon in image_result.state.polygons]
            image_result.state.polygons_dirty = False
        elif image_result.state is not None and not image_result.state.loaded_cif_path:
            # Heal cached frames that kept polygons but lost the CIF target path.
            image_result.state.loaded_cif_path = self._find_matching_cif_path(image_result.image_path)
        if image_result.state is not None and not image_result.vectors_only:
            self._clear_stale_preprocessed_image_for_current_pipeline(image_result.state)
        if (
            image_result.reused_current_state
            and getattr(self, "_last_editor_display_path", None) == image_result.image_path
        ):
            step_start = perf_counter()
            self._defer_frame_chrome_updates(image_result.image_path)
            if profile_enabled:
                profile_timings["defer_frame_chrome"] = (perf_counter() - step_start) * 1000.0
            if image_result.state is not None and self._prepared_image_update_required(image_result):
                self._queue_prepared_image_update(image_result.image_path, image_result.state.source_image)
            step_start = perf_counter()
            self._center_editor_on_current_main_image(force=False)
            if profile_enabled:
                profile_timings["center_main_image"] = (perf_counter() - step_start) * 1000.0
            if getattr(self, "_loading_image_path", None) == image_result.image_path:
                self._loading_image_path = None
            if image_result.state is not None and hasattr(self, "polygon_editor"):
                self._sync_polygons_to_editor(
                    image_result.image_path,
                    self._polygons_visible_in_editor(
                        image_result.state,
                        list(image_result.state.polygons),
                    ),
                )
            # Offer after deferred geometry validation; never scan on the UI switch path.
            self._schedule_deferred_geometry_validation(image_result.image_path)
            step_start = perf_counter()
            self._resume_frame_matrix_thumbnail_loading()
            if profile_enabled:
                profile_timings["resume_frame_matrix_thumbnails"] = (perf_counter() - step_start) * 1000.0
            if profile_enabled:
                profile_timings["sync_reused"] = (perf_counter() - phase_start) * 1000.0
                self._emit_cif_open_profile(
                    profile_timings,
                    image_path=image_result.image_path,
                    cif_path=cif_path_for_profile,
                    polygon_count=0 if image_result.state is None else len(image_result.state.polygons),
                    profiler=profiler,
                    vectors_only=image_result.vectors_only,
                )
            self._schedule_frame_prefetch()
            return
        if image_result.vectors_only:
            self._updating_views = True
            try:
                step_start = perf_counter()
                self._apply_frame_to_editor(
                    defer_neighbors=True,
                    defer_heavy_overlays=True,
                    preserve_view=preserve_editor_view_position,
                    clear_neighbors=True,
                    sync_neighbors=False,
                )
                if profile_enabled:
                    profile_timings["apply_frame_to_editor"] = (perf_counter() - step_start) * 1000.0
            finally:
                self._updating_views = False
            step_start = perf_counter()
            self._center_editor_on_current_main_image(force=True)
            if profile_enabled:
                profile_timings["center_main_image"] = (perf_counter() - step_start) * 1000.0
            if load_vectors:
                self._schedule_deferred_geometry_validation(image_result.image_path)
            step_start = perf_counter()
            self._request_neighbor_frame_sync(delay_ms=0)
            if profile_enabled:
                profile_timings["sync_neighbor_frames"] = (perf_counter() - step_start) * 1000.0
            step_start = perf_counter()
            self._defer_frame_chrome_updates(image_result.image_path)
            if profile_enabled:
                profile_timings["defer_frame_chrome"] = (perf_counter() - step_start) * 1000.0
            step_start = perf_counter()
            self._update_vector_edit_status_label()
            if profile_enabled:
                profile_timings["update_vector_status"] = (perf_counter() - step_start) * 1000.0
        elif image_result.cache_hit or image_result.reused_current_state:
            self._updating_views = True
            try:
                step_start = perf_counter()
                self._apply_frame_to_editor(
                    defer_neighbors=True,
                    defer_heavy_overlays=True,
                    preserve_view=preserve_editor_view_position,
                    clear_neighbors=True,
                    sync_neighbors=False,
                )
                if profile_enabled:
                    profile_timings["apply_frame_to_editor"] = (perf_counter() - step_start) * 1000.0
            finally:
                self._updating_views = False
            step_start = perf_counter()
            self._center_editor_on_current_main_image(force=True)
            if profile_enabled:
                profile_timings["center_main_image"] = (perf_counter() - step_start) * 1000.0
            if load_vectors:
                self._schedule_deferred_geometry_validation(image_result.image_path)
            step_start = perf_counter()
            self._request_neighbor_frame_sync(delay_ms=0)
            if profile_enabled:
                profile_timings["sync_neighbor_frames"] = (perf_counter() - step_start) * 1000.0
            step_start = perf_counter()
            self._defer_frame_chrome_updates(image_result.image_path)
            if profile_enabled:
                profile_timings["defer_frame_chrome"] = (perf_counter() - step_start) * 1000.0
        else:
            self._updating_views = True
            try:
                step_start = perf_counter()
                self._apply_frame_to_editor(
                    defer_neighbors=True,
                    defer_heavy_overlays=True,
                    preserve_view=preserve_editor_view_position,
                    clear_neighbors=True,
                    sync_neighbors=False,
                )
                if profile_enabled:
                    profile_timings["apply_frame_to_editor"] = (perf_counter() - step_start) * 1000.0
            finally:
                self._updating_views = False
            step_start = perf_counter()
            self._center_editor_on_current_main_image(force=True)
            if profile_enabled:
                profile_timings["center_main_image"] = (perf_counter() - step_start) * 1000.0
            if load_vectors:
                self._schedule_deferred_geometry_validation(image_result.image_path)
            step_start = perf_counter()
            self._request_neighbor_frame_sync(delay_ms=0)
            if profile_enabled:
                profile_timings["sync_neighbor_frames"] = (perf_counter() - step_start) * 1000.0
            step_start = perf_counter()
            self._defer_frame_chrome_updates(image_result.image_path)
            if profile_enabled:
                profile_timings["defer_frame_chrome"] = (perf_counter() - step_start) * 1000.0
        if getattr(self, "_loading_image_path", None) == image_result.image_path:
            self._loading_image_path = None
        step_start = perf_counter()
        self._resume_frame_matrix_thumbnail_loading()
        if profile_enabled:
            profile_timings["resume_frame_matrix_thumbnails"] = (perf_counter() - step_start) * 1000.0
        if bool(getattr(self, "_pending_thumbnail_rebuild_after_vectors", False)):
            step_start = perf_counter()
            self._flush_pending_thumbnail_grid_rebuild()
            if profile_enabled:
                profile_timings["flush_pending_thumbnail_rebuild"] = (perf_counter() - step_start) * 1000.0
        if profile_enabled:
            profile_timings["sync_views"] = (perf_counter() - phase_start) * 1000.0
            phase_start = perf_counter()
        if image_result.state is not None and self._prepared_image_update_required(image_result):
            self._queue_prepared_image_update(image_result.image_path, image_result.state.source_image)
        if profile_enabled:
            profile_timings["queue_prepared"] = (perf_counter() - phase_start) * 1000.0
            phase_start = perf_counter()
        if image_result.cache_hit and not image_result.vectors_only:
            self._append_log(self._tr("loaded_cached_state_log", image_path=image_result.image_path))
        elif not image_result.vectors_only:
            self._append_log(self._tr("loaded_image_log", image_path=image_result.image_path))
        if profile_enabled:
            profile_timings["log"] = (perf_counter() - phase_start) * 1000.0
            phase_start = perf_counter()
        if load_vectors and not image_result.vectors_only:
            profile_session = self._frame_switch_profile_for_path(image_result.image_path)
            if profile_session is not None:
                profile_session.disable_main_profiler()
            self._try_extract_if_recognition_enabled()
        if profile_enabled:
            profile_timings["maybe_extract"] = (perf_counter() - phase_start) * 1000.0
            profile_session = self._frame_switch_profile_for_path(image_result.image_path)
            if profile_session is not None:
                profile_session.enable_main_profiler()
            self._emit_cif_open_profile(
                profile_timings,
                image_path=image_result.image_path,
                cif_path=cif_path_for_profile,
                polygon_count=0 if image_result.state is None else len(image_result.state.polygons),
                profiler=profiler,
                vectors_only=image_result.vectors_only,
            )
        self._schedule_frame_prefetch()
