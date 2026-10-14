"""processing_mixin: begin frame load."""
from __future__ import annotations
from time import perf_counter
from typing import Any
from ...infrastructure.frame_switch_profiler import (
    profile_callable,
)
from ...infrastructure.profiling import (
    write_profile_report,
)
from contour.widget_parts._imports import (
    FrameLoadPayload,
    FrameLoadRunnable,
    Path,
    PolygonData,
    QMessageBox,
)


class BeginFrameLoadMixin:
    def _begin_frame_load(self: Any, normalized_load_path: str, *, load_vectors: bool) -> None:
        session = self._frame_switch_profile_for_path(normalized_load_path)
        if session is None:
            session = self._start_frame_switch_profile(normalized_load_path)
        if session is not None:
            write_profile_report(
                f"[contour frame switch profiling] started image={Path(normalized_load_path).name} "
                "(runs until UI is interactive)",
            )
        cif_path_for_profile = self._find_matching_cif_path(normalized_load_path)
        profile_timings: dict[str, float] = {}
        profile_total_start = perf_counter()
        try:
            phase_start = perf_counter()
            self._abort_in_flight_interactive_processing(preview=True, prepared=True)
            profile_timings["abort_in_flight"] = (perf_counter() - phase_start) * 1000.0
            phase_start = perf_counter()

            sync_result = self._workspace.resolve_cached_load(normalized_load_path)
            if sync_result is not None:
                _unused_state = sync_result.state
                needs_vectors = load_vectors and self._workspace.needs_vector_overlay(
                    normalized_load_path
                )
                if needs_vectors:
                    # Hold the previous complete frame until CIF overlays arrive so
                    # image and vectors appear together (no empty-overlay flash).
                    self._loading_image_path = normalized_load_path
                    self._begin_frame_vectors_reload(normalized_load_path)
                    return
                session = getattr(self, "_frame_switch_profile", None)
                if session is not None:
                    session.enable_main_profiler()
                self._finish_frame_load_ui(
                    sync_result,
                    load_vectors=load_vectors,
                    profile_enabled=session is not None,
                    profile_timings=profile_timings,
                    profile_total_start=profile_total_start,
                    profiler=session.profiler if session is not None else None,
                    cif_path_for_profile=cif_path_for_profile,
                    phase_start=perf_counter(),
                )
                return

            self._frame_load_running_path = normalized_load_path
            self._frame_load_request_serial = int(getattr(self, "_frame_load_request_serial", 0)) + 1
            request_id = self._frame_load_request_serial

            def load_source_image_timed(image_path: str):
                inner_start = perf_counter()
                session = getattr(self, "_frame_switch_profile", None)

                def _load() -> object:
                    return self._load_scene_source_image_cached(image_path)

                try:
                    return profile_callable("worker_source_image", session, _load)
                finally:
                    profile_timings["source_image_load"] = (perf_counter() - inner_start) * 1000.0

            def load_cif_overlay_timed(image_path: str) -> list[PolygonData]:
                inner_start = perf_counter()
                session = getattr(self, "_frame_switch_profile", None)

                def _load() -> list[PolygonData]:
                    return self._load_cif_overlay_polygons(image_path)

                try:
                    return profile_callable("worker_cif_overlay", session, _load)
                finally:
                    profile_timings["cif_overlay_load"] = (perf_counter() - inner_start) * 1000.0

            runnable = FrameLoadRunnable(
                request_id,
                normalized_load_path,
                load_source_image=load_source_image_timed,
                load_cif_overlay=load_cif_overlay_timed,
                load_vectors=load_vectors,
                vectors_only=False,
                scan_repair=None,
            )

            def _on_result(req_id: int, payload_obj: object) -> None:
                if bool(getattr(self, "_closing", False)):
                    return
                if req_id != self._frame_load_request_serial:
                    return
                payload = payload_obj
                if not isinstance(payload, FrameLoadPayload):
                    return
                desired_path = str(Path(getattr(self, "_desired_image_path", "") or ""))
                superseded = bool(desired_path and desired_path != str(Path(payload.image_path)))
                session = self._frame_switch_profile_for_path(payload.image_path)
                if session is not None:
                    session.enable_main_profiler()
                apply_start = perf_counter()
                image_result = self._workspace.apply_loaded_frame(
                    payload.image_path,
                    source_image=payload.source_image,
                    polygons=list(payload.polygons),
                    make_current=not superseded,
                    loaded_cif_path=self._find_matching_cif_path(payload.image_path),
                    polygons_needing_repair=self._payload_repair_reasons(payload),
                )
                profile_timings["workspace_apply_loaded_frame"] = (perf_counter() - apply_start) * 1000.0
                self._frame_load_running_path = None
                if not superseded:
                    self._finish_frame_load_ui(
                        image_result,
                        load_vectors=load_vectors,
                        profile_enabled=session is not None,
                        profile_timings=profile_timings,
                        profile_total_start=profile_total_start,
                        profiler=session.profiler if session is not None else None,
                        cif_path_for_profile=cif_path_for_profile,
                        phase_start=perf_counter(),
                    )
                if getattr(self, "_loading_image_path", None) == payload.image_path:
                    self._loading_image_path = None
                self._resume_frame_matrix_thumbnail_loading()
                self._flush_pending_thumbnail_grid_rebuild()
                self._drain_pending_frame_load()

            def _on_error(req_id: int, image_path: str, message: str) -> None:
                if bool(getattr(self, "_closing", False)):
                    return
                if req_id != self._frame_load_request_serial:
                    return
                self._frame_load_running_path = None
                if getattr(self, "_loading_image_path", None) == image_path:
                    self._loading_image_path = None
                desired_path = str(Path(getattr(self, "_desired_image_path", "") or ""))
                superseded = bool(desired_path and desired_path != str(Path(image_path)))
                if not superseded:
                    self._append_log(self._tr("failed_to_load_image_log", image_path=image_path, error=message))
                failed_session = getattr(self, "_frame_switch_profile", None)
                if failed_session is not None:
                    failed_session.disable_main_profiler()
                    self._frame_switch_profile = None
                if not superseded:
                    QMessageBox.warning(self, self._tr("image_load_error_title"), message)
                self._resume_frame_matrix_thumbnail_loading()
                self._flush_pending_thumbnail_grid_rebuild()
                self._drain_pending_frame_load()

            runnable.signals.result.connect(_on_result)
            runnable.signals.error.connect(_on_error)
            session = getattr(self, "_frame_switch_profile", None)
            if session is not None:
                session.disable_main_profiler()
            self._frame_load_thread_pool.start(runnable)
        finally:
            pass
