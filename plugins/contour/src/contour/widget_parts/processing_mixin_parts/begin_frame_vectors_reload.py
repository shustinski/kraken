"""processing_mixin: begin frame vectors reload."""
from __future__ import annotations
from time import perf_counter
from typing import Any
from ...infrastructure.frame_switch_profiler import (
    profile_callable,
)
from contour.widget_parts._imports import (
    FrameLoadPayload,
    FrameLoadRunnable,
    Path,
    PolygonData,
)


class BeginFrameVectorsReloadMixin:
    def _begin_frame_vectors_reload(self: Any, normalized_load_path: str) -> None:
        if self._frame_switch_profile_for_path(normalized_load_path) is None:
            self._start_frame_switch_profile(normalized_load_path)
        cif_path_for_profile = self._find_matching_cif_path(normalized_load_path)
        profile_timings: dict[str, float] = {}
        profile_total_start = perf_counter()
        self._frame_load_running_path = normalized_load_path
        self._frame_load_request_serial = int(getattr(self, "_frame_load_request_serial", 0)) + 1
        request_id = self._frame_load_request_serial

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
            load_source_image=None,
            load_cif_overlay=load_cif_overlay_timed,
            load_vectors=True,
            vectors_only=True,
            scan_repair=None,
        )

        def _on_vectors(req_id: int, payload_obj: object) -> None:
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
            image_result = self._workspace.apply_frame_vectors(
                payload.image_path,
                polygons=list(payload.polygons),
                loaded_cif_path=self._find_matching_cif_path(payload.image_path),
                polygons_needing_repair=self._payload_repair_reasons(payload),
            )
            profile_timings["workspace_apply_frame_vectors"] = (perf_counter() - apply_start) * 1000.0
            self._frame_load_running_path = None
            if image_result is not None and not superseded:
                self._finish_frame_load_ui(
                    image_result,
                    load_vectors=True,
                    profile_enabled=session is not None,
                    profile_timings=profile_timings,
                    profile_total_start=profile_total_start,
                    profiler=session.profiler if session is not None else None,
                    cif_path_for_profile=cif_path_for_profile,
                    phase_start=perf_counter(),
                )
            elif image_result is None and not superseded:
                self.load_image(payload.image_path, load_vectors=True)
            if getattr(self, "_loading_image_path", None) == payload.image_path:
                self._loading_image_path = None
            self._flush_pending_thumbnail_grid_rebuild()
            self._drain_pending_frame_load()

        def _on_vectors_error(req_id: int, image_path: str, message: str) -> None:
            if bool(getattr(self, "_closing", False)):
                return
            if req_id != self._frame_load_request_serial:
                return
            self._frame_load_running_path = None
            if getattr(self, "_loading_image_path", None) == image_path:
                self._loading_image_path = None
            self._append_log(self._tr("reload_with_cif_failed_log", error=message))
            profile_timings["total_wall"] = (perf_counter() - profile_total_start) * 1000.0
            self._emit_cif_open_profile(
                profile_timings,
                image_path=image_path,
                cif_path=cif_path_for_profile,
                polygon_count=0,
                profiler=None,
                vectors_only=True,
                failed=True,
            )
            self._flush_pending_thumbnail_grid_rebuild()
            self._drain_pending_frame_load()

        runnable.signals.result.connect(_on_vectors)
        runnable.signals.error.connect(_on_vectors_error)
        session = getattr(self, "_frame_switch_profile", None)
        if session is not None:
            session.disable_main_profiler()
        self._frame_load_thread_pool.start(runnable)
