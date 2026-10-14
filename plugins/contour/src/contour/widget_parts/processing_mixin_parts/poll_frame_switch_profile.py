"""processing_mixin: poll frame switch profile."""
from __future__ import annotations
from typing import Any
from ...infrastructure.frame_switch_profiler import (
    MAX_IDLE_POLLS,
    FrameSwitchProfile,
    frame_switch_profiling_enabled,
)
from ...infrastructure.profiling import (
    write_profile_report,
)
from contour.widget_parts._imports import (
    Path,
    QApplication,
    QTimer,
)


class PollFrameSwitchProfileMixin:
    def _frame_switch_profiling_active(self: Any) -> bool:
        return frame_switch_profiling_enabled()

    def _start_frame_switch_profile(self: Any, image_path: str) -> FrameSwitchProfile | None:
        if not self._frame_switch_profiling_active():
            return None
        self._frame_switch_profile_generation = int(getattr(self, "_frame_switch_profile_generation", 0)) + 1
        session = FrameSwitchProfile.begin(
            image_path,
            generation=self._frame_switch_profile_generation,
        )
        self._frame_switch_profile = session
        return session

    def _frame_switch_profile_for_path(self: Any, image_path: str) -> FrameSwitchProfile | None:
        session = getattr(self, "_frame_switch_profile", None)
        if session is None:
            return None
        if str(Path(session.image_path)) != str(Path(image_path)):
            return None
        return session

    def _frame_switch_profile_is_interactive(self: Any, image_path: str) -> bool:
        normalized = str(Path(image_path))
        if str(Path(self._workspace.current_image_path or "")) != normalized:
            return False
        if getattr(self, "_loading_image_path", None) is not None:
            return False
        if getattr(self, "_frame_load_running_path", None) is not None:
            return False
        if getattr(self, "_pending_editor_frame_apply", None) is not None:
            return False
        if getattr(self, "_thumbnail_rebuild_in_progress", False):
            return False
        if self._editor_display_thread_pool.activeThreadCount() > 0:
            return False
        chrome_timer = getattr(self, "_frame_chrome_update_timer", None)
        if chrome_timer is not None and chrome_timer.isActive():
            return False
        neighbor_timer = getattr(self, "_neighbor_sync_timer", None)
        if neighbor_timer is not None and neighbor_timer.isActive():
            return False
        session = self._frame_switch_profile_for_path(normalized)
        return not (session is not None and session.pending_since)

    def _schedule_frame_switch_profile_until_interactive(
        self: Any,
        image_path: str,
        *,
        cif_path: str | None,
        polygon_count: int,
        vectors_only: bool = False,
        failed: bool = False,
    ) -> None:
        session = self._frame_switch_profile_for_path(image_path)
        if session is None:
            return
        generation = session.generation
        QTimer.singleShot(
            0,
            lambda: self._poll_frame_switch_profile_until_interactive(
                image_path,
                generation,
                cif_path=cif_path,
                polygon_count=polygon_count,
                vectors_only=vectors_only,
                failed=failed,
            ),
        )

    def _poll_frame_switch_profile_until_interactive(
        self: Any,
        image_path: str,
        generation: int,
        *,
        cif_path: str | None,
        polygon_count: int,
        vectors_only: bool,
        failed: bool,
    ) -> None:
        session = getattr(self, "_frame_switch_profile", None)
        if session is None or session.generation != generation:
            return
        session.enable_main_profiler()
        session.poll_count += 1
        interactive = failed or self._frame_switch_profile_is_interactive(image_path)
        if not interactive and session.poll_count < MAX_IDLE_POLLS:
            QTimer.singleShot(
                0,
                lambda: self._poll_frame_switch_profile_until_interactive(
                    image_path,
                    generation,
                    cif_path=cif_path,
                    polygon_count=polygon_count,
                    vectors_only=vectors_only,
                    failed=failed,
                ),
            )
            return
        if not session.claim_finalization():
            return
        if getattr(self, "_frame_switch_profile", None) is session:
            self._frame_switch_profile = None
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        self._finalize_frame_switch_profile(
            session,
            cif_path=cif_path,
            polygon_count=polygon_count,
            vectors_only=vectors_only,
            failed=failed,
            interactive=interactive and not failed,
        )

    def _finalize_frame_switch_profile(
        self: Any,
        session: FrameSwitchProfile,
        *,
        cif_path: str | None,
        polygon_count: int,
        vectors_only: bool,
        failed: bool,
        interactive: bool,
    ) -> None:
        session.timings_ms["total_wall"] = session.total_wall_ms()
        main_profiler = session.disable_main_profiler()
        summary = session.format_summary(
            polygon_count=polygon_count,
            cif_path=cif_path,
            vectors_only=vectors_only,
            failed=failed,
            interactive=interactive,
        )
        reports = [summary]
        if session.profiling_active or main_profiler.getstats():
            reports.append(session.format_stats(main_profiler, title="main_thread"))
        elif session.main_stats_skipped:
            reports.append(
                "[contour frame switch profiling stats] main_thread skipped "
                "(another cProfile session was active, e.g. contour processing extract)",
            )
        for label, worker_profiler in session.worker_profilers:
            reports.append(session.format_stats(worker_profiler, title=f"worker_{label}"))
        write_profile_report(*reports)
        if getattr(self, "_prefetch_deferred_for_profile", False):
            self._prefetch_deferred_for_profile = False
            QTimer.singleShot(0, self._schedule_frame_prefetch)
        pending_validation = getattr(self, "_pending_geometry_validation_path", None)
        if pending_validation:
            self._pending_geometry_validation_path = None
            QTimer.singleShot(0, lambda: self._start_deferred_geometry_validation(pending_validation))
        if hasattr(self, "polygon_editor") and hasattr(self.polygon_editor, "flush_deferred_object_colors"):
            QTimer.singleShot(0, self.polygon_editor.flush_deferred_object_colors)
        QTimer.singleShot(100, self._resume_frame_matrix_thumbnail_loading)
