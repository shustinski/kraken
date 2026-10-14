"""processing_mixin: emit cif open profile."""
from __future__ import annotations
import cProfile
from typing import Any
from ...infrastructure.profiling import (
    write_profile_report,
)
from contour.widget_parts._imports import (
    Path,
    PolygonData,
    QTimer,
)


class EmitCifOpenProfileMixin:
    def _emit_cif_open_profile(
        self,
        timings_ms: dict[str, float],
        *,
        image_path: str,
        cif_path: str | None,
        polygon_count: int,
        profiler: cProfile.Profile | None,
        vectors_only: bool = False,
        failed: bool = False,
    ) -> None:
        session = self._frame_switch_profile_for_path(image_path)
        if session is not None:
            session.merge_timings(timings_ms)
            if profiler is not None and profiler is not session.profiler:
                session.attach_worker_profile("ui_finish", profiler)
            self._schedule_frame_switch_profile_until_interactive(
                image_path,
                cif_path=cif_path,
                polygon_count=polygon_count,
                vectors_only=vectors_only,
                failed=failed,
            )
            return
        total_ms = timings_ms.get("total_wall", sum(timings_ms.values()))
        detail = " ".join(
            f"{name}={elapsed:.3f}ms" for name, elapsed in timings_ms.items() if name != "total_wall"
        )
        mode = "vector" if vectors_only else "image+vector"
        status = "failed" if failed else "ok"
        message = (
            f"[contour frame open profiling] mode={mode} status={status} total={total_ms:.3f}ms "
            f"polygons={polygon_count} image={Path(image_path).name} "
            f"cif={Path(cif_path).name if cif_path else '<none>'} {detail}"
        )
        write_profile_report(message)

    def _set_work_simulation_running(self: Any, running: bool) -> None:
        running = bool(running)
        if getattr(self, "_work_simulation_running", False) == running:
            return
        self._work_simulation_running = running
        if hasattr(self, "workSimulationActiveChanged"):
            self.workSimulationActiveChanged.emit(running)

    def _toggle_work_simulation(self: Any) -> None:
        if getattr(self, "_work_simulation_running", False):
            self._stop_work_simulation(restore_current=True)
            self._append_log("Симуляция остановлена." if self._ui_language == "ru" else "Work simulation stopped.")
            return
        self._start_work_simulation()

    def _start_work_simulation(self: Any) -> None:
        self._stop_work_simulation(restore_current=True)
        if self._workspace.current_state is not None and not self._try_leave_current_frame():
            return
        paths = [
            str(Path(path))
            for path in self._workspace.image_paths
            if self._workspace.resolve_cif_path(str(Path(path)))
        ]
        if not paths:
            self._append_log(
                "Нет кадров с изображением и вектором для симуляции."
                if self._ui_language == "ru"
                else "No image+vector frames are available for simulation."
            )
            return
        self._set_work_simulation_running(True)
        self._work_simulation_paths = paths
        self._work_simulation_path_index = -1
        self._work_simulation_target_polygons: list[PolygonData] = []
        self._work_simulation_visible_points = 0
        self._work_simulation_total_points = 0
        self._work_simulation_timer.setInterval(max(1, int(self._work_simulation_interval_ms)))
        self._advance_work_simulation()

    def _stop_work_simulation(self: Any, *, restore_current: bool = True) -> None:
        if hasattr(self, "_work_simulation_timer"):
            self._work_simulation_timer.stop()
        if restore_current and getattr(self, "_work_simulation_target_polygons", None):
            self._restore_work_simulation_frame()
        self._work_simulation_paths = []
        self._work_simulation_path_index = -1
        self._work_simulation_target_polygons = []
        self._work_simulation_visible_points = 0
        self._work_simulation_total_points = 0
        self._work_simulation_original_dirty = None
        self._work_simulation_original_reference_polygons: list[PolygonData] = []
        self._set_work_simulation_running(False)

    def _advance_work_simulation(self: Any) -> None:
        if not getattr(self, "_work_simulation_running", False):
            return
        if not getattr(self, "_work_simulation_target_polygons", None):
            self._begin_next_work_simulation_frame()
            return
        self._work_simulation_visible_points += 1
        if self._work_simulation_visible_points >= self._work_simulation_total_points:
            self._work_simulation_timer.stop()
            self._restore_work_simulation_frame()
            QTimer.singleShot(max(1, int(self._work_simulation_interval_ms)), self._begin_next_work_simulation_frame)
            return
        partial = self._partial_work_simulation_polygons(
            self._work_simulation_target_polygons,
            self._work_simulation_visible_points,
        )
        self._set_editor_polygons_for_work_simulation(partial)
        self._work_simulation_timer.start(max(1, int(self._work_simulation_interval_ms)))

    def _begin_next_work_simulation_frame(self: Any) -> None:
        if not getattr(self, "_work_simulation_running", False):
            return
        self._work_simulation_timer.stop()
        self._work_simulation_path_index += 1
        if self._work_simulation_path_index >= len(self._work_simulation_paths):
            # Restore once more at the transaction boundary.  Queued editor
            # notifications from the drawing animation must not leave the last
            # frame at an intermediate vertex count.
            self._stop_work_simulation(restore_current=True)
            self._append_log("Симуляция завершена." if self._ui_language == "ru" else "Work simulation finished.")
            return
        path = self._work_simulation_paths[self._work_simulation_path_index]
        if self._workspace.current_image_path != path:
            if self._image_path_in_image_list(path):
                self._set_image_list_current_path(path, fallback_to_first=False)
            else:
                self.load_image(path)
        state = self._workspace.current_state
        if state is None or self._workspace.current_image_path != path:
            QTimer.singleShot(0, self._begin_next_work_simulation_frame)
            return
        target = [polygon.clone() for polygon in state.polygons]
        if not target:
            QTimer.singleShot(0, self._begin_next_work_simulation_frame)
            return
        self._work_simulation_target_polygons = target
        self._work_simulation_visible_points = 0
        self._work_simulation_total_points = max(1, sum(len(polygon.points) for polygon in target))
        self._work_simulation_original_dirty = state.polygons_dirty
        self._work_simulation_original_reference_polygons = [polygon.clone() for polygon in state.reference_polygons]
        self._set_editor_polygons_for_work_simulation([])
        self._work_simulation_timer.start(max(1, int(self._work_simulation_interval_ms)))

    def _set_editor_polygons_for_work_simulation(self: Any, polygons: list[PolygonData]) -> None:
        self._updating_views = True
        try:
            self.polygon_editor.set_polygons([polygon.clone() for polygon in polygons])
        finally:
            self._updating_views = False

    def _partial_work_simulation_polygons(
        self,
        polygons: list[PolygonData],
        visible_points: int,
    ) -> list[PolygonData]:
        remaining = max(0, int(visible_points))
        partial: list[PolygonData] = []
        for polygon in polygons:
            if remaining <= 0:
                break
            point_count = min(len(polygon.points), remaining)
            if point_count > 0:
                clone = polygon.clone()
                clone.points = list(clone.points[:point_count])
                partial.append(clone)
            remaining -= len(polygon.points)
        return partial

    def _restore_work_simulation_frame(self: Any) -> None:
        target = [polygon.clone() for polygon in getattr(self, "_work_simulation_target_polygons", [])]
        if not target:
            return
        state = self._workspace.current_state
        if state is not None:
            state.polygons = [polygon.clone() for polygon in target]
            state.reference_polygons = [
                polygon.clone() for polygon in getattr(self, "_work_simulation_original_reference_polygons", [])
            ]
            state.polygons_dirty = getattr(self, "_work_simulation_original_dirty", None)
        self._set_editor_polygons_for_work_simulation(target)
        self._update_frame_item_status(self._workspace.current_image_path)
        self._update_vector_edit_status_label()

    def _is_extraction_mode_enabled(self: Any) -> bool:
        if not hasattr(self, "recognition_mode_combo"):
            return False
        return str(self.recognition_mode_combo.currentData() or "") != "disabled"

    def get_polygons(self: Any) -> list[PolygonData]:
        return self.polygon_editor.get_polygons()
