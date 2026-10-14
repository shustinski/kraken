"""processing_mixin: contact preview profile."""
from __future__ import annotations
import cProfile
from typing import Any
from ...infrastructure.contact_placement_profiler import (
    ContactPlacementProfile,
)
from ...infrastructure.profiling import (
    contact_deletion_profiling_enabled,
    contact_placement_profiling_enabled,
    write_profile_report,
)
from contour.widget_parts._imports import (
    LARGE_FRAME_COUNT_THRESHOLD,
    Path,
    QListWidgetItem,
    QTimer,
    SaveOptions,
    paint_image_row_item,
)


class ContactPreviewProfileMixin:
    def _finish_contact_multi_selection_profile(self: Any, contact_count: int) -> None:
        session = getattr(self, "_contact_multi_selection_profile", None)
        if session is None:
            return
        self._contact_multi_selection_profile = None
        session.contact_count = max(0, int(contact_count))
        applied_at = session.elapsed_ms()
        gesture_ms = session.timings_ms.get("selection_gesture")
        if gesture_ms is not None:
            session.timings_ms["selection_apply"] = max(
                0.0,
                applied_at - gesture_ms,
            )
        session.timings_ms["selection_applied_to_ui"] = applied_at
        session.stop()
        if int(contact_count) < 2:
            return
        self._emit_contact_profile_output(
            session.format_summary(status=f"selected_{int(contact_count)}")
        )
        self._emit_contact_profile_output(session.format_stats())

    def _start_contact_deletion_profile(self: Any, contact_count: int) -> None:
        if not contact_deletion_profiling_enabled() or int(contact_count) <= 0:
            return
        if getattr(self, "_contact_placement_profile", None) is not None:
            self._finish_contact_placement_profile("replaced_by_deletion")
        previous = getattr(self, "_contact_deletion_profile", None)
        if previous is not None:
            self._finish_contact_deletion_profile("superseded")
        self._contact_deletion_profile = ContactPlacementProfile.begin(action="deletion")
        self._contact_deletion_profile.contact_count = int(contact_count)
        self._emit_contact_profile_output(
            f"[contour contact deletion profiling] started contacts={int(contact_count)}"
        )

    def _on_contact_deletion_finished(self: Any, contact_count: int) -> None:
        session = getattr(self, "_contact_deletion_profile", None)
        if session is None:
            return
        session.note("deletion_applied_to_ui")
        if int(contact_count) <= 0:
            self._finish_contact_deletion_profile("rejected")
        elif not session.waiting_for_preview:
            QTimer.singleShot(
                0,
                lambda: self._finish_contact_deletion_profile("displayed"),
            )

    def _finish_contact_deletion_profile(self: Any, status: str) -> None:
        session = getattr(self, "_contact_deletion_profile", None)
        if session is None:
            return
        self._contact_deletion_profile = None
        session.stop()
        self._emit_contact_profile_output(session.format_summary(status=status))
        self._emit_contact_profile_output(session.format_stats())

    def _start_contact_placement_profile(self: Any) -> None:
        if not contact_placement_profiling_enabled():
            return
        previous = getattr(self, "_contact_placement_profile", None)
        if previous is not None:
            self._finish_contact_placement_profile("superseded")
        self._contact_placement_profile = ContactPlacementProfile.begin()
        self._emit_contact_profile_output(
            "[contour contact placement profiling] started at contact hotkey"
        )

    def _ensure_contact_placement_profile(self: Any) -> None:
        if getattr(self, "_contact_placement_profile", None) is not None:
            return
        if not contact_placement_profiling_enabled():
            return
        self._contact_placement_profile = ContactPlacementProfile.begin()
        self._emit_contact_profile_output(
            "[contour contact placement profiling] started at contact placement"
        )

    def _emit_contact_profile_output(self: Any, message: str) -> None:
        write_profile_report(message)

    def _on_contact_placement_attempt_finished(self: Any, added: bool) -> None:
        session = getattr(self, "_contact_placement_profile", None)
        if session is None:
            return
        session.note("contact_attempt")
        if not added:
            self._finish_contact_placement_profile("rejected")

    def _finish_contact_placement_profile(self: Any, status: str) -> None:
        session = getattr(self, "_contact_placement_profile", None)
        if session is None:
            return
        self._contact_placement_profile = None
        session.stop()
        self._emit_contact_profile_output(session.format_summary(status=status))
        self._emit_contact_profile_output(session.format_stats())

    def _on_contact_preview_profile(
        self: Any,
        request_id: int,
        profiler: cProfile.Profile,
        wall_ms: float,
    ) -> None:
        for attribute in (
            "_contact_deletion_profile",
            "_contact_placement_profile",
        ):
            session = getattr(self, attribute, None)
            if session is not None and session.preview_request_id == request_id:
                session.attach_worker(profiler, wall_ms)
                session.resume_main_profiler()
                break
        recognition_session = getattr(self, "_image_recognition_profile", None)
        if (
            recognition_session is not None
            and recognition_session.preview_request_id == request_id
        ):
            recognition_session.attach_worker(profiler, wall_ms)
            recognition_session.resume_main_profiler()

    def _contact_profile_waiting_for_preview(self: Any):
        for attribute in (
            "_contact_deletion_profile",
            "_contact_placement_profile",
        ):
            session = getattr(self, attribute, None)
            if session is not None and session.waiting_for_preview:
                return session
        return None

    def _resume_preview_main_profiles(self: Any, request_id: int) -> None:
        for attribute in (
            "_contact_deletion_profile",
            "_contact_placement_profile",
            "_image_recognition_profile",
        ):
            session = getattr(self, attribute, None)
            if session is not None and session.preview_request_id == request_id:
                session.resume_main_profiler()

    def _current_save_options(self: Any) -> SaveOptions:
        return SaveOptions(
            save_cif=self.save_cif_checkbox.isChecked(),
            save_cv=self.save_cv_checkbox.isChecked(),
            save_preview=self.save_preview_checkbox.isChecked(),
        )

    def _paint_image_row_item(self: Any, item: QListWidgetItem, image_path: str, *, show_text: bool = True) -> None:
        normalized = str(Path(image_path))
        extraction_enabled = self._is_extraction_mode_enabled()
        paint_image_row_item(
            item,
            normalized,
            image_has_changes=(
                self._workspace.image_has_changes(normalized)
                or normalized in self._restored_modified_image_paths
            ),
            has_vector_overlay=bool(self._workspace.resolve_cif_path(normalized)),
            vector_index_active=self._vector_index_active(),
            extraction_enabled=extraction_enabled,
            viewed=normalized in self._viewed_image_paths,
            persisted_highlight=normalized in self._persisted_highlight_paths,
            theme=getattr(self, "_ui_theme", "dark"),
            show_text=show_text,
        )

    def _uses_large_frame_list(self: Any) -> bool:
        image_paths = getattr(self._workspace, "_image_paths", None)
        if image_paths is None:
            image_paths = self._workspace.image_paths
        return len(image_paths) > LARGE_FRAME_COUNT_THRESHOLD

    def _image_path_index(self: Any, image_path: str | Path) -> int | None:
        return self._image_path_to_index.get(str(Path(image_path)))

    def _ordered_workspace_image_paths(self: Any) -> list[str]:
        """Return normalized image paths in list order without re-parsing pathlib."""

        ordered = getattr(self, "_image_paths_ordered", None)
        if ordered is not None:
            return ordered
        mapping = getattr(self, "_image_path_to_index", None)
        if mapping:
            rebuilt = [""] * len(mapping)
            for path, index in mapping.items():
                rebuilt[index] = path
            self._image_paths_ordered = rebuilt
            return rebuilt
        workspace_paths = getattr(self._workspace, "_image_paths", None)
        if workspace_paths is not None:
            return list(workspace_paths)
        return list(self._workspace.image_paths)
