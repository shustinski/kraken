"""processing_mixin: start image recognition profile."""
from __future__ import annotations
from typing import Any
from ...infrastructure.contact_placement_profiler import (
    ContactPlacementProfile,
    ImageRecognitionProfile,
)
from ...infrastructure.profiling import (
    contact_copy_profiling_enabled,
    contact_multi_selection_profiling_enabled,
    contact_paste_profiling_enabled,
    contact_redo_profiling_enabled,
    contact_undo_profiling_enabled,
    image_recognition_profiling_enabled,
    write_profile_report,
)
from contour.widget_parts._imports import (
    Path,
    PreviewProcessingRequest,
    QTimer,
    normalize_recognition_mode,
)


class StartImageRecognitionProfileMixin:
    def _start_image_recognition_profile(
        self: Any,
        request: PreviewProcessingRequest,
    ) -> None:
        if not image_recognition_profiling_enabled():
            return
        recognition_mode = normalize_recognition_mode(
            request.contour_settings.recognition_mode
        )
        if recognition_mode == "disabled":
            return
        previous = getattr(self, "_image_recognition_profile", None)
        if previous is not None:
            self._finish_image_recognition_profile("superseded")
        session = ImageRecognitionProfile.begin(
            image_path=Path(request.image_path).name,
            recognition_mode=recognition_mode,
        )
        session.note("queued")
        self._image_recognition_profile = session
        write_profile_report(
            "[contour image recognition profiling] "
            f"started mode={recognition_mode} "
            f"image={Path(request.image_path).name!r}",
        )

    def _finish_image_recognition_profile(self: Any, status: str) -> None:
        session = getattr(self, "_image_recognition_profile", None)
        if session is None:
            return
        self._image_recognition_profile = None
        session.stop()
        write_profile_report(session.format_summary(status=status), session.format_stats())

    def _start_contact_undo_profile(self: Any) -> None:
        self._start_contact_history_profile("undo")

    def _finish_contact_undo_profile(self: Any, applied: bool, contact_count: int) -> None:
        self._finish_contact_history_profile("undo", applied, contact_count)

    def _start_contact_redo_profile(self: Any) -> None:
        self._start_contact_history_profile("redo")

    def _finish_contact_redo_profile(self: Any, applied: bool, contact_count: int) -> None:
        self._finish_contact_history_profile("redo", applied, contact_count)

    def _start_contact_history_profile(self: Any, action: str) -> None:
        enabled = (
            contact_undo_profiling_enabled()
            if action == "undo"
            else contact_redo_profiling_enabled()
        )
        if not enabled:
            return
        attribute = f"_contact_{action}_profile"
        previous = getattr(self, attribute, None)
        if previous is not None:
            previous.stop()
        setattr(self, attribute, ContactPlacementProfile.begin(action=action))
        self._emit_contact_profile_output(
            f"[contour contact {action} profiling] started"
        )

    def _finish_contact_history_profile(
        self: Any,
        action: str,
        applied: bool,
        contact_count: int,
    ) -> None:
        attribute = f"_contact_{action}_profile"
        session = getattr(self, attribute, None)
        if session is None:
            return
        session.contact_count = max(0, int(contact_count))
        session.note(f"{action}_applied_to_ui")
        status = "displayed" if applied else "rejected"
        QTimer.singleShot(
            0,
            lambda: self._complete_contact_history_profile(action, status, session),
        )

    def _complete_contact_history_profile(
        self: Any,
        action: str,
        status: str,
        expected_session: ContactPlacementProfile,
    ) -> None:
        attribute = f"_contact_{action}_profile"
        session = getattr(self, attribute, None)
        if session is not expected_session:
            return
        setattr(self, attribute, None)
        session.stop()
        self._emit_contact_profile_output(session.format_summary(status=status))
        self._emit_contact_profile_output(session.format_stats())

    def _start_contact_copy_profile(self: Any) -> None:
        if not contact_copy_profiling_enabled():
            return
        previous = getattr(self, "_contact_copy_profile", None)
        if previous is not None:
            self._finish_contact_copy_profile(0, status="superseded")
        self._contact_copy_profile = ContactPlacementProfile.begin(action="copy")

    def _finish_contact_copy_profile(
        self: Any,
        contact_count: int,
        *,
        status: str | None = None,
    ) -> None:
        session = getattr(self, "_contact_copy_profile", None)
        if session is None:
            return
        self._contact_copy_profile = None
        session.contact_count = max(0, int(contact_count))
        session.note("copied_to_clipboard")
        session.stop()
        resolved_status = status or (
            f"copied_{int(contact_count)}" if int(contact_count) > 0 else "rejected"
        )
        self._emit_contact_profile_output(session.format_summary(status=resolved_status))
        self._emit_contact_profile_output(session.format_stats())

    def _start_contact_paste_profile(self: Any, contact_count: int) -> None:
        if not contact_paste_profiling_enabled():
            return
        previous = getattr(self, "_contact_paste_profile", None)
        if previous is not None:
            self._finish_contact_paste_profile(0, status="superseded")
        self._contact_paste_profile = ContactPlacementProfile.begin(action="paste")
        self._contact_paste_profile.contact_count = max(0, int(contact_count))
        self._emit_contact_profile_output(
            f"[contour contact paste profiling] started contacts={max(0, int(contact_count))}"
        )

    def _finish_contact_paste_profile(
        self: Any,
        contact_count: int,
        *,
        status: str | None = None,
    ) -> None:
        session = getattr(self, "_contact_paste_profile", None)
        if session is None:
            return
        session.note("paste_applied_to_ui")
        if int(contact_count) <= 0:
            resolved_status = status or "cancelled"
            self._complete_contact_paste_profile(resolved_status)
            return
        session.contact_count = int(contact_count)
        resolved_status = status or "displayed"
        QTimer.singleShot(
            0,
            lambda: self._complete_contact_paste_profile(resolved_status),
        )

    def _complete_contact_paste_profile(self: Any, status: str) -> None:
        session = getattr(self, "_contact_paste_profile", None)
        if session is None:
            return
        self._contact_paste_profile = None
        session.stop()
        self._emit_contact_profile_output(session.format_summary(status=status))
        self._emit_contact_profile_output(session.format_stats())

    def _start_contact_multi_selection_profile(self: Any) -> None:
        if not contact_multi_selection_profiling_enabled():
            return
        previous = getattr(self, "_contact_multi_selection_profile", None)
        if previous is not None:
            previous.stop()
        self._contact_multi_selection_profile = ContactPlacementProfile.begin(
            action="multi_selection"
        )

    def _mark_contact_multi_selection_apply_started(self: Any) -> None:
        session = getattr(self, "_contact_multi_selection_profile", None)
        if session is None:
            return
        session.note("selection_gesture")
        session.restart_main_profiler()
