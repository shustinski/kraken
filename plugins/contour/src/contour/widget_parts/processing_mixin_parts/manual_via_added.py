"""processing_mixin: manual via added."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    QTimer,
)


class ManualViaAddedMixin:
    def _on_manual_via_added(self: Any, center_x: float, center_y: float) -> None:
        context = self._heuristic_contact_feedback_context()
        if context is None:
            self._finish_contact_placement_profile("displayed_without_checks")
            return
        _path, _state, image, settings = context
        from ...application.use_cases.contact_feedback import fit_positive_contact
        from ...vision.via_detection import analyze_vias_at
        from ...vision.via_detection.settings_bridge import heuristic_config_from_settings

        detection = analyze_vias_at(
            image,
            [(float(center_x), float(center_y))],
            heuristic_config_from_settings(settings),
        )[0]
        adjustment = fit_positive_contact(settings, detection)
        if self._apply_contact_feedback_adjustments([adjustment]):
            self._append_log(
                self._contact_feedback_text(
                    "Пороги распознавания расширены по установленному контакту.",
                    "Recognition thresholds were expanded from the added contact.",
                )
            )
        elif adjustment.reason == "measurement_failed":
            self._append_log(
                self._contact_feedback_text(
                    "Не удалось измерить установленный контакт; настройки не изменены.",
                    "The added contact could not be measured; settings were not changed.",
                )
            )
        elif adjustment.reason == "already_within_thresholds":
            self._append_log(
                self._contact_feedback_text(
                    "Контакт уже лежит внутри всех настраиваемых порогов; распознавание перезапущено.",
                    "The contact is already inside every adjustable threshold; recognition was restarted.",
                )
            )
            self._restart_after_contact_feedback()
        session = getattr(self, "_contact_placement_profile", None)
        if session is not None:
            session.note("contact_checks_finished")
            if (
                adjustment.reason == "measurement_failed"
                or (
                    not adjustment.changes
                    and adjustment.reason != "already_within_thresholds"
                )
            ):
                QTimer.singleShot(
                    0,
                    lambda: self._finish_contact_placement_profile("displayed"),
                )
