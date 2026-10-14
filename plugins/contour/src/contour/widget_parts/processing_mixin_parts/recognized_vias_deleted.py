"""processing_mixin: recognized vias deleted."""
from __future__ import annotations
from collections.abc import Iterable
from typing import Any
from contour.widget_parts._imports import (
    PolygonData,
)


class RecognizedViasDeletedMixin:
    def _on_recognized_vias_deleted(self: Any, polygons: Iterable[PolygonData] | None) -> None:
        removed_polygons = [
            polygon
            for polygon in list(polygons or [])
            if isinstance(polygon, PolygonData) and polygon.recognition_score is not None
        ]
        if not removed_polygons:
            return
        context = self._heuristic_contact_feedback_context()
        if context is None:
            return
        _path, current_state, image, settings = context
        from ...application.use_cases.contact_feedback import fit_negative_contact
        from ...vision.via_detection import analyze_vias_at
        from ...vision.via_detection.settings_bridge import heuristic_config_from_settings

        references = sorted(
            (
                polygon
                for polygon in current_state.polygons
                if (polygon.category == "via" or polygon.shape_hint == "box")
                and polygon.recognition_score is not None
            ),
            key=lambda polygon: float(polygon.recognition_score or 0.0),
            reverse=True,
        )[:10]
        centers = [
            *(self._contact_center(polygon) for polygon in removed_polygons),
            *(self._contact_center(polygon) for polygon in references),
        ]
        measured = analyze_vias_at(
            image,
            centers,
            heuristic_config_from_settings(settings),
        )
        removed_count = len(removed_polygons)
        removed_detections = measured[:removed_count]
        reference_detections = [
            detection for detection in measured[removed_count:] if detection is not None
        ]
        adjustments = []
        for detection in removed_detections:
            adjustment = fit_negative_contact(settings, detection, reference_detections)
            adjustments.append(adjustment)
            for change in adjustment.changes:
                setattr(settings, change.field, change.new_value)
        if self._apply_contact_feedback_adjustments(adjustments):
            features = ", ".join(
                adjustment.selected_feature or ""
                for adjustment in adjustments
                if adjustment.changes
            )
            self._append_log(
                self._contact_feedback_text(
                    f"Пороги распознавания ужесточены после удаления контакта: {features}.",
                    f"Recognition thresholds were tightened after contact removal: {features}.",
                )
            )
            return
        reason = next(
            (adjustment.reason for adjustment in adjustments if adjustment.reason),
            "no_separating_threshold",
        )
        if reason == "insufficient_references":
            message = self._contact_feedback_text(
                "Контакт удалён; для изменения порогов нужно не менее двух эталонных контактов.",
                "The contact was removed; at least two reference contacts are required to change thresholds.",
            )
        else:
            message = self._contact_feedback_text(
                "Контакт удалён, но подходящая разделяющая граница не найдена.",
                "The contact was removed, but no separating threshold was found.",
            )
        self._append_log(message)
