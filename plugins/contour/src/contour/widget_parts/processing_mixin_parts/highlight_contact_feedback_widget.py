"""processing_mixin: highlight contact feedback widget."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    QTimer,
    QWidget,
)


class HighlightContactFeedbackWidgetMixin:
    def _restart_after_contact_feedback(self: Any) -> None:
        for attribute in (
            "_contact_deletion_profile",
            "_contact_placement_profile",
        ):
            session = getattr(self, attribute, None)
            if session is not None:
                session.waiting_for_preview = True
                session.note("preview_requested")
        self._abort_in_flight_interactive_processing(preview=True, prepared=False)
        self.process_current_image(debounced=False)

    def _format_contact_feedback_value(self: Any, value: float) -> str:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        text = f"{numeric:.3f}".rstrip("0").rstrip(".")
        return text.replace(".", ",") if getattr(self, "_ui_language", "en") == "ru" else text

    def _highlight_contact_feedback_widget(self: Any, widget: QWidget) -> None:
        previous_style = widget.styleSheet()
        highlight_style = (
            previous_style
            + " QSpinBox, QDoubleSpinBox { border: 2px solid #FACC15; "
            "background-color: rgba(250, 204, 21, 0.18); }"
        )
        widget.setStyleSheet(highlight_style)
        label = None
        for form_name in ("bright_via_form", "bright_via_basics_form"):
            form = getattr(self, form_name, None)
            if form is not None:
                candidate = form.labelForField(widget)
                if candidate is not None:
                    label = candidate
                    break
        previous_label_style = label.styleSheet() if label is not None else ""
        if label is not None:
            label.setStyleSheet(previous_label_style + " color: #FACC15; font-weight: 600;")

        def clear_highlight() -> None:
            try:
                if widget.styleSheet() == highlight_style:
                    widget.setStyleSheet(previous_style)
                if (
                    label is not None
                    and label.styleSheet()
                    == previous_label_style + " color: #FACC15; font-weight: 600;"
                ):
                    label.setStyleSheet(previous_label_style)
            except RuntimeError:
                pass

        QTimer.singleShot(3000, clear_highlight)

    def _antialias_selected_polygons(self: Any) -> None:
        grade = int(self.antialias_grade_spin.value()) if hasattr(self, "antialias_grade_spin") else 1
        if not self.polygon_editor.antialias_selected_polygons(grade):
            self._append_log(
                self._tr(
                    "antialias_selected_none_log",
                    "Выберите полигоны для сглаживания."
                    if self._ui_language == "ru"
                    else "Select polygons to antialias.",
                )
            )
            return
        self._append_log(
            self._tr(
                "antialias_selected_done_log",
                "Сглаживание применено к выбранным полигонам, grade={grade}."
                if self._ui_language == "ru"
                else "Antialiasing applied to selected polygons, grade={grade}.",
                grade=grade,
            )
        )
