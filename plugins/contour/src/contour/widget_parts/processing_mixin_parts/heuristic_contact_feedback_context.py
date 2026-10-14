"""processing_mixin: heuristic contact feedback context."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    VIA_SEARCH_MODE_HEURISTIC,
    normalize_via_search_mode,
    np,
)


class HeuristicContactFeedbackContextMixin:
    def _heuristic_contact_feedback_context(self: Any):
        if not hasattr(self, "recognition_mode_combo") or str(
            self.recognition_mode_combo.currentData() or ""
        ) != "via":
            self._append_log(
                self._contact_feedback_text(
                    "Подбор по контактам доступен только в режиме распознавания контактов.",
                    "Contact feedback is available only in contact-recognition mode.",
                )
            )
            return None
        if (
            not hasattr(self, "via_search_mode_combo")
            or normalize_via_search_mode(self.via_search_mode_combo.currentData())
            != VIA_SEARCH_MODE_HEURISTIC
        ):
            self._append_log(
                self._contact_feedback_text(
                    "Подбор порогов доступен только для эвристического метода.",
                    "Threshold feedback is available only for the heuristic method.",
                )
            )
            return None
        current_path = self._workspace.current_image_path
        current_state = self._workspace.current_state
        if (
            not current_path
            or current_state is None
            or current_state.image_path != current_path
        ):
            self._append_log(
                self._contact_feedback_text(
                    "Нет актуального кадра для подбора.",
                    "No current frame for feedback.",
                )
            )
            return None
        image = (
            current_state.preprocessed_image
            if current_state.preprocessed_image is not None
            else current_state.source_image
        )
        if image is None:
            self._append_log(
                self._contact_feedback_text(
                    "Нет изображения для измерения контакта.",
                    "No image is available for contact measurement.",
                )
            )
            return None
        return current_path, current_state, np.asarray(image), self._current_contour_settings()
