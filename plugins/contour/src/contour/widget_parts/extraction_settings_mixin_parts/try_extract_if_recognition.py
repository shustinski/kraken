"""extraction_settings_mixin: try extract if recognition."""
from __future__ import annotations


class TryExtractIfRecognitionMixin:
    def _auto_apply_pipeline(self) -> None:
        current_path = self._workspace.current_image_path
        state = self._workspace.current_state
        if not current_path or state is None or state.source_image is None:
            return
        # Filtering is an image-display operation and must not depend on the
        # recognition mode or its preview worker. The prepared-image result is
        # always displayed; it starts extraction afterwards only when enabled.
        self._abort_in_flight_interactive_processing(preview=True, prepared=True)
        self._queue_prepared_image_update(current_path, state.source_image)

    def _auto_apply_recognition_settings(self) -> None:
        if not self._workspace.current_image_path or not self._is_extraction_mode_enabled():
            return
        if hasattr(self, "auto_apply_checkbox") and not self.auto_apply_checkbox.isChecked():
            return
        self._abort_in_flight_interactive_processing(preview=True, prepared=False)
        self.process_current_image(debounced=False)

    def _try_extract_if_recognition_enabled(self) -> None:
        if not hasattr(self, "recognition_mode_combo"):
            return
        if str(self.recognition_mode_combo.currentData() or "") == "disabled":
            return
        current_path = self._workspace.current_image_path
        if not current_path:
            return
        state = self._workspace.current_state
        if state is None or state.image_path != current_path or state.preprocessed_image is None:
            return
        if state.pipeline_config != self.get_pipeline():
            return
        self.process_current_image(debounced=False)
