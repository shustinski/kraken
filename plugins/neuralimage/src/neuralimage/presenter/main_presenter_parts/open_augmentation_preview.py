"""main_presenter: open augmentation preview."""
from __future__ import annotations
from neuralimage.application.services import (
    build_workflow_parameters,
)


class OpenAugmentationPreviewMixin:
    def _on_batch_preview_visibility_changed(self, enabled: bool):
        self.settings_state.show_batch_preview = bool(enabled)

    def _open_augmentation_preview(self):
        self._update_main_window_state()
        self._update_settings_window_state()
        _work_mode, training_parameters, _recognition_parameters = build_workflow_parameters(
            self.main_window_state,
            self.settings_state,
        )
        from neuralimage.view.augmentation_preview_dialog import AugmentationPreviewDialog

        existing_dialog = getattr(self, '_augmentation_preview_dialog', None)
        if existing_dialog is not None:
            try:
                existing_dialog.close()
            finally:
                self._augmentation_preview_dialog = None

        dialog = AugmentationPreviewDialog(training_parameters, self.settings_panel)
        dialog.apply_to_main_requested.connect(self._apply_augmentation_preview_settings)
        dialog.destroyed.connect(lambda *_args: setattr(self, '_augmentation_preview_dialog', None))
        self._augmentation_preview_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
