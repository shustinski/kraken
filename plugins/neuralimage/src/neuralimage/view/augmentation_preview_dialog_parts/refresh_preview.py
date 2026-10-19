"""augmentation_preview_dialog: refresh preview."""
from __future__ import annotations
from neuralimage.augmentations import (
    PCBDefectAugmentor as PCBDefectAugmentor,
)


class RefreshPreviewMixin:
    def _refresh_preview(self) -> None:
        synthetic_enabled = self._panel.synthetic_defect_generator_check_box.isChecked()
        if not synthetic_enabled:
            self._rebuild_frame_plan()
            if self._frame_plan:
                self._current_frame_coords()
        if not self._sample_pairs and not synthetic_enabled:
            self._sync_sample_list_selection()
            error_text = str(
                self._load_error
                or self._texts.get('empty_error', 'No matched sample/label pairs were found.')
            )
            self.sample_label.setText(error_text)
            self.status_label.setText(error_text)
            self.image_preview.setText(error_text)
            self.label_preview.setText(error_text)
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.resample_button.setEnabled(False)
            self._update_preview_mode_label()
            return

        self._sync_sample_list_selection()
        if synthetic_enabled:
            self.sample_label.setText(
                str(
                    self._texts.get(
                        'synthetic_label',
                        'Синтетическая топология' if self._is_russian_ui else 'Synthetic topology',
                    )
                )
            )
        else:
            sample_path, _label_path = self._sample_pairs[self._current_sample_index]
            self.sample_label.setText(
                str(
                    self._texts.get('sample_label_template', '{index}/{total}: {name}')
                ).format(
                    index=self._current_frame_index + 1,
                    total=max(1, len(self._frame_plan)),
                    name=sample_path.name,
                )
            )
        self.status_label.setText(
            str(
                self._texts.get(
                    'status_hold_template',
                    'Variant #{variant}. Preview: {preview_mode}. Hold the middle mouse button to inspect the original image and label.',
                )
            ).format(
                variant=self._variant_serial + 1,
                preview_mode=self._preview_mode_text(),
            )
        )
        try:
            from neuralimage.remote.ui import remote_url
            if remote_url() is not None:
                from neuralimage.remote.preview_qt import request_preview
                request_preview(self)
            else:
                (self._original_image_array, self._original_label_array,
                 self._augmented_image_array, self._augmented_label_array) = self._build_preview_arrays(self._current_sample_index)
                self._update_preview_mode_label()
                self._update_visible_preview()
        except Exception as error:
            self.status_label.setText(str(error))
        self.prev_button.setEnabled((not synthetic_enabled) and len(self._frame_plan) > 1)
        self.next_button.setEnabled((not synthetic_enabled) and len(self._frame_plan) > 1)
        self.resample_button.setEnabled(True)
