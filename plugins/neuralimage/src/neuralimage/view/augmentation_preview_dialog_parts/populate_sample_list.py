"""augmentation_preview_dialog: populate sample list."""
from __future__ import annotations
from neuralimage.augmentations import (
    PCBDefectAugmentor as PCBDefectAugmentor,
)


class PopulateSampleListMixin:
    def _sync_group_boxes(self) -> None:
        panel = self._panel
        panel._sync_augmentation_controls(panel.additional_augmentation_check_box.isChecked())
        panel._sync_training_augmentation_controls()
        panel._sync_synthetic_defect_generator_controls(panel.synthetic_defect_generator_check_box.isChecked())
        panel._sync_sem_segmentation_controls()

    def _on_settings_changed(self, *_args: object) -> None:
        self._prepared_arrays_cache.clear()
        self._cutter_length_cache.clear()
        self._sync_group_boxes()
        self._rebuild_frame_plan()
        self._refresh_preview()

    def _on_full_image_toggled(self, _checked: bool) -> None:
        self._refresh_preview()

    def _emit_apply_to_main(self) -> None:
        self.apply_to_main_requested.emit(self._build_apply_payload())

    def _populate_sample_list(self) -> None:
        self._sample_list_updating = True
        self.sample_list_widget.setUpdatesEnabled(False)
        self.sample_list_widget.clear()
        self.sample_list_widget.addItems(
            [sample_path.name for sample_path, _label_path in self._sample_pairs]
        )
        self.sample_list_widget.setToolTip(str(self._training_parameters.image_path))
        synthetic_enabled = bool(self._panel.synthetic_defect_generator_check_box.isChecked())
        self.sample_list_widget.setEnabled(bool(self._sample_pairs) and not synthetic_enabled)
        if self._sample_pairs:
            self.sample_list_widget.setCurrentRow(self._current_sample_index)
        self.sample_list_widget.setUpdatesEnabled(True)
        self._sample_list_updating = False

    def _show_loading_state(self) -> None:
        loading_text = str(self._texts.get('loading', 'Loading preview...'))
        self.image_preview.setText(loading_text)
        self.label_preview.setText(loading_text)
        self.status_label.setText(loading_text)

    def _sync_sample_list_selection(self) -> None:
        if self.sample_list_widget.count() != len(self._sample_pairs):
            self._populate_sample_list()
            return
        self._sample_list_updating = True
        self.sample_list_widget.setEnabled(bool(self._sample_pairs) and not self._panel.synthetic_defect_generator_check_box.isChecked())
        if self._sample_pairs and self.sample_list_widget.currentRow() != self._current_sample_index:
            self.sample_list_widget.setCurrentRow(self._current_sample_index)
        elif not self._sample_pairs and self.sample_list_widget.currentRow() != -1:
            self.sample_list_widget.setCurrentRow(-1)
        self._sample_list_updating = False

    def _on_sample_list_row_changed(self, row: int) -> None:
        if self._sample_list_updating or not self._sample_pairs:
            return
        if row < 0 or row >= len(self._sample_pairs) or row == self._current_sample_index:
            return
        self._current_sample_index = row
        self._rebuild_frame_plan()
        for frame_index, (source_index, _cutter_item, _aug_variant) in enumerate(self._frame_plan):
            if source_index == row:
                self._current_frame_index = frame_index
                break
        self._refresh_preview()
