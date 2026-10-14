"""settings_panel: sync patch size layout."""
from __future__ import annotations
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )


class SyncPatchSizeLayoutMixin:
    def _sync_patch_size_layout(self, *, random_enabled: bool, patch_sync: bool) -> None:
        self.patch_size_groups_layout.removeWidget(self.train_patch_size_groupbox)
        self.patch_size_groups_layout.removeWidget(self.recognition_patch_size_groupbox)

        self.train_patch_size_widget.setVisible(not random_enabled)
        self.random_patch_widgets_container.setVisible(random_enabled)
        self.train_patch_size_groupbox.setVisible(self._training_controls_applicable)

        show_recognition = self._recognition_controls_applicable and (random_enabled or not patch_sync)
        self.recognition_patch_size_groupbox.setVisible(show_recognition)

        if random_enabled:
            self.patch_size_groups_layout.addWidget(self.train_patch_size_groupbox, 0, 0, 1, 2)
            if show_recognition:
                self.patch_size_groups_layout.addWidget(self.recognition_patch_size_groupbox, 1, 0, 1, 2)
        else:
            self.patch_size_groups_layout.addWidget(self.train_patch_size_groupbox, 0, 0)
            if show_recognition:
                self.patch_size_groups_layout.addWidget(self.recognition_patch_size_groupbox, 0, 1)
        self.patch_size_groups_layout.setColumnStretch(0, 1)
        self.patch_size_groups_layout.setColumnStretch(1, 1)

    def _sync_rare_patch_oversampling_controls(self, _enabled: bool | None = None) -> None:
        online_mode_enabled = self._training_controls_applicable
        self.rare_patch_oversampling_check_box.setEnabled(online_mode_enabled)
        self.edit_rare_regions_button.setEnabled(online_mode_enabled)
        self._set_field_enabled(
            self.rare_patch_oversampling_factor_spinbox,
            online_mode_enabled and bool(self.rare_patch_oversampling_check_box.isChecked()),
        )

    def _sync_patch_sampling_mode(self, *_args: object) -> None:
        random_mode = bool(self.random_crop_check_box.isChecked())
        if self.random_sampling_radio.isChecked() != random_mode:
            self.random_sampling_radio.setChecked(random_mode)
        if self.grid_sampling_radio.isChecked() == random_mode:
            self.grid_sampling_radio.setChecked(not random_mode)
        enabled = self._training_controls_applicable
        self.shift_spinbox.setEnabled(enabled and not random_mode)
        self.crops_per_image_spinbox.setEnabled(enabled and random_mode)
