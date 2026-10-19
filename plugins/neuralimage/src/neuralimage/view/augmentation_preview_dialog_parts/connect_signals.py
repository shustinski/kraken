"""augmentation_preview_dialog: connect signals."""
from __future__ import annotations
from neuralimage.augmentations import (
    PCBDefectAugmentor as PCBDefectAugmentor,
)


class ConnectSignalsMixin:
    def _connect_signals(self) -> None:
        panel = self._panel
        self.prev_button.clicked.connect(self._show_previous_sample)
        self.next_button.clicked.connect(self._show_next_sample)
        self.resample_button.clicked.connect(self._resample_current_sample)
        self.apply_to_main_button.clicked.connect(self._emit_apply_to_main)
        self.sample_list_widget.currentRowChanged.connect(self._on_sample_list_row_changed)
        self.image_preview.middle_pressed.connect(self._show_original_preview)
        self.image_preview.middle_released.connect(self._restore_augmented_preview)
        self.label_preview.middle_pressed.connect(self._show_original_preview)
        self.label_preview.middle_released.connect(self._restore_augmented_preview)
        self.full_image_check_box.toggled.connect(self._on_full_image_toggled)
        self.sem_normalization_editor.changed.connect(self._on_sem_pipeline_changed)
        self.sem_augmentation_editor.changed.connect(self._on_sem_pipeline_changed)
        for checkbox in (
            panel.horizontal_rotation,
            panel.vertical_rotation,
            panel.flip_x,
            panel.flip_y,
            panel.random_crop_check_box,
            panel.scale_augmentation_check_box,
            panel.photometric_groupbox,
            panel.brightness_augmentation_check_box,
            panel.contrast_augmentation_check_box,
            panel.gamma_augmentation_check_box,
            panel.noise_augmentation_check_box,
            panel.blur_augmentation_check_box,
            panel.tech_augmentation_check_box,
            panel.cutout_check_box,
            panel.random_artifacts_check_box,
            panel.mixup_check_box,
            panel.synthetic_defect_generator_check_box,
            panel.pcb_defects_check_box,
            *panel.random_artifact_type_checkboxes.values(),
            *panel.pcb_defect_type_checkboxes.values(),
            *panel.ic_defect_type_checkboxes.values(),
        ):
            checkbox.toggled.connect(self._on_settings_changed)
        for widget in (
            panel.shift_spinbox,
            panel.crops_per_image_spinbox,
            panel.augmentation_multiplier_spinbox,
            panel.scale_augmentation_strength_spinbox,
            panel.augmentation_brightness_spinbox,
            panel.augmentation_brightness_probability_spinbox,
            panel.augmentation_contrast_probability_spinbox,
            panel.augmentation_gamma_probability_spinbox,
            panel.augmentation_contrast_spinbox,
            panel.augmentation_gamma_spinbox,
            panel.augmentation_noise_probability_spinbox,
            panel.augmentation_noise_sigma_spinbox,
            panel.augmentation_blur_probability_spinbox,
            panel.augmentation_blur_radius_spinbox,
            panel.tech_aug_min_operations_spinbox,
            panel.tech_aug_max_operations_spinbox,
            panel.tech_aug_max_changed_pixels_ratio_spinbox,
            panel.tech_aug_max_foreground_ratio_delta_spinbox,
            panel.tech_aug_global_width_probability_spinbox,
            panel.tech_aug_scale_rethreshold_probability_spinbox,
            panel.tech_aug_blur_threshold_probability_spinbox,
            panel.tech_aug_boundary_aware_probability_spinbox,
            panel.tech_aug_local_morphology_probability_spinbox,
            panel.tech_aug_gap_variation_probability_spinbox,
            panel.cutout_probability_spinbox,
            panel.cutout_holes_spinbox,
            panel.cutout_size_ratio_spinbox,
            panel.random_artifacts_probability_spinbox,
            panel.random_artifacts_count_spinbox,
            panel.random_artifacts_size_ratio_spinbox,
            panel.mixup_probability_spinbox,
            panel.mixup_alpha_spinbox,
            panel.synthetic_dataset_factor_spinbox,
            panel.synthetic_image_width_spinbox,
            panel.synthetic_image_height_spinbox,
            panel.synthetic_trace_count_min_spinbox,
            panel.synthetic_trace_count_max_spinbox,
            panel.synthetic_segment_count_min_spinbox,
            panel.synthetic_segment_count_max_spinbox,
            panel.synthetic_trace_half_width_min_spinbox,
            panel.synthetic_trace_half_width_max_spinbox,
            panel.synthetic_background_noise_sigma_min_spinbox,
            panel.synthetic_background_noise_sigma_max_spinbox,
            panel.synthetic_trace_noise_sigma_min_spinbox,
            panel.synthetic_trace_noise_sigma_max_spinbox,
            panel.pcb_defects_probability_spinbox,
            panel.pcb_defects_min_count_spinbox,
            panel.pcb_defects_max_count_spinbox,
            panel.synthetic_topology_domain_combo,
            panel.pcb_topology_family_combo,
            panel.ic_topology_family_combo,
            *panel.pcb_defect_type_spinboxes.values(),
            *panel.ic_defect_type_spinboxes.values(),
        ):
            if hasattr(widget, 'valueChanged'):
                widget.valueChanged.connect(self._on_settings_changed)
            elif hasattr(widget, 'currentIndexChanged'):
                widget.currentIndexChanged.connect(self._on_settings_changed)
