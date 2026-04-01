from __future__ import annotations

from typing import Any, Callable, Iterable


def _wrap_noarg(callback: Callable[[], None]) -> Callable[..., None]:
    return lambda *_args, **_kwargs: callback()


def _connect_noarg(signal: Any, callback: Callable[[], None]) -> None:
    signal.connect(_wrap_noarg(callback))


def _connect_emitters(panel_signals: Iterable[Any], emitter: Any) -> None:
    for signal in panel_signals:
        _connect_noarg(signal, emitter.emit)


def connect_settings_panel_signals(panel: Any) -> None:
    """Connect widget events to panel-level signals and sync handlers."""
    _connect_noarg(panel.horizontal_rotation.clicked, panel.horisontal_rotate_clicked.emit)
    _connect_noarg(panel.vertical_rotation.clicked, panel.vertical_rotate_clicked.emit)
    _connect_noarg(panel.nn_model_type.currentIndexChanged, panel.model_changed.emit)
    _connect_noarg(panel.deprecated_model_type.currentIndexChanged, panel.model_changed.emit)
    _connect_noarg(panel.experimental_model_type.currentIndexChanged, panel.model_changed.emit)

    _connect_emitters(
        (
            panel.additional_augmentation_check_box.toggled,
            panel.flip_x.toggled,
            panel.flip_y.toggled,
            panel.random_crop_check_box.toggled,
            panel.crops_per_image_spinbox.valueChanged,
            panel.scale_augmentation_check_box.toggled,
            panel.scale_augmentation_strength_spinbox.valueChanged,
            panel.synthetic_defect_generator_check_box.toggled,
            panel.synthetic_topology_domain_combo.currentIndexChanged,
            panel.pcb_topology_family_combo.currentIndexChanged,
            panel.ic_topology_family_combo.currentIndexChanged,
            panel.synthetic_dataset_factor_spinbox.valueChanged,
            panel.synthetic_image_width_spinbox.valueChanged,
            panel.synthetic_image_height_spinbox.valueChanged,
            panel.synthetic_trace_count_min_spinbox.valueChanged,
            panel.synthetic_trace_count_max_spinbox.valueChanged,
            panel.synthetic_segment_count_min_spinbox.valueChanged,
            panel.synthetic_segment_count_max_spinbox.valueChanged,
            panel.synthetic_trace_half_width_min_spinbox.valueChanged,
            panel.synthetic_trace_half_width_max_spinbox.valueChanged,
            panel.synthetic_background_noise_sigma_min_spinbox.valueChanged,
            panel.synthetic_background_noise_sigma_max_spinbox.valueChanged,
            panel.synthetic_trace_noise_sigma_min_spinbox.valueChanged,
            panel.synthetic_trace_noise_sigma_max_spinbox.valueChanged,
            panel.tech_augmentation_check_box.toggled,
            panel.tech_augmentation_debug_pair_check_box.toggled,
            panel.tech_aug_min_operations_spinbox.valueChanged,
            panel.tech_aug_max_operations_spinbox.valueChanged,
            panel.tech_aug_max_changed_pixels_ratio_spinbox.valueChanged,
            panel.tech_aug_max_foreground_ratio_delta_spinbox.valueChanged,
            panel.tech_aug_global_width_probability_spinbox.valueChanged,
            panel.tech_aug_scale_rethreshold_probability_spinbox.valueChanged,
            panel.tech_aug_blur_threshold_probability_spinbox.valueChanged,
            panel.tech_aug_boundary_aware_probability_spinbox.valueChanged,
            panel.tech_aug_local_morphology_probability_spinbox.valueChanged,
            panel.tech_aug_gap_variation_probability_spinbox.valueChanged,
            panel.cutout_check_box.toggled,
            panel.cutout_probability_spinbox.valueChanged,
            panel.cutout_holes_spinbox.valueChanged,
            panel.cutout_size_ratio_spinbox.valueChanged,
            panel.random_artifacts_check_box.toggled,
            panel.random_artifacts_probability_spinbox.valueChanged,
            panel.random_artifacts_count_spinbox.valueChanged,
            panel.random_artifacts_size_ratio_spinbox.valueChanged,
            panel.mixup_check_box.toggled,
            panel.mixup_probability_spinbox.valueChanged,
            panel.mixup_alpha_spinbox.valueChanged,
            panel.pcb_defects_check_box.toggled,
            panel.pcb_defects_probability_spinbox.valueChanged,
            panel.pcb_defects_min_count_spinbox.valueChanged,
            panel.pcb_defects_max_count_spinbox.valueChanged,
            panel.pcb_defects_use_input_mask_check_box.toggled,
            panel.pcb_defects_use_defect_mask_as_label_check_box.toggled,
            panel.shuffle_frames_check_box.toggled,
            panel.shuffle_patches_in_frame_check_box.toggled,
            panel.augmentation_brightness_spinbox.valueChanged,
            panel.augmentation_contrast_spinbox.valueChanged,
            panel.augmentation_gamma_spinbox.valueChanged,
            panel.augmentation_noise_probability_spinbox.valueChanged,
            panel.augmentation_noise_sigma_spinbox.valueChanged,
            panel.augmentation_blur_probability_spinbox.valueChanged,
            panel.augmentation_blur_radius_spinbox.valueChanged,
            panel.recognition_patch_x_size.valueChanged,
            panel.recognition_patch_y_size.valueChanged,
            panel.optimizer_type.currentIndexChanged,
            panel.learning_rate_spinbox.valueChanged,
            panel.weight_decay_spinbox.valueChanged,
            panel.mixed_precision_type.currentIndexChanged,
            panel.deep_supervision_check_box.toggled,
            panel.train_batch_spinbox.valueChanged,
            panel.dataloader_num_workers_spinbox.valueChanged,
            panel.recognition_batch_spinbox.valueChanged,
            panel.recognition_jpeg_quality_spinbox.valueChanged,
            panel.recognition_multiprocessing_check_box.toggled,
            panel.recognition_binarize_output_check_box.toggled,
            panel.recognition_use_auto_threshold_check_box.toggled,
            panel.recognition_threshold_spinbox.valueChanged,
            panel.recognition_tta_check_box.toggled,
            panel.confidence_tta_check_box.toggled,
            panel.confidence_save_mode_combo.currentIndexChanged,
            panel.recognition_postprocess_check_box.toggled,
            panel.recognition_postprocess_kernel_size_spinbox.valueChanged,
            panel.multi_gpu_mode_combo.currentIndexChanged,
            panel.sync_patch_sizes_check_box.toggled,
            panel.torch_compile_check_box.toggled,
            panel.warmup_check_box.toggled,
            panel.warmup_epochs_spinbox.valueChanged,
            panel.warmup_start_factor_spinbox.valueChanged,
            panel.scheduler_type_combo.currentIndexChanged,
            panel.scheduler_plateau_factor_spinbox.valueChanged,
            panel.scheduler_plateau_patience_spinbox.valueChanged,
            panel.scheduler_plateau_threshold_spinbox.valueChanged,
            panel.scheduler_plateau_min_lr_spinbox.valueChanged,
            panel.scheduler_plateau_cooldown_spinbox.valueChanged,
            panel.scheduler_cosine_t_max_spinbox.valueChanged,
            panel.scheduler_cosine_eta_min_spinbox.valueChanged,
            panel.scheduler_one_cycle_max_lr_spinbox.valueChanged,
            panel.scheduler_one_cycle_pct_start_spinbox.valueChanged,
            panel.scheduler_one_cycle_anneal_strategy_combo.currentIndexChanged,
            panel.scheduler_one_cycle_div_factor_spinbox.valueChanged,
            panel.scheduler_one_cycle_final_div_factor_spinbox.valueChanged,
            panel.scheduler_one_cycle_three_phase_check_box.toggled,
            panel.scheduler_step_lr_step_size_spinbox.valueChanged,
            panel.scheduler_step_lr_gamma_spinbox.valueChanged,
            panel.hard_mining_check_box.toggled,
            panel.hard_mining_strength_spinbox.valueChanged,
            panel.hard_mining_ema_alpha_spinbox.valueChanged,
            panel.hard_pixel_mining_check_box.toggled,
            panel.hard_pixel_mining_ratio_spinbox.valueChanged,
            panel.log_update_frequency_spinbox.valueChanged,
            panel.skip_uniform_labels_check_box.toggled,
            panel.rare_patch_oversampling_check_box.toggled,
            panel.rare_patch_oversampling_factor_spinbox.valueChanged,
            panel.early_stopping_check_box.toggled,
            panel.early_stopping_patience_spinbox.valueChanged,
            panel.early_stopping_min_delta_spinbox.valueChanged,
            panel.restore_best_weights_check_box.toggled,
            panel.enable_crop_processing.toggled,
            panel.enable_resize_processing.toggled,
            panel.cut_dataset_type.toggled,
            panel.no_cut_dataset_type.toggled,
            panel.cut_corner_spinbox.valueChanged,
            panel.target_x_size.valueChanged,
            panel.target_y_size.valueChanged,
            *(checkbox.toggled for checkbox in getattr(panel, 'random_artifact_type_checkboxes', {}).values()),
        ),
        panel.optimizer_settings_changed,
    )
    _connect_emitters(
        tuple(spinbox.valueChanged for spinbox in getattr(panel, 'pcb_defect_type_spinboxes', {}).values()),
        panel.optimizer_settings_changed,
    )
    _connect_emitters(
        tuple(spinbox.valueChanged for spinbox in getattr(panel, 'ic_defect_type_spinboxes', {}).values()),
        panel.optimizer_settings_changed,
    )
    for loss_name in getattr(panel, 'loss_term_checkboxes', {}):
        _connect_emitters(
            (
                panel.loss_term_checkboxes[loss_name].toggled,
                panel.loss_term_spinboxes[loss_name].valueChanged,
            ),
            panel.optimizer_settings_changed,
        )
    _connect_emitters(
        (
            panel.additional_augmentation_check_box.toggled,
            panel.flip_x.toggled,
            panel.flip_y.toggled,
            panel.random_crop_check_box.toggled,
            panel.crops_per_image_spinbox.valueChanged,
            panel.scale_augmentation_check_box.toggled,
            panel.synthetic_defect_generator_check_box.toggled,
            panel.synthetic_image_width_spinbox.valueChanged,
            panel.synthetic_image_height_spinbox.valueChanged,
            panel.cut_dataset_type.toggled,
            panel.no_cut_dataset_type.toggled,
            panel.shift_spinbox.valueChanged,
        ),
        panel.cut_slider_shifted,
    )
    _connect_emitters(
        (
            panel.train_patch_x_size.valueChanged,
            panel.train_patch_y_size.valueChanged,
        ),
        panel.sample_size_changed,
    )
    _connect_emitters(
        (
            panel.validation_check_box.toggled,
            panel.validation_mode_combo.currentIndexChanged,
            panel.validation_spinbox.valueChanged,
            panel.save_validation_binary_images_check_box.toggled,
        ),
        panel.validation_settings_changed,
    )
    _connect_noarg(panel.validation_image_path_label.clicked, panel.validation_image_path_requested.emit)
    _connect_noarg(panel.validation_label_path_label.clicked, panel.validation_label_path_requested.emit)
    _connect_noarg(panel.reset_defaults_button.clicked, panel.reset_defaults_requested.emit)
    _connect_noarg(panel.augmentation_preview_button.clicked, panel.augmentation_preview_requested.emit)
    _connect_noarg(panel.edit_rare_regions_button.clicked, panel.rare_patch_editor_requested.emit)

    panel.train_patch_x_size.valueChanged.connect(panel._sync_patch_size_controls)
    panel.train_patch_y_size.valueChanged.connect(panel._sync_patch_size_controls)
    panel.sync_patch_sizes_check_box.toggled.connect(panel._sync_patch_size_controls)
    panel.optimizer_type.currentIndexChanged.connect(panel._sync_active_optimizer_preset)
    panel.learning_rate_spinbox.valueChanged.connect(panel._sync_active_optimizer_preset)
    panel.weight_decay_spinbox.valueChanged.connect(panel._sync_active_optimizer_preset)
    panel.scale_augmentation_check_box.toggled.connect(
        lambda *_args, **_kwargs: panel._sync_augmentation_controls(panel.additional_augmentation_check_box.isChecked())
    )
    panel.random_crop_check_box.toggled.connect(
        lambda *_args, **_kwargs: panel._sync_augmentation_controls(panel.additional_augmentation_check_box.isChecked())
    )
    panel.synthetic_defect_generator_check_box.toggled.connect(panel._sync_synthetic_defect_generator_controls)
    panel.synthetic_topology_domain_combo.currentIndexChanged.connect(panel._sync_synthetic_domain_controls)
    panel.cut_dataset_type.toggled.connect(
        lambda *_args, **_kwargs: panel._sync_augmentation_controls(panel.additional_augmentation_check_box.isChecked())
    )
    panel.no_cut_dataset_type.toggled.connect(
        lambda *_args, **_kwargs: panel._sync_augmentation_controls(panel.additional_augmentation_check_box.isChecked())
    )
    panel.augmentation_blur_probability_spinbox.valueChanged.connect(
        lambda *_args, **_kwargs: panel._sync_augmentation_controls(panel.additional_augmentation_check_box.isChecked())
    )
    panel.cutout_check_box.toggled.connect(panel._sync_training_augmentation_controls)
    panel.random_artifacts_check_box.toggled.connect(panel._sync_training_augmentation_controls)
    panel.mixup_check_box.toggled.connect(panel._sync_training_augmentation_controls)
    panel.synthetic_defect_generator_check_box.toggled.connect(panel._sync_training_augmentation_controls)
    panel.cut_dataset_type.toggled.connect(panel._sync_rare_patch_oversampling_controls)
    panel.no_cut_dataset_type.toggled.connect(panel._sync_rare_patch_oversampling_controls)
    panel.recognition_binarize_output_check_box.toggled.connect(panel._sync_recognition_output_controls)
    panel.recognition_use_auto_threshold_check_box.toggled.connect(panel._sync_recognition_output_controls)
    panel.recognition_postprocess_check_box.toggled.connect(panel._sync_recognition_output_controls)

    panel._sync_active_optimizer_preset()
    panel._sync_patch_size_controls()
