from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import QWidget

TEXT_FIELD_LOG_UPDATE_FREQUENCY = 'log_update_frequency'
TEXT_FIELD_RECOGNITION_JPEG_QUALITY = 'recognition_jpeg_quality'


def _copy_text_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return dict(value)


def apply_description_tooltips(
    panel: Any,
    descriptions: dict[str, Any],
    labels_map: dict[str, Any],
) -> None:
    for key, label in panel._desc_labels.items():
        short_text = str(labels_map.get(key, descriptions.get(key, '')))
        detailed_text = str(descriptions.get(key, short_text))
        label.setText(short_text)
        label.setToolTip(detailed_text)
        field = panel._desc_fields.get(key)
        if field is not None:
            panel._apply_tooltip_to_widget_and_children(field, detailed_text)


def apply_settings_panel_texts(panel: Any) -> None:
    t = panel._texts if isinstance(panel._texts, dict) else {}
    labels_map = _copy_text_dict(t.get('labels', {}))
    descriptions = _copy_text_dict(t.get('field_descriptions', {}))
    log_update_key = TEXT_FIELD_LOG_UPDATE_FREQUENCY
    jpeg_quality_key = TEXT_FIELD_RECOGNITION_JPEG_QUALITY
    sync_patch_sizes_key = 'sync_patch_sizes'
    crops_per_image_key = 'crops_per_image'
    scale_augmentation_strength_key = 'scale_augmentation_strength'
    augmentation_gamma_strength_key = 'augmentation_gamma_strength'
    augmentation_blur_probability_key = 'augmentation_blur_probability'
    augmentation_blur_radius_key = 'augmentation_blur_radius'
    cutout_probability_key = 'cutout_probability'
    cutout_holes_key = 'cutout_holes'
    cutout_size_ratio_key = 'cutout_size_ratio'
    mixup_probability_key = 'mixup_probability'
    mixup_alpha_key = 'mixup_alpha'
    rare_patch_oversampling_factor_key = 'rare_patch_oversampling_factor'
    recognition_threshold_key = 'recognition_threshold'
    recognition_postprocess_kernel_size_key = 'recognition_postprocess_kernel_size'
    if log_update_key not in labels_map:
        labels_map[log_update_key] = 'Частота логов (батчей)'
    if log_update_key not in descriptions:
        descriptions[log_update_key] = str(
            t.get('log_update_frequency_tip', 'Частота обновления логов в батчах (0 = авто).')
        )
    if sync_patch_sizes_key not in labels_map:
        labels_map[sync_patch_sizes_key] = str(t.get('sync_patch_sizes', 'Use the same patch size'))
    if sync_patch_sizes_key not in descriptions:
        descriptions[sync_patch_sizes_key] = str(
            t.get(
                'sync_patch_sizes_tip',
                'If enabled, recognition patch size follows training patch size.',
            )
        )
    if jpeg_quality_key not in labels_map:
        labels_map[jpeg_quality_key] = 'JPEG quality'
    if jpeg_quality_key not in descriptions:
        descriptions[jpeg_quality_key] = str(
            t.get('recognition_jpeg_quality_tip', 'JPEG quality for recognition output (1..100).')
        )
    if crops_per_image_key not in labels_map:
        labels_map[crops_per_image_key] = 'Crops per image'
    if crops_per_image_key not in descriptions:
        descriptions[crops_per_image_key] = str(
            t.get(
                'crops_per_image_tip',
                'How many random crops to sample from one image in online random crop mode.',
            )
        )
    if scale_augmentation_strength_key not in labels_map:
        labels_map[scale_augmentation_strength_key] = 'Scale augmentation strength'
    if scale_augmentation_strength_key not in descriptions:
        descriptions[scale_augmentation_strength_key] = str(
            t.get(
                'scale_augmentation_strength_tip',
                'How strongly scale augmentation zooms patches in or out before resizing them back.',
            )
        )
    if augmentation_gamma_strength_key not in labels_map:
        labels_map[augmentation_gamma_strength_key] = 'Gamma change'
    if augmentation_gamma_strength_key not in descriptions:
        descriptions[augmentation_gamma_strength_key] = str(
            t.get(
                'extra_aug_gamma_tip',
                'How strongly gamma correction may shift darker and lighter tones.',
            )
        )
    if augmentation_blur_probability_key not in labels_map:
        labels_map[augmentation_blur_probability_key] = 'Blur probability'
    if augmentation_blur_probability_key not in descriptions:
        descriptions[augmentation_blur_probability_key] = str(
            t.get(
                'extra_aug_blur_prob_tip',
                'Probability of applying Gaussian blur to a training patch.',
            )
        )
    if augmentation_blur_radius_key not in labels_map:
        labels_map[augmentation_blur_radius_key] = 'Blur radius'
    if augmentation_blur_radius_key not in descriptions:
        descriptions[augmentation_blur_radius_key] = str(
            t.get(
                'extra_aug_blur_radius_tip',
                'Blur radius used by Gaussian blur augmentation.',
            )
        )
    if cutout_probability_key not in labels_map:
        labels_map[cutout_probability_key] = 'Cutout probability'
    if cutout_probability_key not in descriptions:
        descriptions[cutout_probability_key] = str(
            t.get('cutout_probability_tip', 'Probability of applying cutout to a training sample.')
        )
    if cutout_holes_key not in labels_map:
        labels_map[cutout_holes_key] = 'Cutout holes'
    if cutout_holes_key not in descriptions:
        descriptions[cutout_holes_key] = str(
            t.get('cutout_holes_tip', 'How many rectangular regions to erase in one sample.')
        )
    if cutout_size_ratio_key not in labels_map:
        labels_map[cutout_size_ratio_key] = 'Cutout size ratio'
    if cutout_size_ratio_key not in descriptions:
        descriptions[cutout_size_ratio_key] = str(
            t.get('cutout_size_ratio_tip', 'Size of each cutout rectangle relative to patch width and height.')
        )
    if mixup_probability_key not in labels_map:
        labels_map[mixup_probability_key] = 'Mixup probability'
    if mixup_probability_key not in descriptions:
        descriptions[mixup_probability_key] = str(
            t.get('mixup_probability_tip', 'Probability of applying mixup to a training batch.')
        )
    if mixup_alpha_key not in labels_map:
        labels_map[mixup_alpha_key] = 'Mixup alpha'
    if mixup_alpha_key not in descriptions:
        descriptions[mixup_alpha_key] = str(
            t.get('mixup_alpha_tip', 'Beta distribution alpha parameter used to sample mixup lambda.')
        )
    if rare_patch_oversampling_factor_key not in labels_map:
        labels_map[rare_patch_oversampling_factor_key] = 'Rare patch oversampling factor'
    if rare_patch_oversampling_factor_key not in descriptions:
        descriptions[rare_patch_oversampling_factor_key] = str(
            t.get(
                'rare_patch_oversampling_factor_tip',
                'How many times to repeat patches intersecting selected rare regions.',
            )
        )
    if recognition_threshold_key not in labels_map:
        labels_map[recognition_threshold_key] = 'Recognition threshold'
    if recognition_threshold_key not in descriptions:
        descriptions[recognition_threshold_key] = str(
            t.get(
                'recognition_threshold_tip',
                'Probability threshold used to convert recognition output into a binary mask.',
            )
        )
    if recognition_postprocess_kernel_size_key not in labels_map:
        labels_map[recognition_postprocess_kernel_size_key] = 'Postprocess kernel'
    if recognition_postprocess_kernel_size_key not in descriptions:
        descriptions[recognition_postprocess_kernel_size_key] = str(
            t.get(
                'recognition_postprocess_kernel_size_tip',
                'Odd kernel size used for binary mask cleanup after thresholding.',
            )
        )

    panel.samples_number.setText(str(t.get('samples_count', 'Samples in dataset: 0')))
    panel.shuffle_frames_check_box.setText(str(t.get('shuffle_frames', 'Shuffle frames')))
    panel.shuffle_frames_check_box.setToolTip(
        str(t.get('shuffle_frames_tip', 'Randomize frame order during training.'))
    )
    panel.shuffle_patches_in_frame_check_box.setText(
        str(t.get('shuffle_patches_in_frame', 'Shuffle patches inside frame'))
    )
    panel.shuffle_patches_in_frame_check_box.setToolTip(
        str(
            t.get(
                'shuffle_patches_in_frame_tip',
                'In online mode, randomize patch order extracted from a single frame.',
            )
        )
    )
    panel.sync_patch_sizes_check_box.setText('')
    panel.sync_patch_sizes_check_box.setToolTip(
        str(
            t.get(
                'sync_patch_sizes_tip',
                'If enabled, recognition patch size follows training patch size.',
            )
        )
    )
    panel.train_patch_size_widget.setToolTip(str(t.get('train_patch_tip', t.get('sample_size_tip', ''))))
    panel.recognition_patch_size_widget.setToolTip(str(t.get('recognition_patch_tip', t.get('sample_size_tip', ''))))
    panel.vertical_rotation.setText(str(t.get('rotate_180', 'Rotate frame by 180 degrees')))
    panel.vertical_rotation.setToolTip(str(t.get('rotate_180_tip', '')))
    panel.horizontal_rotation.setText(str(t.get('rotate_90', 'Rotate frame by 90 degrees')))
    panel.horizontal_rotation.setToolTip(str(t.get('rotate_90_tip', '')))
    panel.additional_augmentation_check_box.setText(
        str(t.get('extra_aug', 'Additional sample augmentation'))
    )
    panel.additional_augmentation_check_box.setToolTip(str(t.get('extra_aug_tip', '')))
    panel.random_crop_check_box.setText(str(t.get('random_crop', 'Random crop in online mode')))
    panel.random_crop_check_box.setToolTip(
        str(
            t.get(
                'random_crop_tip',
                'In online mode, use random patch coordinates instead of a fixed extraction grid. Patch step is ignored.',
            )
        )
    )
    panel.scale_augmentation_check_box.setText(str(t.get('scale_augmentation', 'Scale augmentation in online mode')))
    panel.scale_augmentation_check_box.setToolTip(
        str(
            t.get(
                'scale_augmentation_tip',
                'In online mode, randomly zoom patches in or out before resizing them back to patch size.',
            )
        )
    )
    panel.cutout_check_box.setText(str(t.get('cutout_enable', 'Enable cutout')))
    panel.cutout_check_box.setToolTip(
        str(t.get('cutout_tip', 'Randomly erase rectangular regions in training images only.'))
    )
    panel.mixup_check_box.setText(str(t.get('mixup_enable', 'Enable mixup')))
    panel.mixup_check_box.setToolTip(
        str(t.get('mixup_tip', 'Mix pairs of training samples inside a batch using a random interpolation factor.'))
    )
    panel.augmentation_brightness_spinbox.setToolTip(str(t.get('extra_aug_brightness_tip', '')))
    panel.augmentation_contrast_spinbox.setToolTip(str(t.get('extra_aug_contrast_tip', '')))
    panel.augmentation_gamma_spinbox.setToolTip(str(t.get('extra_aug_gamma_tip', '')))
    panel.augmentation_noise_probability_spinbox.setToolTip(str(t.get('extra_aug_noise_prob_tip', '')))
    panel.augmentation_noise_sigma_spinbox.setToolTip(str(t.get('extra_aug_noise_sigma_tip', '')))
    panel.augmentation_blur_probability_spinbox.setToolTip(str(t.get('extra_aug_blur_prob_tip', '')))
    panel.augmentation_blur_radius_spinbox.setToolTip(str(t.get('extra_aug_blur_radius_tip', '')))
    panel.scale_augmentation_strength_spinbox.setToolTip(str(t.get('scale_augmentation_strength_tip', '')))
    panel.cutout_probability_spinbox.setToolTip(str(t.get('cutout_probability_tip', '')))
    panel.cutout_holes_spinbox.setToolTip(str(t.get('cutout_holes_tip', '')))
    panel.cutout_size_ratio_spinbox.setToolTip(str(t.get('cutout_size_ratio_tip', '')))
    panel.mixup_probability_spinbox.setToolTip(str(t.get('mixup_probability_tip', '')))
    panel.mixup_alpha_spinbox.setToolTip(str(t.get('mixup_alpha_tip', '')))
    panel.shift_spinbox.setToolTip(str(t.get('shift_tip', '')))
    panel.validation_check_box.setText(str(t.get('validation', 'Use validation during training')))
    panel.validation_check_box.setToolTip(str(t.get('validation_tip', '')))
    panel.general_groupbox.setTitle(str(t.get('general_group', 'Data and model')))
    panel.augmentation_groupbox.setTitle(str(t.get('augmentation_group', 'Augmentation and shift')))
    panel.validation_groupbox.setTitle(str(t.get('validation_group', 'Validation')))
    panel.sample_type_groupbox.setTitle(str(t.get('sample_group', 'Dataset mode')))
    panel.sample_type_groupbox.setToolTip(str(t.get('sample_group_tip', '')))
    panel.cut_dataset_type.setText(str(t.get('cut_to_disk', 'Cut to files')))
    panel.no_cut_dataset_type.setText(str(t.get('cut_online', 'Cut online')))
    panel.prepare_samples_groupbox.setTitle(str(t.get('preprocess_group', 'Sample preprocessing')))
    panel.enable_crop_processing.setText(str(t.get('preprocess_crop_enable', 'Enable edge crop')))
    panel.enable_resize_processing.setText(str(t.get('preprocess_resize_enable', 'Enable resize')))
    panel.nn_auxilary_settings_groupbox.setTitle(str(t.get('aux_group', 'Additional settings')))
    panel.optimizer_groupbox.setTitle(str(t.get('optimizer_group', 'Optimizer and batch')))
    panel.precision_loss_groupbox.setTitle(str(t.get('loss_precision_group', 'Loss and mixed precision')))
    panel.runtime_groupbox.setTitle(str(t.get('runtime_group', 'Runtime and filtering')))
    panel.warmup_groupbox.setTitle(str(t.get('warmup_group', 'Warmup')))
    panel.hard_mining_groupbox.setTitle(str(t.get('hard_mining_group', 'Hard mining')))
    panel.early_stopping_groupbox.setTitle(str(t.get('early_stopping_group', 'Early stopping')))
    panel.optimizer_type.setToolTip(str(t.get('optimizer_tip', '')))
    panel.learning_rate_spinbox.setToolTip(str(t.get('lr_tip', '')))
    panel.weight_decay_spinbox.setToolTip(str(t.get('wd_tip', '')))
    panel.train_batch_spinbox.setToolTip(str(t.get('train_batch_tip', t.get('batch_tip', ''))))
    panel.recognition_batch_spinbox.setToolTip(str(t.get('recognition_batch_tip', t.get('batch_tip', ''))))
    panel.overlap_spinbox.setToolTip(str(t.get('overlap_tip', '')))
    panel.recognition_jpeg_quality_spinbox.setToolTip(
        str(t.get('recognition_jpeg_quality_tip', 'JPEG quality for recognition output (1..100).'))
    )
    panel.recognition_binarize_output_check_box.setText(
        str(t.get('recognition_binarize_output', 'Binarize recognition output'))
    )
    panel.recognition_binarize_output_check_box.setToolTip(
        str(t.get('recognition_binarize_output_tip', 'Convert probabilities into a binary mask before saving.'))
    )
    panel.recognition_use_auto_threshold_check_box.setText(
        str(t.get('recognition_use_auto_threshold', 'Use recommended threshold from model'))
    )
    panel.recognition_use_auto_threshold_check_box.setToolTip(
        str(
            t.get(
                'recognition_use_auto_threshold_tip',
                'Use the threshold selected on validation and saved inside the model artifact.',
            )
        )
    )
    panel.recognition_threshold_spinbox.setToolTip(
        str(
            t.get(
                'recognition_threshold_tip',
                'Probability threshold used to convert recognition output into a binary mask.',
            )
        )
    )
    panel.recognition_postprocess_check_box.setText(
        str(t.get('recognition_postprocess', 'Postprocess binary mask'))
    )
    panel.recognition_postprocess_check_box.setToolTip(
        str(t.get('recognition_postprocess_tip', 'Apply simple morphology to clean up the binary mask.'))
    )
    panel.recognition_postprocess_kernel_size_spinbox.setToolTip(
        str(
            t.get(
                'recognition_postprocess_kernel_size_tip',
                'Odd kernel size used for binary mask cleanup after thresholding.',
            )
        )
    )
    panel.log_update_frequency_spinbox.setToolTip(
        str(t.get('log_update_frequency_tip', 'Log update frequency in batches (0 = auto).'))
    )
    panel.mixed_precision_type.setToolTip(str(t.get('mixed_precision_tip', '')))
    loss_tip = str(t.get('loss_function_tip', ''))
    loss_group_title = str(labels_map.get('loss_function', t.get('loss_function', 'Loss terms')))
    panel.loss_terms_groupbox.setTitle(loss_group_title)
    panel.loss_terms_groupbox.setToolTip(loss_tip)
    panel.loss_terms_widget.setToolTip(loss_tip)
    panel.loss_formula_label.setToolTip(loss_tip)
    for loss_name in getattr(panel, 'loss_term_checkboxes', {}):
        panel.loss_term_checkboxes[loss_name].setToolTip(loss_tip)
        panel.loss_term_spinboxes[loss_name].setToolTip(loss_tip)
        panel.loss_term_labels[loss_name].setToolTip(loss_tip)
    panel.multi_gpu_mode_combo.setToolTip(str(t.get('multi_gpu_tip', '')))
    panel.torch_compile_check_box.setText(str(t.get('torch_compile_enable', 'Enable torch.compile')))
    panel.torch_compile_check_box.setToolTip(
        str(t.get('torch_compile_tip', 'Compile PyTorch graph for speedup when supported.'))
    )
    panel.skip_uniform_labels_check_box.setText(
        str(t.get('skip_uniform_labels', 'Skip samples where label is all 0 or all 1'))
    )
    panel.skip_uniform_labels_check_box.setToolTip(str(t.get('skip_uniform_labels_tip', '')))
    panel.rare_patch_oversampling_check_box.setText(
        str(
            t.get(
                'rare_patch_oversampling_enable',
                'Oversample patches with selected rare regions',
            )
        )
    )
    panel.rare_patch_oversampling_check_box.setToolTip(
        str(
            t.get(
                'rare_patch_oversampling_tip',
                'In online mode, repeats patches that intersect manually selected rare regions.',
            )
        )
    )
    panel.rare_patch_oversampling_factor_spinbox.setToolTip(
        str(
            t.get(
                'rare_patch_oversampling_factor_tip',
                'How many times to repeat patches that intersect selected rare regions.',
            )
        )
    )
    panel.edit_rare_regions_button.setText(str(t.get('edit_rare_regions', 'Edit rare regions')))
    panel.edit_rare_regions_button.setToolTip(
        str(
            t.get(
                'edit_rare_regions_tip',
                'Open the editor for manual selection of rare and difficult regions.',
            )
        )
    )
    panel.warmup_check_box.setText(str(t.get('warmup_enable', 'Enable warmup')))
    panel.warmup_check_box.setToolTip(str(t.get('warmup_tip', '')))
    panel.hard_mining_check_box.setText(str(t.get('hard_mining_enable', 'Train more on hard samples (high loss)')))
    panel.hard_mining_check_box.setToolTip(str(t.get('hard_mining_tip', '')))
    panel.hard_pixel_mining_check_box.setText(
        str(t.get('hard_pixel_mining_enable', 'Focus loss on hardest pixels'))
    )
    panel.hard_pixel_mining_check_box.setToolTip(str(t.get('hard_pixel_mining_tip', '')))
    panel.early_stopping_check_box.setText(str(t.get('early_stopping_enable', 'Enable early stopping')))
    panel.early_stopping_check_box.setToolTip(str(t.get('early_stopping_tip', '')))
    panel.restore_best_weights_check_box.setText(str(t.get('restore_best', 'Restore best weights')))
    panel.restore_best_weights_check_box.setToolTip(str(t.get('restore_best_tip', '')))
    panel.reset_defaults_button.setText(str(t.get('reset_defaults', 'Reset defaults')))
    panel.reset_defaults_button.setToolTip(str(t.get('reset_defaults_tip', 'Reset all parameters to initial values.')))
    color_modes = t.get('color_modes', {'RGB': 'RGB', 'ЧБ': 'ЧБ'})
    if isinstance(color_modes, dict) and color_modes:
        panel.set_color_mode_items([(str(value), str(label)) for value, label in color_modes.items()])
    elif isinstance(color_modes, list) and color_modes:
        panel.set_color_mode_items([(str(v), str(v)) for v in color_modes])
    apply_description_tooltips(panel, descriptions, labels_map)


