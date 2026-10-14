"""apply_settings_panel_texts section 3."""
from __future__ import annotations
from typing import Any
from neuralimage.configuration import (
    SEM_UI_SECTIONS,
    sem_ui_section_help,
    sem_ui_section_label,
)
from neuralimage.lib.ui_texts import get_ui_language
from neuralimage.view.settings_panel_i18n_parts.module_state import (
    TEXT_FIELD_LOG_UPDATE_FREQUENCY,
    TEXT_FIELD_RECOGNITION_JPEG_QUALITY,
)
from neuralimage.view.settings_panel_i18n_parts.apply_description_tooltips import (
    _copy_text_dict,
    _read_text_from_mappings,
    apply_description_tooltips,
)

def _apply_settings_panel_texts_section_3(panel, descriptions, labels_map, pcb_break_weight_key, pcb_defects_max_count_key, pcb_excess_copper_weight_key, pcb_misalignment_weight_key, pcb_missing_copper_weight_key, pcb_pinhole_weight_key, pcb_short_weight_key, pcb_spurious_copper_weight_key, pcb_via_weight_key, rare_patch_oversampling_factor_key, recognition_postprocess_kernel_size_key, recognition_threshold_key, t, validation_image_path_key, validation_label_path_key, validation_source_key):
    if pcb_defects_max_count_key not in descriptions:
        descriptions[pcb_defects_max_count_key] = str(
            t.get(
                'pcb_defects_max_count_tip',
                'Maximum number of sequential PCB defects generated for one augmented sample.',
            )
        )
    if pcb_break_weight_key not in labels_map:
        labels_map[pcb_break_weight_key] = str(t.get('pcb_break_severity_label', t.get('pcb_break', 'Break')))
    if pcb_break_weight_key not in descriptions:
        descriptions[pcb_break_weight_key] = str(
            t.get('pcb_break_severity_tip', 'At minimum creates small local breaks; at maximum produces larger conductor gaps.')
        )
    if pcb_short_weight_key not in labels_map:
        labels_map[pcb_short_weight_key] = str(t.get('pcb_short_severity_label', t.get('pcb_short', 'Short')))
    if pcb_short_weight_key not in descriptions:
        descriptions[pcb_short_weight_key] = str(
            t.get('pcb_short_severity_tip', 'At minimum bridges only the nearest traces; at maximum can short more distant traces.')
        )
    if pcb_missing_copper_weight_key not in labels_map:
        labels_map[pcb_missing_copper_weight_key] = str(t.get('pcb_missing_copper_severity_label', t.get('pcb_missing_copper', 'Missing copper')))
    if pcb_missing_copper_weight_key not in descriptions:
        descriptions[pcb_missing_copper_weight_key] = str(
            t.get('pcb_missing_copper_severity_tip', 'Controls how much copper is removed in one synthetic missing-copper defect.')
        )
    if pcb_excess_copper_weight_key not in labels_map:
        labels_map[pcb_excess_copper_weight_key] = str(t.get('pcb_excess_copper_severity_label', t.get('pcb_excess_copper', 'Excess copper')))
    if pcb_excess_copper_weight_key not in descriptions:
        descriptions[pcb_excess_copper_weight_key] = str(
            t.get('pcb_excess_copper_severity_tip', 'Controls how large synthetic copper growths and burrs become.')
        )
    if pcb_pinhole_weight_key not in labels_map:
        labels_map[pcb_pinhole_weight_key] = str(t.get('pcb_pinhole_severity_label', t.get('pcb_pinhole', 'Pinhole')))
    if pcb_pinhole_weight_key not in descriptions:
        descriptions[pcb_pinhole_weight_key] = str(
            t.get('pcb_pinhole_severity_tip', 'Controls the size of synthetic pinholes inside copper regions.')
        )
    if pcb_spurious_copper_weight_key not in labels_map:
        labels_map[pcb_spurious_copper_weight_key] = str(t.get('pcb_spurious_copper_severity_label', t.get('pcb_spurious_copper', 'Spurious copper')))
    if pcb_spurious_copper_weight_key not in descriptions:
        descriptions[pcb_spurious_copper_weight_key] = str(
            t.get('pcb_spurious_copper_severity_tip', 'Controls the size and reach of isolated synthetic copper islands.')
        )
    if pcb_via_weight_key not in labels_map:
        labels_map[pcb_via_weight_key] = str(t.get('pcb_via_severity_label', t.get('pcb_via', 'Via defect')))
    if pcb_via_weight_key not in descriptions:
        descriptions[pcb_via_weight_key] = str(
            t.get('pcb_via_severity_tip', 'Controls the amount of synthetic via shift, resize, and partial loss.')
        )
    if pcb_misalignment_weight_key not in labels_map:
        labels_map[pcb_misalignment_weight_key] = str(t.get('pcb_misalignment_severity_label', t.get('pcb_misalignment', 'Misalignment')))
    if pcb_misalignment_weight_key not in descriptions:
        descriptions[pcb_misalignment_weight_key] = str(
            t.get('pcb_misalignment_severity_tip', 'Controls the scale of local synthetic layer misregistration.')
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
    if validation_source_key not in labels_map:
        labels_map[validation_source_key] = str(t.get('validation_source_label', 'Validation source'))
    if validation_source_key not in descriptions:
        descriptions[validation_source_key] = str(
            t.get(
                'validation_source_tip',
                'Choose between validation split from the training dataset or a separate validation dataset.',
            )
        )
    if validation_image_path_key not in labels_map:
        labels_map[validation_image_path_key] = str(t.get('validation_image_path_label', 'Validation images'))
    if validation_image_path_key not in descriptions:
        descriptions[validation_image_path_key] = str(
            t.get(
                'validation_image_path_tip',
                'Folder with validation images used when external validation mode is selected.',
            )
        )
    if validation_label_path_key not in labels_map:
        labels_map[validation_label_path_key] = str(t.get('validation_label_path_label', 'Validation labels'))
    if validation_label_path_key not in descriptions:
        descriptions[validation_label_path_key] = str(
            t.get(
                'validation_label_path_tip',
                'Folder with validation labels used when external validation mode is selected.',
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
    panel.sync_patch_sizes_check_box.setText(
        str(labels_map.get('sync_patch_sizes', t.get('sync_patch_sizes', 'Use one patch size')))
    )
    panel.sync_patch_sizes_check_box.setToolTip(
        str(
            t.get(
                'sync_patch_sizes_tip',
                'If enabled, recognition patch size follows training patch size.',
            )
        )
    )
    train_patch_tip = str(t.get('train_patch_tip', t.get('sample_size_tip', '')))
    recognition_patch_tip = str(t.get('recognition_patch_tip', t.get('sample_size_tip', '')))
    panel.train_patch_size_groupbox.setTitle(
        str(labels_map.get('train_patch_size', t.get('train_patch_size', 'Training patch (X × Y)')))
    )
    panel.recognition_patch_size_groupbox.setTitle(
        str(labels_map.get('recognition_patch_size', t.get('recognition_patch_size', 'Recognition patch (X × Y)')))
    )
    panel.recognition_tab_patch_size_groupbox.setTitle(panel.recognition_patch_size_groupbox.title())
    panel.random_patch_min_size_groupbox.setTitle(
        str(t.get('random_patch_min_group', 'Minimum size'))
    )
    panel.random_patch_max_size_groupbox.setTitle(
        str(t.get('random_patch_max_group', 'Maximum size'))
    )
    panel.train_patch_size_groupbox.setToolTip(train_patch_tip)
    panel.recognition_patch_size_groupbox.setToolTip(recognition_patch_tip)
    panel.recognition_tab_patch_size_groupbox.setToolTip(recognition_patch_tip)
    panel.train_patch_size_widget.setToolTip(train_patch_tip)
    panel.recognition_patch_size_widget.setToolTip(recognition_patch_tip)
    panel.recognition_tab_patch_size_widget.setToolTip(recognition_patch_tip)
    random_min_tip = str(descriptions.get('random_patch_min_size', 'Minimum random patch size.'))
    random_max_tip = str(descriptions.get('random_patch_max_size', 'Maximum random patch size.'))
    panel.random_patch_min_size_groupbox.setToolTip(random_min_tip)
    panel.random_patch_max_size_groupbox.setToolTip(random_max_tip)
    panel.random_patch_min_size_widget.setToolTip(random_min_tip)
    panel.random_patch_max_size_widget.setToolTip(random_max_tip)
    axis_resize_texts = {
        "preview": str(t.get('axis_resize_preview', 'Image size preview')),
        "width_input": str(t.get('axis_resize_width_input', 'Width in pixels')),
        "height_input": str(t.get('axis_resize_height_input', 'Height in pixels')),
        "width_axis": str(t.get('axis_resize_width_axis', 'Horizontal size')),
        "height_axis": str(t.get('axis_resize_height_axis', 'Vertical size')),
        "pixels": str(t.get('axis_resize_pixels', 'Pixels')),
        "unit": str(t.get('axis_resize_unit', 'px')),
        "linked": str(
            t.get('axis_resize_linked', 'X and Y are linked. Click to edit them independently.')
        ),
        "unlinked": str(
            t.get('axis_resize_unlinked', 'X and Y are independent. Click to make them equal.')
        ),
        "height_linked": str(t.get('axis_resize_height_linked', 'Height is kept equal to width.')),
    }
    for axis_widget in (
        panel.train_patch_size_widget,
        panel.recognition_patch_size_widget,
        panel.recognition_tab_patch_size_widget,
        panel.random_patch_min_size_widget,
        panel.random_patch_max_size_widget,
        panel.synthetic_image_size_widget,
    ):
        axis_widget.apply_texts(axis_resize_texts)
    panel.random_patch_size_check_box.setText(
        str(t.get('random_patch_size_enable', 'Use random patch size'))
    )
    panel.random_patch_size_check_box.setToolTip(
        str(t.get('random_patch_size_tip', 'Available for online cutting; one aligned size is selected per batch.'))
    )
    panel.vertical_rotation.setText(str(t.get('rotate_180', 'Rotate frame by 180 degrees')))
    panel.vertical_rotation.setToolTip(str(t.get('rotate_180_tip', '')))
    panel.horizontal_rotation.setText(str(t.get('rotate_90', 'Rotate frame by 90 degrees')))
    panel.horizontal_rotation.setToolTip(str(t.get('rotate_90_tip', '')))
    panel.flip_x.setText(str(t.get('flip_x', 'Flip across X axis')))
    panel.flip_x.setToolTip(str(t.get('flip_x_tip', 'Mirror frames relative to the X axis.')))
    panel.flip_y.setText(str(t.get('flip_y', 'Flip across Y axis')))
    panel.flip_y.setToolTip(str(t.get('flip_y_tip', 'Mirror frames relative to the Y axis.')))
    panel.photometric_groupbox.setTitle(
        str(t.get('photometric_augmentation_toggle', t.get('extra_aug', 'Photometric augmentation')))
    )
    panel.photometric_groupbox.setToolTip(
        str(t.get('photometric_augmentation_toggle_tip', t.get('extra_aug_tip', '')))
    )
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
    panel.synthetic_defect_generator_groupbox.setTitle(
        str(
            t.get(
                'synthetic_defect_generator_enabled',
                'Enable synthetic topology generator',
            )
        )
    )
    panel.synthetic_defect_generator_groupbox.setToolTip(
        str(
            t.get(
                'synthetic_defect_generator_enabled_tip',
                'Generate synthetic image/mask pairs with non-intersecting procedural traces and optional image-only synthetic defects.',
            )
        )
    )
    panel.cutout_groupbox.setTitle(
        _read_text_from_mappings(
            (t, labels_map),
            ('cutout_enabled', 'cutout_enable'),
            'Enable cutout',
        )
    )
    panel.cutout_groupbox.setToolTip(
        _read_text_from_mappings(
            (t, descriptions),
            ('cutout_tip', 'cutout_enabled'),
            'Randomly erase rectangular regions in training images only.',
        )
    )
    panel.random_artifacts_groupbox.setTitle(
        _read_text_from_mappings(
            (t, labels_map),
            ('random_artifacts_enabled', 'random_artifacts_enable'),
            'Enable random artifacts',
        )
    )
    panel.random_artifacts_groupbox.setToolTip(
        _read_text_from_mappings(
            (t, descriptions),
            ('random_artifacts_tip', 'random_artifacts_enabled'),
            'Overlay synthetic textured artifacts on training images only.',
        )
    )
    for artifact_name, default_text in (
        ('dust', 'Dust'),
        ('resist_residue', 'Resist residue'),
        ('etch_residue', 'Etch residue'),
        ('particle_cluster', 'Particle cluster'),
        ('flake', 'Flake'),
    ):
        checkbox = panel.random_artifact_type_checkboxes[artifact_name]
        checkbox.setText(str(t.get(f'random_artifacts_type_{artifact_name}', default_text)))
        checkbox.setToolTip(
            str(
                t.get(
                    f'random_artifacts_type_{artifact_name}_tip',
                    f'Enable synthetic {default_text.lower()} artifacts.',
                )
            )
        )
    panel.mixup_groupbox.setTitle(
        _read_text_from_mappings(
            (t, labels_map),
            ('mixup_enabled', 'mixup_enable'),
            'Enable mixup',
        )
    )
    panel.mixup_groupbox.setToolTip(
        _read_text_from_mappings(
            (t, descriptions),
            ('mixup_tip', 'mixup_enabled'),
            'Mix pairs of training samples inside a batch using a random interpolation factor.',
        )
    )
    panel.augmentation_brightness_spinbox.setToolTip(str(t.get('extra_aug_brightness_tip', '')))
    panel.augmentation_contrast_spinbox.setToolTip(str(t.get('extra_aug_contrast_tip', '')))
    panel.augmentation_gamma_spinbox.setToolTip(str(t.get('extra_aug_gamma_tip', '')))
    panel.augmentation_noise_probability_spinbox.setToolTip(str(t.get('extra_aug_noise_prob_tip', '')))
    panel.augmentation_noise_sigma_spinbox.setToolTip(str(t.get('extra_aug_noise_sigma_tip', '')))
    panel.augmentation_blur_probability_spinbox.setToolTip(str(t.get('extra_aug_blur_prob_tip', '')))
    panel.augmentation_blur_radius_spinbox.setToolTip(str(t.get('extra_aug_blur_radius_tip', '')))
    panel.scale_augmentation_strength_spinbox.setToolTip(str(t.get('scale_augmentation_strength_tip', '')))
    panel.synthetic_dataset_factor_spinbox.setToolTip(str(t.get('synthetic_dataset_factor_tip', '')))
    panel.synthetic_image_size_widget.setToolTip(str(t.get('synthetic_image_size_tip', '')))
    panel.synthetic_topology_domain_combo.setToolTip(str(t.get('synthetic_topology_domain_tip', '')))
    panel.pcb_topology_family_combo.setToolTip(str(t.get('pcb_topology_family_tip', '')))
    panel.ic_topology_family_combo.setToolTip(str(t.get('ic_topology_family_tip', '')))
    if panel.synthetic_topology_domain_combo.count() >= 2:
        panel.synthetic_topology_domain_combo.setItemText(0, str(t.get('synthetic_topology_domain_pcb', 'PCB')))
        panel.synthetic_topology_domain_combo.setItemText(1, str(t.get('synthetic_topology_domain_ic', 'IC')))
