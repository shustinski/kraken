from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import QWidget

TEXT_FIELD_LOG_UPDATE_FREQUENCY = 'log_update_frequency'
TEXT_FIELD_RECOGNITION_JPEG_QUALITY = 'recognition_jpeg_quality'


def _copy_text_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return dict(value)


def _read_first_text(mapping: dict[str, Any], keys: tuple[str, ...], default: str = '') -> str:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return str(mapping[key])
    return default


def _read_text_from_mappings(
    mappings: tuple[dict[str, Any], ...],
    keys: tuple[str, ...],
    default: str = '',
) -> str:
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        value = _read_first_text(mapping, keys, '')
        if value:
            return value
    return default


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
    dataloader_workers_key = 'dataloader_num_workers'
    scheduler_name_key = 'scheduler_name'
    scheduler_plateau_factor_key = 'scheduler_plateau_factor'
    scheduler_plateau_patience_key = 'scheduler_plateau_patience'
    scheduler_plateau_threshold_key = 'scheduler_plateau_threshold'
    scheduler_plateau_min_lr_key = 'scheduler_plateau_min_lr'
    scheduler_plateau_cooldown_key = 'scheduler_plateau_cooldown'
    scheduler_cosine_t_max_key = 'scheduler_cosine_t_max'
    scheduler_cosine_eta_min_key = 'scheduler_cosine_eta_min'
    scheduler_one_cycle_max_lr_key = 'scheduler_one_cycle_max_lr'
    scheduler_one_cycle_pct_start_key = 'scheduler_one_cycle_pct_start'
    scheduler_one_cycle_anneal_strategy_key = 'scheduler_one_cycle_anneal_strategy'
    scheduler_one_cycle_div_factor_key = 'scheduler_one_cycle_div_factor'
    scheduler_one_cycle_final_div_factor_key = 'scheduler_one_cycle_final_div_factor'
    scheduler_one_cycle_three_phase_key = 'scheduler_one_cycle_three_phase'
    scheduler_step_lr_step_size_key = 'scheduler_step_lr_step_size'
    scheduler_step_lr_gamma_key = 'scheduler_step_lr_gamma'
    sync_patch_sizes_key = 'sync_patch_sizes'
    crops_per_image_key = 'crops_per_image'
    scale_augmentation_strength_key = 'scale_augmentation_strength'
    synthetic_dataset_factor_key = 'synthetic_dataset_factor'
    synthetic_topology_domain_key = 'synthetic_topology_domain'
    pcb_topology_family_key = 'pcb_topology_family'
    ic_topology_family_key = 'ic_topology_family'
    synthetic_trace_count_key = 'synthetic_trace_count'
    synthetic_segment_count_key = 'synthetic_segment_count'
    synthetic_trace_half_width_key = 'synthetic_trace_half_width'
    synthetic_background_noise_sigma_key = 'synthetic_background_noise_sigma'
    synthetic_trace_noise_sigma_key = 'synthetic_trace_noise_sigma'
    tech_aug_min_operations_key = 'tech_aug_min_operations'
    tech_aug_max_operations_key = 'tech_aug_max_operations'
    tech_aug_max_changed_pixels_ratio_key = 'tech_aug_max_changed_pixels_ratio'
    tech_aug_max_foreground_ratio_delta_key = 'tech_aug_max_foreground_ratio_delta'
    tech_aug_global_width_probability_key = 'tech_aug_global_width_probability'
    tech_aug_scale_rethreshold_probability_key = 'tech_aug_scale_rethreshold_probability'
    tech_aug_blur_threshold_probability_key = 'tech_aug_blur_threshold_probability'
    tech_aug_boundary_aware_probability_key = 'tech_aug_boundary_aware_probability'
    tech_aug_local_morphology_probability_key = 'tech_aug_local_morphology_probability'
    tech_aug_gap_variation_probability_key = 'tech_aug_gap_variation_probability'
    augmentation_gamma_strength_key = 'augmentation_gamma_strength'
    augmentation_blur_probability_key = 'augmentation_blur_probability'
    augmentation_blur_radius_key = 'augmentation_blur_radius'
    cutout_probability_key = 'cutout_probability'
    cutout_holes_key = 'cutout_holes'
    cutout_size_ratio_key = 'cutout_size_ratio'
    random_artifacts_probability_key = 'random_artifacts_probability'
    random_artifacts_count_key = 'random_artifacts_count'
    random_artifacts_size_ratio_key = 'random_artifacts_size_ratio'
    mixup_probability_key = 'mixup_probability'
    mixup_alpha_key = 'mixup_alpha'
    pcb_defects_probability_key = 'pcb_defects_probability'
    pcb_defects_min_count_key = 'pcb_defects_min_count'
    pcb_defects_max_count_key = 'pcb_defects_max_count'
    pcb_break_weight_key = 'pcb_break_severity'
    pcb_short_weight_key = 'pcb_short_severity'
    pcb_missing_copper_weight_key = 'pcb_missing_copper_severity'
    pcb_excess_copper_weight_key = 'pcb_excess_copper_severity'
    pcb_pinhole_weight_key = 'pcb_pinhole_severity'
    pcb_spurious_copper_weight_key = 'pcb_spurious_copper_severity'
    pcb_via_weight_key = 'pcb_via_severity'
    pcb_misalignment_weight_key = 'pcb_misalignment_severity'
    ic_line_break_weight_key = 'ic_line_break_severity'
    ic_bridge_weight_key = 'ic_bridge_severity'
    ic_necking_weight_key = 'ic_necking_severity'
    ic_missing_metal_weight_key = 'ic_missing_metal_severity'
    ic_spur_weight_key = 'ic_spur_severity'
    ic_pinhole_weight_key = 'ic_pinhole_severity'
    ic_via_open_weight_key = 'ic_via_open_severity'
    ic_line_shift_weight_key = 'ic_line_shift_severity'
    pcb_break_toggle_key = 'pcb_break'
    pcb_short_toggle_key = 'pcb_short'
    pcb_missing_copper_toggle_key = 'pcb_missing_copper'
    pcb_excess_copper_toggle_key = 'pcb_excess_copper'
    pcb_pinhole_toggle_key = 'pcb_pinhole'
    pcb_spurious_copper_toggle_key = 'pcb_spurious_copper'
    pcb_via_toggle_key = 'pcb_via'
    pcb_misalignment_toggle_key = 'pcb_misalignment'
    ic_line_break_toggle_key = 'ic_line_break'
    ic_bridge_toggle_key = 'ic_bridge'
    ic_necking_toggle_key = 'ic_necking'
    ic_missing_metal_toggle_key = 'ic_missing_metal'
    ic_spur_toggle_key = 'ic_spur'
    ic_pinhole_toggle_key = 'ic_pinhole'
    ic_via_open_toggle_key = 'ic_via_open'
    ic_line_shift_toggle_key = 'ic_line_shift'
    labels_map.setdefault(synthetic_topology_domain_key, str(t.get('synthetic_topology_domain', 'Synthetic domain')))
    labels_map.setdefault(pcb_topology_family_key, str(t.get('pcb_topology_family', 'PCB topology family')))
    labels_map.setdefault(ic_topology_family_key, str(t.get('ic_topology_family', 'IC topology family')))
    labels_map.setdefault(ic_line_break_weight_key, str(t.get('ic_line_break_severity_label', t.get('ic_line_break', 'Line break'))))
    labels_map.setdefault(ic_bridge_weight_key, str(t.get('ic_bridge_severity_label', t.get('ic_bridge', 'Bridge'))))
    labels_map.setdefault(ic_necking_weight_key, str(t.get('ic_necking_severity_label', t.get('ic_necking', 'Necking'))))
    labels_map.setdefault(ic_missing_metal_weight_key, str(t.get('ic_missing_metal_severity_label', t.get('ic_missing_metal', 'Missing metal'))))
    labels_map.setdefault(ic_spur_weight_key, str(t.get('ic_spur_severity_label', t.get('ic_spur', 'Spur'))))
    labels_map.setdefault(ic_pinhole_weight_key, str(t.get('ic_pinhole_severity_label', t.get('ic_pinhole', 'Pinhole'))))
    labels_map.setdefault(ic_via_open_weight_key, str(t.get('ic_via_open_severity_label', t.get('ic_via_open', 'Via open'))))
    labels_map.setdefault(ic_line_shift_weight_key, str(t.get('ic_line_shift_severity_label', t.get('ic_line_shift', 'Line shift'))))
    labels_map.setdefault(pcb_break_toggle_key, str(t.get('pcb_break', labels_map.get(pcb_break_weight_key, 'Break'))))
    labels_map.setdefault(pcb_short_toggle_key, str(t.get('pcb_short', labels_map.get(pcb_short_weight_key, 'Short'))))
    labels_map.setdefault(pcb_missing_copper_toggle_key, str(t.get('pcb_missing_copper', labels_map.get(pcb_missing_copper_weight_key, 'Missing copper'))))
    labels_map.setdefault(pcb_excess_copper_toggle_key, str(t.get('pcb_excess_copper', labels_map.get(pcb_excess_copper_weight_key, 'Excess copper'))))
    labels_map.setdefault(pcb_pinhole_toggle_key, str(t.get('pcb_pinhole', labels_map.get(pcb_pinhole_weight_key, 'Pinhole'))))
    labels_map.setdefault(pcb_spurious_copper_toggle_key, str(t.get('pcb_spurious_copper', labels_map.get(pcb_spurious_copper_weight_key, 'Spurious copper'))))
    labels_map.setdefault(pcb_via_toggle_key, str(t.get('pcb_via', labels_map.get(pcb_via_weight_key, 'Via defect'))))
    labels_map.setdefault(pcb_misalignment_toggle_key, str(t.get('pcb_misalignment', labels_map.get(pcb_misalignment_weight_key, 'Misalignment'))))
    labels_map.setdefault(ic_line_break_toggle_key, str(t.get('ic_line_break', labels_map.get(ic_line_break_weight_key, 'Line break'))))
    labels_map.setdefault(ic_bridge_toggle_key, str(t.get('ic_bridge', labels_map.get(ic_bridge_weight_key, 'Bridge'))))
    labels_map.setdefault(ic_necking_toggle_key, str(t.get('ic_necking', labels_map.get(ic_necking_weight_key, 'Necking'))))
    labels_map.setdefault(ic_missing_metal_toggle_key, str(t.get('ic_missing_metal', labels_map.get(ic_missing_metal_weight_key, 'Missing metal'))))
    labels_map.setdefault(ic_spur_toggle_key, str(t.get('ic_spur', labels_map.get(ic_spur_weight_key, 'Spur'))))
    labels_map.setdefault(ic_pinhole_toggle_key, str(t.get('ic_pinhole', labels_map.get(ic_pinhole_weight_key, 'Pinhole'))))
    labels_map.setdefault(ic_via_open_toggle_key, str(t.get('ic_via_open', labels_map.get(ic_via_open_weight_key, 'Via open'))))
    labels_map.setdefault(ic_line_shift_toggle_key, str(t.get('ic_line_shift', labels_map.get(ic_line_shift_weight_key, 'Line shift'))))
    rare_patch_oversampling_factor_key = 'rare_patch_oversampling_factor'
    recognition_threshold_key = 'recognition_threshold'
    deprecated_models_key = 'deprecated_models'
    experimental_models_key = 'experimental_models'
    epochs_key = 'epochs'
    recognition_tta_key = 'recognition_tta'
    confidence_save_mode_key = 'confidence_save_mode'
    deep_supervision_key = 'deep_supervision'
    recognition_postprocess_kernel_size_key = 'recognition_postprocess_kernel_size'
    validation_source_key = 'validation_source'
    validation_image_path_key = 'validation_image_path'
    validation_label_path_key = 'validation_label_path'
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
    if synthetic_dataset_factor_key not in labels_map:
        labels_map[synthetic_dataset_factor_key] = str(t.get('synthetic_dataset_factor', 'Synthetic epoch factor'))
    if synthetic_dataset_factor_key not in descriptions:
        descriptions[synthetic_dataset_factor_key] = str(
            t.get(
                'synthetic_dataset_factor_tip',
                'How many synthetic samples to generate per epoch relative to the real train dataset size.',
            )
        )
    if synthetic_trace_count_key not in labels_map:
        labels_map[synthetic_trace_count_key] = str(t.get('synthetic_trace_count', 'Trace count'))
    if synthetic_trace_count_key not in descriptions:
        descriptions[synthetic_trace_count_key] = str(
            t.get(
                'synthetic_trace_count_tip',
                'Number of conductive traces generated in one synthetic sample.',
            )
        )
    if synthetic_segment_count_key not in labels_map:
        labels_map[synthetic_segment_count_key] = str(t.get('synthetic_segment_count', 'Segments per trace'))
    if synthetic_segment_count_key not in descriptions:
        descriptions[synthetic_segment_count_key] = str(
            t.get(
                'synthetic_segment_count_tip',
                'Number of Manhattan segments used to build each synthetic trace.',
            )
        )
    if synthetic_trace_half_width_key not in labels_map:
        labels_map[synthetic_trace_half_width_key] = str(t.get('synthetic_trace_half_width', 'Trace half-width'))
    if synthetic_trace_half_width_key not in descriptions:
        descriptions[synthetic_trace_half_width_key] = str(
            t.get(
                'synthetic_trace_half_width_tip',
                'Half-width of the generated copper traces in pixels.',
            )
        )
    if synthetic_background_noise_sigma_key not in labels_map:
        labels_map[synthetic_background_noise_sigma_key] = str(t.get('synthetic_background_noise_sigma', 'Background noise sigma'))
    if synthetic_background_noise_sigma_key not in descriptions:
        descriptions[synthetic_background_noise_sigma_key] = str(
            t.get(
                'synthetic_background_noise_sigma_tip',
                'Standard deviation of the low-level grayscale noise in synthetic images.',
            )
        )
    if synthetic_trace_noise_sigma_key not in labels_map:
        labels_map[synthetic_trace_noise_sigma_key] = str(t.get('synthetic_trace_noise_sigma', 'Trace noise sigma'))
    if synthetic_trace_noise_sigma_key not in descriptions:
        descriptions[synthetic_trace_noise_sigma_key] = str(
            t.get(
                'synthetic_trace_noise_sigma_tip',
                'Standard deviation of the grayscale noise applied only inside synthetic traces.',
            )
        )
    if jpeg_quality_key not in labels_map:
        labels_map[jpeg_quality_key] = 'JPEG quality'
    if jpeg_quality_key not in descriptions:
        descriptions[jpeg_quality_key] = str(
            t.get('recognition_jpeg_quality_tip', 'JPEG quality for recognition output (1..100).')
        )
    if deprecated_models_key not in labels_map:
        labels_map[deprecated_models_key] = str(t.get('deprecated_models', 'Deprecated models'))
    if deprecated_models_key not in descriptions:
        descriptions[deprecated_models_key] = str(
            t.get(
                'deprecated_models_tip',
                'Older architectures kept only for backward compatibility and comparison runs.',
            )
        )
    if experimental_models_key not in labels_map:
        labels_map[experimental_models_key] = str(t.get('experimental_models', 'Experimental models'))
    if experimental_models_key not in descriptions:
        descriptions[experimental_models_key] = str(
            t.get(
                'experimental_models_tip',
                'Architectures under evaluation that may require extra validation before production use.',
            )
        )
    if epochs_key not in labels_map:
        labels_map[epochs_key] = str(t.get('epochs', 'Epoch count'))
    if epochs_key not in descriptions:
        descriptions[epochs_key] = str(
            t.get('epochs_tip', 'How many full passes over the training data to run.')
        )
    if confidence_save_mode_key not in labels_map:
        labels_map[confidence_save_mode_key] = str(t.get('confidence_save_mode_label', 'Confidence map'))
    if confidence_save_mode_key not in descriptions:
        descriptions[confidence_save_mode_key] = str(
            t.get(
                'confidence_save_mode_tip',
                'Choose whether to save a separate confidence map next to the main recognition mask.',
            )
        )
    if recognition_tta_key not in labels_map:
        labels_map[recognition_tta_key] = str(t.get('recognition_tta', 'Use TTA for recognition'))
    if recognition_tta_key not in descriptions:
        descriptions[recognition_tta_key] = str(
            t.get(
                'recognition_tta_tip',
                'Apply test-time augmentation to the main recognition mask during inference.',
            )
        )
    if deep_supervision_key not in labels_map:
        labels_map[deep_supervision_key] = str(t.get('deep_supervision', 'Enable deep supervision'))
    if deep_supervision_key not in descriptions:
        descriptions[deep_supervision_key] = str(
            t.get(
                'deep_supervision_tip',
                'Train convolutional segmentation models with auxiliary decoder heads.',
            )
        )
    if dataloader_workers_key not in labels_map:
        labels_map[dataloader_workers_key] = str(t.get('dataloader_num_workers_label', 'DataLoader workers'))
    if dataloader_workers_key not in descriptions:
        descriptions[dataloader_workers_key] = str(
            t.get(
                'dataloader_num_workers_tip',
                'Number of worker processes used by the training DataLoader. Use -1 for automatic selection.',
            )
        )
    if scheduler_name_key not in labels_map:
        labels_map[scheduler_name_key] = str(t.get('scheduler_name_label', 'Scheduler'))
    if scheduler_name_key not in descriptions:
        descriptions[scheduler_name_key] = str(
            t.get(
                'scheduler_tip',
                'Learning-rate schedule applied after optimizer updates.',
            )
        )
    if scheduler_plateau_factor_key not in labels_map:
        labels_map[scheduler_plateau_factor_key] = 'Plateau factor'
    if scheduler_plateau_factor_key not in descriptions:
        descriptions[scheduler_plateau_factor_key] = str(
            t.get(
                'scheduler_plateau_factor_tip',
                'Multiplier applied to the learning rate when ReduceLROnPlateau triggers.',
            )
        )
    if scheduler_plateau_patience_key not in labels_map:
        labels_map[scheduler_plateau_patience_key] = 'Plateau patience'
    if scheduler_plateau_patience_key not in descriptions:
        descriptions[scheduler_plateau_patience_key] = str(
            t.get(
                'scheduler_plateau_patience_tip',
                'How many epochs without improvement to wait before reducing the learning rate.',
            )
        )
    if scheduler_plateau_threshold_key not in labels_map:
        labels_map[scheduler_plateau_threshold_key] = 'Plateau threshold'
    if scheduler_plateau_threshold_key not in descriptions:
        descriptions[scheduler_plateau_threshold_key] = str(
            t.get(
                'scheduler_plateau_threshold_tip',
                'Minimum loss improvement that counts as progress for ReduceLROnPlateau.',
            )
        )
    if scheduler_plateau_min_lr_key not in labels_map:
        labels_map[scheduler_plateau_min_lr_key] = 'Plateau min LR'
    if scheduler_plateau_min_lr_key not in descriptions:
        descriptions[scheduler_plateau_min_lr_key] = str(
            t.get(
                'scheduler_plateau_min_lr_tip',
                'Lower bound for the learning rate used by ReduceLROnPlateau.',
            )
        )
    if scheduler_plateau_cooldown_key not in labels_map:
        labels_map[scheduler_plateau_cooldown_key] = 'Plateau cooldown'
    if scheduler_plateau_cooldown_key not in descriptions:
        descriptions[scheduler_plateau_cooldown_key] = str(
            t.get(
                'scheduler_plateau_cooldown_tip',
                'How many epochs to wait after a reduction before resuming Plateau checks.',
            )
        )
    if scheduler_cosine_t_max_key not in labels_map:
        labels_map[scheduler_cosine_t_max_key] = 'Cosine T_max'
    if scheduler_cosine_t_max_key not in descriptions:
        descriptions[scheduler_cosine_t_max_key] = str(
            t.get(
                'scheduler_cosine_t_max_tip',
                'Number of scheduler epochs used by one cosine annealing cycle.',
            )
        )
    if scheduler_cosine_eta_min_key not in labels_map:
        labels_map[scheduler_cosine_eta_min_key] = 'Cosine eta_min'
    if scheduler_cosine_eta_min_key not in descriptions:
        descriptions[scheduler_cosine_eta_min_key] = str(
            t.get(
                'scheduler_cosine_eta_min_tip',
                'Minimum learning rate reached by CosineAnnealingLR.',
            )
        )
    if scheduler_one_cycle_max_lr_key not in labels_map:
        labels_map[scheduler_one_cycle_max_lr_key] = 'OneCycle max LR'
    if scheduler_one_cycle_max_lr_key not in descriptions:
        descriptions[scheduler_one_cycle_max_lr_key] = str(
            t.get(
                'scheduler_one_cycle_max_lr_tip',
                'Peak learning rate reached by OneCycleLR.',
            )
        )
    if scheduler_one_cycle_pct_start_key not in labels_map:
        labels_map[scheduler_one_cycle_pct_start_key] = 'OneCycle pct_start'
    if scheduler_one_cycle_pct_start_key not in descriptions:
        descriptions[scheduler_one_cycle_pct_start_key] = str(
            t.get(
                'scheduler_one_cycle_pct_start_tip',
                'Fraction of training steps spent increasing the learning rate in OneCycleLR.',
            )
        )
    if scheduler_one_cycle_anneal_strategy_key not in labels_map:
        labels_map[scheduler_one_cycle_anneal_strategy_key] = 'OneCycle anneal'
    if scheduler_one_cycle_anneal_strategy_key not in descriptions:
        descriptions[scheduler_one_cycle_anneal_strategy_key] = str(
            t.get(
                'scheduler_one_cycle_anneal_strategy_tip',
                'Shape of the learning-rate decay curve in OneCycleLR.',
            )
        )
    if scheduler_one_cycle_div_factor_key not in labels_map:
        labels_map[scheduler_one_cycle_div_factor_key] = 'OneCycle div factor'
    if scheduler_one_cycle_div_factor_key not in descriptions:
        descriptions[scheduler_one_cycle_div_factor_key] = str(
            t.get(
                'scheduler_one_cycle_div_factor_tip',
                'Initial LR is max_lr divided by this factor in OneCycleLR.',
            )
        )
    if scheduler_one_cycle_final_div_factor_key not in labels_map:
        labels_map[scheduler_one_cycle_final_div_factor_key] = 'OneCycle final div'
    if scheduler_one_cycle_final_div_factor_key not in descriptions:
        descriptions[scheduler_one_cycle_final_div_factor_key] = str(
            t.get(
                'scheduler_one_cycle_final_div_factor_tip',
                'Minimum LR is initial LR divided by this factor in OneCycleLR.',
            )
        )
    if scheduler_one_cycle_three_phase_key not in labels_map:
        labels_map[scheduler_one_cycle_three_phase_key] = 'OneCycle three phase'
    if scheduler_one_cycle_three_phase_key not in descriptions:
        descriptions[scheduler_one_cycle_three_phase_key] = str(
            t.get(
                'scheduler_one_cycle_three_phase_tip',
                'Use the original three-phase OneCycle schedule instead of the fast two-phase variant.',
            )
        )
    if scheduler_step_lr_step_size_key not in labels_map:
        labels_map[scheduler_step_lr_step_size_key] = 'StepLR step size'
    if scheduler_step_lr_step_size_key not in descriptions:
        descriptions[scheduler_step_lr_step_size_key] = str(
            t.get(
                'scheduler_step_lr_step_size_tip',
                'How many epochs to wait between StepLR reductions.',
            )
        )
    if scheduler_step_lr_gamma_key not in labels_map:
        labels_map[scheduler_step_lr_gamma_key] = 'StepLR gamma'
    if scheduler_step_lr_gamma_key not in descriptions:
        descriptions[scheduler_step_lr_gamma_key] = str(
            t.get(
                'scheduler_step_lr_gamma_tip',
                'Multiplier applied to the learning rate on each StepLR decay step.',
            )
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
    if tech_aug_min_operations_key not in labels_map:
        labels_map[tech_aug_min_operations_key] = 'Tech aug min ops'
    if tech_aug_min_operations_key not in descriptions:
        descriptions[tech_aug_min_operations_key] = str(
            t.get(
                'tech_aug_min_operations_tip',
                'Minimum number of geometry variation operators to sample for one mask.',
            )
        )
    if tech_aug_max_operations_key not in labels_map:
        labels_map[tech_aug_max_operations_key] = 'Tech aug max ops'
    if tech_aug_max_operations_key not in descriptions:
        descriptions[tech_aug_max_operations_key] = str(
            t.get(
                'tech_aug_max_operations_tip',
                'Maximum number of geometry variation operators to sample for one mask.',
            )
        )
    if tech_aug_max_changed_pixels_ratio_key not in labels_map:
        labels_map[tech_aug_max_changed_pixels_ratio_key] = 'Changed pixels limit'
    if tech_aug_max_changed_pixels_ratio_key not in descriptions:
        descriptions[tech_aug_max_changed_pixels_ratio_key] = str(
            t.get(
                'tech_aug_max_changed_pixels_ratio_tip',
                'Reject augmented masks that modify too many pixels relative to the reference topology.',
            )
        )
    if tech_aug_max_foreground_ratio_delta_key not in labels_map:
        labels_map[tech_aug_max_foreground_ratio_delta_key] = 'Foreground delta limit'
    if tech_aug_max_foreground_ratio_delta_key not in descriptions:
        descriptions[tech_aug_max_foreground_ratio_delta_key] = str(
            t.get(
                'tech_aug_max_foreground_ratio_delta_tip',
                'Reject augmented masks when total metal fill changes too much.',
            )
        )
    if tech_aug_global_width_probability_key not in labels_map:
        labels_map[tech_aug_global_width_probability_key] = 'Width variation prob.'
    if tech_aug_global_width_probability_key not in descriptions:
        descriptions[tech_aug_global_width_probability_key] = str(
            t.get(
                'tech_aug_global_width_probability_tip',
                'Probability of global dilation or erosion that changes line width.',
            )
        )
    if tech_aug_scale_rethreshold_probability_key not in labels_map:
        labels_map[tech_aug_scale_rethreshold_probability_key] = 'Scale/rethreshold prob.'
    if tech_aug_scale_rethreshold_probability_key not in descriptions:
        descriptions[tech_aug_scale_rethreshold_probability_key] = str(
            t.get(
                'tech_aug_scale_rethreshold_probability_tip',
                'Probability of scale drift simulated via resize and rethreshold.',
            )
        )
    if tech_aug_blur_threshold_probability_key not in labels_map:
        labels_map[tech_aug_blur_threshold_probability_key] = 'Blur/rethreshold prob.'
    if tech_aug_blur_threshold_probability_key not in descriptions:
        descriptions[tech_aug_blur_threshold_probability_key] = str(
            t.get(
                'tech_aug_blur_threshold_probability_tip',
                'Probability of edge smoothing followed by binary rethreshold.',
            )
        )
    if tech_aug_boundary_aware_probability_key not in labels_map:
        labels_map[tech_aug_boundary_aware_probability_key] = 'Boundary-aware prob.'
    if tech_aug_boundary_aware_probability_key not in descriptions:
        descriptions[tech_aug_boundary_aware_probability_key] = str(
            t.get(
                'tech_aug_boundary_aware_probability_tip',
                'Probability of perturbing only a narrow band around polygon boundaries.',
            )
        )
    if tech_aug_local_morphology_probability_key not in labels_map:
        labels_map[tech_aug_local_morphology_probability_key] = 'Local morphology prob.'
    if tech_aug_local_morphology_probability_key not in descriptions:
        descriptions[tech_aug_local_morphology_probability_key] = str(
            t.get(
                'tech_aug_local_morphology_probability_tip',
                'Probability of local dilation or erosion inside random regions of interest.',
            )
        )
    if tech_aug_gap_variation_probability_key not in labels_map:
        labels_map[tech_aug_gap_variation_probability_key] = 'Gap open/close prob.'
    if tech_aug_gap_variation_probability_key not in descriptions:
        descriptions[tech_aug_gap_variation_probability_key] = str(
            t.get(
                'tech_aug_gap_variation_probability_tip',
                'Probability of closing narrow gaps or opening thin bridges.',
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
    if random_artifacts_probability_key not in labels_map:
        labels_map[random_artifacts_probability_key] = 'Random artifacts probability'
    if random_artifacts_probability_key not in descriptions:
        descriptions[random_artifacts_probability_key] = str(
            t.get(
                'random_artifacts_probability_tip',
                'Probability of adding synthetic artifacts to a training sample.',
            )
        )
    if random_artifacts_count_key not in labels_map:
        labels_map[random_artifacts_count_key] = 'Random artifacts count'
    if random_artifacts_count_key not in descriptions:
        descriptions[random_artifacts_count_key] = str(
            t.get(
                'random_artifacts_count_tip',
                'How many synthetic artifacts may be added to one training sample.',
            )
        )
    if random_artifacts_size_ratio_key not in labels_map:
        labels_map[random_artifacts_size_ratio_key] = 'Random artifacts size ratio'
    if random_artifacts_size_ratio_key not in descriptions:
        descriptions[random_artifacts_size_ratio_key] = str(
            t.get(
                'random_artifacts_size_ratio_tip',
                'Maximum artifact size relative to patch width and height.',
            )
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
    if pcb_defects_probability_key not in labels_map:
        labels_map[pcb_defects_probability_key] = 'PCB defect probability'
    if pcb_defects_probability_key not in descriptions:
        descriptions[pcb_defects_probability_key] = str(
            t.get(
                'pcb_defects_probability_tip',
                'Probability of injecting one or more synthetic PCB defects into a training sample.',
            )
        )
    if pcb_defects_min_count_key not in labels_map:
        labels_map[pcb_defects_min_count_key] = 'Min defect count'
    if pcb_defects_min_count_key not in descriptions:
        descriptions[pcb_defects_min_count_key] = str(
            t.get(
                'pcb_defects_min_count_tip',
                'Minimum number of sequential PCB defects generated for one augmented sample.',
            )
        )
    if pcb_defects_max_count_key not in labels_map:
        labels_map[pcb_defects_max_count_key] = 'Max defect count'
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
    panel.flip_x.setText(str(t.get('flip_x', 'Flip across X axis')))
    panel.flip_x.setToolTip(str(t.get('flip_x_tip', 'Mirror frames relative to the X axis.')))
    panel.flip_y.setText(str(t.get('flip_y', 'Flip across Y axis')))
    panel.flip_y.setToolTip(str(t.get('flip_y_tip', 'Mirror frames relative to the Y axis.')))
    panel.additional_augmentation_check_box.setText(
        str(t.get('photometric_augmentation_toggle', t.get('extra_aug', 'Photometric augmentation')))
    )
    panel.additional_augmentation_check_box.setToolTip(
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
    panel.synthetic_defect_generator_check_box.setText(
        str(
            t.get(
                'synthetic_defect_generator_enabled',
                'Enable synthetic topology generator',
            )
        )
    )
    panel.synthetic_defect_generator_check_box.setToolTip(
        str(
            t.get(
                'synthetic_defect_generator_enabled_tip',
                'Generate synthetic image/mask pairs with non-intersecting procedural traces and optional image-only synthetic defects.',
            )
        )
    )
    panel.cutout_check_box.setText(
        _read_text_from_mappings(
            (t, labels_map),
            ('cutout_enabled', 'cutout_enable'),
            'Enable cutout',
        )
    )
    panel.cutout_check_box.setToolTip(
        _read_text_from_mappings(
            (t, descriptions),
            ('cutout_tip', 'cutout_enabled'),
            'Randomly erase rectangular regions in training images only.',
        )
    )
    panel.random_artifacts_check_box.setText(
        _read_text_from_mappings(
            (t, labels_map),
            ('random_artifacts_enabled', 'random_artifacts_enable'),
            'Enable random artifacts',
        )
    )
    panel.random_artifacts_check_box.setToolTip(
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
    panel.mixup_check_box.setText(
        _read_text_from_mappings(
            (t, labels_map),
            ('mixup_enabled', 'mixup_enable'),
            'Enable mixup',
        )
    )
    panel.mixup_check_box.setToolTip(
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
    if panel.pcb_topology_family_combo.count() >= 3:
        panel.pcb_topology_family_combo.setItemText(0, str(t.get('pcb_topology_family_parallel', 'Parallel traces')))
        panel.pcb_topology_family_combo.setItemText(1, str(t.get('pcb_topology_family_independent', 'Independent traces')))
        panel.pcb_topology_family_combo.setItemText(2, str(t.get('pcb_topology_family_fanout', 'Fanout tree')))
    if panel.ic_topology_family_combo.count() >= 3:
        panel.ic_topology_family_combo.setItemText(0, str(t.get('ic_topology_family_channels', 'Routing channels')))
        panel.ic_topology_family_combo.setItemText(1, str(t.get('ic_topology_family_cell_array', 'Cell array')))
        panel.ic_topology_family_combo.setItemText(2, str(t.get('ic_topology_family_tree', 'Tree routing')))
    panel.synthetic_trace_count_range_widget.setToolTip(str(t.get('synthetic_trace_count_tip', '')))
    panel.synthetic_segment_count_range_widget.setToolTip(str(t.get('synthetic_segment_count_tip', '')))
    panel.synthetic_trace_half_width_range_widget.setToolTip(str(t.get('synthetic_trace_half_width_tip', '')))
    panel.synthetic_background_noise_sigma_range_widget.setToolTip(
        str(t.get('synthetic_background_noise_sigma_tip', ''))
    )
    panel.synthetic_trace_noise_sigma_range_widget.setToolTip(str(t.get('synthetic_trace_noise_sigma_tip', '')))
    panel.cutout_probability_spinbox.setToolTip(str(t.get('cutout_probability_tip', '')))
    panel.cutout_holes_spinbox.setToolTip(str(t.get('cutout_holes_tip', '')))
    panel.cutout_size_ratio_spinbox.setToolTip(str(t.get('cutout_size_ratio_tip', '')))
    panel.random_artifacts_probability_spinbox.setToolTip(str(t.get('random_artifacts_probability_tip', '')))
    panel.random_artifacts_count_spinbox.setToolTip(str(t.get('random_artifacts_count_tip', '')))
    panel.random_artifacts_size_ratio_spinbox.setToolTip(str(t.get('random_artifacts_size_ratio_tip', '')))
    panel.mixup_probability_spinbox.setToolTip(str(t.get('mixup_probability_tip', '')))
    panel.mixup_alpha_spinbox.setToolTip(str(t.get('mixup_alpha_tip', '')))
    panel.pcb_defects_probability_spinbox.setToolTip(str(t.get('pcb_defects_probability_tip', '')))
    panel.pcb_defects_min_count_spinbox.setToolTip(str(t.get('pcb_defects_min_count_tip', '')))
    panel.pcb_defects_max_count_spinbox.setToolTip(str(t.get('pcb_defects_max_count_tip', '')))
    panel.synthetic_defect_generator_groupbox.setToolTip(
        str(
            t.get(
                'synthetic_defect_generator_group_tip',
                'Generate additional fully synthetic topology samples and image-only synthetic defects for training.',
            )
        )
    )
    for defect_name, text_key in (
        ('break', 'pcb_break_severity_tip'),
        ('short', 'pcb_short_severity_tip'),
        ('missing_copper', 'pcb_missing_copper_severity_tip'),
        ('excess_copper', 'pcb_excess_copper_severity_tip'),
        ('pinhole', 'pcb_pinhole_severity_tip'),
        ('spurious_copper', 'pcb_spurious_copper_severity_tip'),
        ('via', 'pcb_via_severity_tip'),
        ('misalignment', 'pcb_misalignment_severity_tip'),
    ):
        panel.pcb_defect_type_spinboxes[defect_name].setToolTip(str(t.get(text_key, '')))
    for defect_name, text_key in (
        ('line_break', 'ic_line_break_severity_tip'),
        ('bridge', 'ic_bridge_severity_tip'),
        ('necking', 'ic_necking_severity_tip'),
        ('missing_metal', 'ic_missing_metal_severity_tip'),
        ('spur', 'ic_spur_severity_tip'),
        ('pinhole', 'ic_pinhole_severity_tip'),
        ('via_open', 'ic_via_open_severity_tip'),
        ('line_shift', 'ic_line_shift_severity_tip'),
    ):
        panel.ic_defect_type_spinboxes[defect_name].setToolTip(str(t.get(text_key, '')))
    panel.shift_spinbox.setToolTip(str(t.get('shift_tip', '')))
    panel.validation_check_box.setText(str(t.get('validation', 'Use validation during training')))
    panel.validation_check_box.setToolTip(str(t.get('validation_tip', '')))
    panel.save_validation_binary_images_check_box.setText(
        str(t.get('save_validation_binary_images', 'Save binary validation predictions after each epoch'))
    )
    panel.save_validation_binary_images_check_box.setToolTip(
        str(
            t.get(
                'save_validation_binary_images_tip',
                'Save binary predictions for validation samples after each epoch into the run artifact folder.',
            )
        )
    )
    panel.validation_mode_combo.setToolTip(
        str(
            t.get(
                'validation_source_tip',
                'Choose between validation split from the training dataset or a separate validation dataset.',
            )
        )
    )
    split_text = str(t.get('validation_source_split', 'Split from training dataset'))
    external_text = str(t.get('validation_source_external', 'Use external image/label folders'))
    panel.validation_mode_combo.setItemText(0, split_text)
    panel.validation_mode_combo.setItemText(1, external_text)
    panel.validation_image_path_label.setToolTip(
        str(
            t.get(
                'validation_image_path_tip',
                'Folder with validation images used when external validation mode is selected.',
            )
        )
    )
    panel.validation_label_path_label.setToolTip(
        str(
            t.get(
                'validation_label_path_tip',
                'Folder with validation labels used when external validation mode is selected.',
            )
        )
    )
    panel.general_groupbox.setTitle(str(t.get('general_group', 'Data and model')))
    panel.augmentation_groupbox.setTitle(str(t.get('augmentation_group', 'Augmentation and shift')))
    panel.shuffle_groupbox.setTitle(str(t.get('shuffle_group', 'Shuffle order')))
    panel.spatial_groupbox.setTitle(str(t.get('spatial_group', 'Spatial augmentations')))
    panel.photometric_groupbox.setTitle(str(t.get('photometric_group', 'Photometric augmentations')))
    panel.cutout_groupbox.setTitle(str(t.get('cutout', 'Cutout')))
    panel.random_artifacts_groupbox.setTitle(str(t.get('random_artifacts', 'Random artifacts')))
    panel.mixup_groupbox.setTitle(str(t.get('mixup', 'Mixup')))
    panel.synthetic_defect_generator_groupbox.setTitle(
        str(t.get('synthetic_defect_generator_group', 'Synthetic topology'))
    )
    panel.augmentation_preview_button.setText(
        str(t.get('augmentation_preview_button', 'Open augmentation preview'))
    )
    panel.augmentation_preview_button.setToolTip(
        str(
            t.get(
                'augmentation_preview_button_tip',
                'Open a preview window that applies the selected training augmentations to dataset samples.',
            )
        )
    )
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
    panel.precision_loss_groupbox.setTitle(str(t.get('loss_precision_group', 'Loss function')))
    panel.optimizer_advanced_groupbox.setTitle(str(t.get('advanced_group', 'Advanced settings')))
    panel.loss_advanced_groupbox.setTitle(str(t.get('advanced_group', 'Advanced settings')))
    panel.rare_patch_groupbox.setTitle(str(t.get('rare_patch_group', 'Rare patch oversampling')))
    panel.expert_groupbox.setTitle(str(t.get('expert_group', 'Expert settings')))
    panel.model_variants_groupbox.setTitle(str(t.get('model_variants_group', 'Additional model families')))
    panel.recognition_groupbox.setTitle(str(t.get('recognition_group', 'Recognition')))
    panel.runtime_groupbox.setTitle(str(t.get('runtime_group', 'Runtime and filtering')))
    panel.warmup_groupbox.setTitle(str(t.get('warmup_group', 'Warmup')))
    panel.scheduler_groupbox.setTitle(str(t.get('scheduler_group', 'Scheduler')))
    panel.hard_mining_groupbox.setTitle(str(t.get('hard_mining_group', 'Hard mining')))
    panel.early_stopping_groupbox.setTitle(str(t.get('early_stopping_group', 'Early stopping')))
    if hasattr(panel, 'settings_tabs'):
        training_index = panel._page_indexes.get('training')
        recognition_index = panel._page_indexes.get('recognition')
        if training_index is not None:
            panel.settings_tabs.setTabText(training_index, str(t.get('tab_training', 'Обучение')))
        if recognition_index is not None:
            panel.settings_tabs.setTabText(
                recognition_index,
                str(t.get('tab_recognition', 'Распознавание')),
            )
    panel.optimizer_type.setToolTip(str(t.get('optimizer_tip', '')))
    panel.learning_rate_spinbox.setToolTip(str(t.get('lr_tip', '')))
    panel.weight_decay_spinbox.setToolTip(str(t.get('wd_tip', '')))
    panel.train_batch_spinbox.setToolTip(str(t.get('train_batch_tip', t.get('batch_tip', ''))))
    panel.dataloader_num_workers_spinbox.setToolTip(
        str(
            t.get(
                'dataloader_num_workers_tip',
                'Number of worker processes used by the training DataLoader. Use -1 for automatic selection.',
            )
        )
    )
    panel.dataloader_num_workers_spinbox.setSpecialValueText(str(t.get('auto_value', 'auto')))
    panel.scheduler_type_combo.setToolTip(
        str(
            t.get(
                'scheduler_tip',
                'Learning-rate schedule applied after optimizer updates.',
            )
        )
    )
    panel.scheduler_type_combo.setItemText(0, str(t.get('scheduler_off', 'Off')))
    panel.scheduler_type_combo.setItemText(1, str(t.get('scheduler_reduce_on_plateau', 'ReduceLROnPlateau')))
    panel.scheduler_type_combo.setItemText(2, str(t.get('scheduler_cosine_annealing', 'CosineAnnealingLR')))
    panel.scheduler_type_combo.setItemText(3, str(t.get('scheduler_one_cycle', 'OneCycleLR')))
    panel.scheduler_type_combo.setItemText(4, str(t.get('scheduler_step_lr', 'StepLR')))
    panel.scheduler_plateau_factor_spinbox.setToolTip(str(t.get('scheduler_plateau_factor_tip', '')))
    panel.scheduler_plateau_patience_spinbox.setToolTip(str(t.get('scheduler_plateau_patience_tip', '')))
    panel.scheduler_plateau_threshold_spinbox.setToolTip(str(t.get('scheduler_plateau_threshold_tip', '')))
    panel.scheduler_plateau_min_lr_spinbox.setToolTip(str(t.get('scheduler_plateau_min_lr_tip', '')))
    panel.scheduler_plateau_cooldown_spinbox.setToolTip(str(t.get('scheduler_plateau_cooldown_tip', '')))
    panel.scheduler_cosine_t_max_spinbox.setToolTip(str(t.get('scheduler_cosine_t_max_tip', '')))
    panel.scheduler_cosine_eta_min_spinbox.setToolTip(str(t.get('scheduler_cosine_eta_min_tip', '')))
    panel.scheduler_one_cycle_max_lr_spinbox.setToolTip(str(t.get('scheduler_one_cycle_max_lr_tip', '')))
    panel.scheduler_one_cycle_pct_start_spinbox.setToolTip(str(t.get('scheduler_one_cycle_pct_start_tip', '')))
    panel.scheduler_one_cycle_anneal_strategy_combo.setToolTip(
        str(t.get('scheduler_one_cycle_anneal_strategy_tip', ''))
    )
    panel.scheduler_one_cycle_anneal_strategy_combo.setItemText(
        0,
        str(t.get('scheduler_one_cycle_anneal_cos', 'Cosine')),
    )
    panel.scheduler_one_cycle_anneal_strategy_combo.setItemText(
        1,
        str(t.get('scheduler_one_cycle_anneal_linear', 'Linear')),
    )
    panel.scheduler_one_cycle_div_factor_spinbox.setToolTip(str(t.get('scheduler_one_cycle_div_factor_tip', '')))
    panel.scheduler_one_cycle_final_div_factor_spinbox.setToolTip(
        str(t.get('scheduler_one_cycle_final_div_factor_tip', ''))
    )
    panel.scheduler_one_cycle_three_phase_check_box.setToolTip(
        str(t.get('scheduler_one_cycle_three_phase_tip', ''))
    )
    panel.scheduler_step_lr_step_size_spinbox.setToolTip(str(t.get('scheduler_step_lr_step_size_tip', '')))
    panel.scheduler_step_lr_gamma_spinbox.setToolTip(str(t.get('scheduler_step_lr_gamma_tip', '')))
    panel.recognition_batch_spinbox.setToolTip(str(t.get('recognition_batch_tip', t.get('batch_tip', ''))))
    panel.overlap_spinbox.setToolTip(str(t.get('overlap_tip', '')))
    panel.recognition_jpeg_quality_spinbox.setToolTip(
        str(t.get('recognition_jpeg_quality_tip', 'JPEG quality for recognition output (1..100).'))
    )
    panel.recognition_multiprocessing_check_box.setText(
        str(t.get('recognition_multiprocessing_enable', 'Use multiprocessing recognition'))
    )
    panel.recognition_multiprocessing_check_box.setToolTip(
        str(
            t.get(
                'recognition_multiprocessing_tip',
                'Speeds up recognition via separate cut/predict/sew processes and automatically falls back to single-thread mode when unsuitable.',
            )
        )
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
    panel.recognition_tta_check_box.setText(
        str(t.get('recognition_tta', 'Use TTA for recognition'))
    )
    panel.recognition_tta_check_box.setToolTip(
        str(
            t.get(
                'recognition_tta_tip',
                'Apply test-time augmentation to the main recognition mask during inference.',
            )
        )
    )
    panel.confidence_save_mode_combo.setToolTip(
        str(
            t.get(
                'confidence_save_mode_tip',
                'Choose whether to skip confidence output or save it from the model output or TTA result.',
            )
        )
    )
    panel.confidence_save_mode_combo.setItemText(
        0,
        str(t.get('confidence_save_mode_off', 'Do not save')),
    )
    panel.confidence_save_mode_combo.setItemText(
        1,
        str(t.get('confidence_save_mode_model_output', 'From model output')),
    )
    panel.confidence_save_mode_combo.setItemText(
        2,
        str(t.get('confidence_save_mode_tta', 'From TTA')),
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
    panel.deep_supervision_check_box.setText(str(t.get('deep_supervision', 'Enable deep supervision')))
    panel.deep_supervision_check_box.setToolTip(
        str(
            t.get(
                'deep_supervision_tip',
                'Train convolutional segmentation models with auxiliary decoder heads.',
            )
        )
    )
    loss_tip = str(t.get('loss_function_tip', ''))
    loss_group_title = str(labels_map.get('loss_function', t.get('loss_function', 'Loss terms')))
    panel.loss_terms_groupbox.setTitle(loss_group_title)
    panel.loss_terms_groupbox.setToolTip(loss_tip)
    panel.loss_terms_widget.setToolTip(loss_tip)
    panel.loss_formula_label.setToolTip(loss_tip)
    if 'conductors' in getattr(panel, 'loss_preset_buttons', {}):
        panel.loss_preset_buttons['conductors'].setText(str(t.get('loss_preset_conductors', 'Conductors')))
        panel.loss_preset_buttons['conductors'].setToolTip(
            str(
                t.get(
                    'loss_preset_conductors_tip',
                    'Preset: 0.5 BCE + 0.5 Dice.',
                )
            )
        )
    if 'contacts' in getattr(panel, 'loss_preset_buttons', {}):
        panel.loss_preset_buttons['contacts'].setText(str(t.get('loss_preset_contacts', 'Contacts')))
        panel.loss_preset_buttons['contacts'].setToolTip(
            str(
                t.get(
                    'loss_preset_contacts_tip',
                    'Preset: 0.5 BCE + 0.5 Focal Tversky.',
                )
            )
        )
    if hasattr(panel, 'loss_presets_widget'):
        panel.loss_presets_widget.setToolTip(loss_tip)
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
    panel._sync_validation_path_labels()


