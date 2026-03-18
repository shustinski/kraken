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
    pcb_break_weight_key = 'pcb_break_weight'
    pcb_short_weight_key = 'pcb_short_weight'
    pcb_missing_copper_weight_key = 'pcb_missing_copper_weight'
    pcb_excess_copper_weight_key = 'pcb_excess_copper_weight'
    pcb_pinhole_weight_key = 'pcb_pinhole_weight'
    pcb_spurious_copper_weight_key = 'pcb_spurious_copper_weight'
    pcb_via_weight_key = 'pcb_via_weight'
    pcb_misalignment_weight_key = 'pcb_misalignment_weight'
    rare_patch_oversampling_factor_key = 'rare_patch_oversampling_factor'
    recognition_threshold_key = 'recognition_threshold'
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
    if jpeg_quality_key not in labels_map:
        labels_map[jpeg_quality_key] = 'JPEG quality'
    if jpeg_quality_key not in descriptions:
        descriptions[jpeg_quality_key] = str(
            t.get('recognition_jpeg_quality_tip', 'JPEG quality for recognition output (1..100).')
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
        labels_map[pcb_break_weight_key] = 'Break weight'
    if pcb_break_weight_key not in descriptions:
        descriptions[pcb_break_weight_key] = str(
            t.get('pcb_break_weight_tip', 'Relative sampling weight for local conductor breaks.')
        )
    if pcb_short_weight_key not in labels_map:
        labels_map[pcb_short_weight_key] = 'Short weight'
    if pcb_short_weight_key not in descriptions:
        descriptions[pcb_short_weight_key] = str(
            t.get('pcb_short_weight_tip', 'Relative sampling weight for parasitic bridges between nearby traces.')
        )
    if pcb_missing_copper_weight_key not in labels_map:
        labels_map[pcb_missing_copper_weight_key] = 'Missing copper weight'
    if pcb_missing_copper_weight_key not in descriptions:
        descriptions[pcb_missing_copper_weight_key] = str(
            t.get('pcb_missing_copper_weight_tip', 'Relative sampling weight for local copper loss defects.')
        )
    if pcb_excess_copper_weight_key not in labels_map:
        labels_map[pcb_excess_copper_weight_key] = 'Excess copper weight'
    if pcb_excess_copper_weight_key not in descriptions:
        descriptions[pcb_excess_copper_weight_key] = str(
            t.get('pcb_excess_copper_weight_tip', 'Relative sampling weight for copper burrs and growths.')
        )
    if pcb_pinhole_weight_key not in labels_map:
        labels_map[pcb_pinhole_weight_key] = 'Pinhole weight'
    if pcb_pinhole_weight_key not in descriptions:
        descriptions[pcb_pinhole_weight_key] = str(
            t.get('pcb_pinhole_weight_tip', 'Relative sampling weight for small holes inside copper regions.')
        )
    if pcb_spurious_copper_weight_key not in labels_map:
        labels_map[pcb_spurious_copper_weight_key] = 'Spurious copper weight'
    if pcb_spurious_copper_weight_key not in descriptions:
        descriptions[pcb_spurious_copper_weight_key] = str(
            t.get('pcb_spurious_copper_weight_tip', 'Relative sampling weight for isolated parasitic copper islands.')
        )
    if pcb_via_weight_key not in labels_map:
        labels_map[pcb_via_weight_key] = 'Via defect weight'
    if pcb_via_weight_key not in descriptions:
        descriptions[pcb_via_weight_key] = str(
            t.get('pcb_via_weight_tip', 'Relative sampling weight for via shift, size, or partial-loss defects.')
        )
    if pcb_misalignment_weight_key not in labels_map:
        labels_map[pcb_misalignment_weight_key] = 'Misalignment weight'
    if pcb_misalignment_weight_key not in descriptions:
        descriptions[pcb_misalignment_weight_key] = str(
            t.get('pcb_misalignment_weight_tip', 'Relative sampling weight for local layer misregistration effects.')
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
    panel.tech_augmentation_check_box.setText(
        str(
            t.get(
                'tech_augmentation',
                'Technology variation augmentation',
            )
        )
    )
    panel.tech_augmentation_check_box.setToolTip(
        str(
            t.get(
                'tech_augmentation_tip',
                'Apply geometry-only process variations to binary metallization masks during training.',
            )
        )
    )
    panel.tech_augmentation_debug_pair_check_box.setText(
        str(
            t.get(
                'tech_augmentation_debug_pair',
                'Debug: return original and augmented mask',
            )
        )
    )
    panel.tech_augmentation_debug_pair_check_box.setToolTip(
        str(
            t.get(
                'tech_augmentation_debug_pair_tip',
                'Keep both the original and augmented mask for visual validation of synthetic process drift.',
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
    panel.pcb_defects_check_box.setText(
        _read_text_from_mappings(
            (t, labels_map),
            ('pcb_defects_enabled', 'pcb_defects_enable'),
            'Enable synthetic PCB defects',
        )
    )
    panel.pcb_defects_check_box.setToolTip(
        _read_text_from_mappings(
            (t, descriptions),
            ('pcb_defects_tip', 'pcb_defects_enabled'),
            'Inject realistic topology defects into training samples only.',
        )
    )
    panel.pcb_defects_use_input_mask_check_box.setText(
        str(t.get('pcb_defects_use_input_mask', 'Use provided PCB mask for defect placement'))
    )
    panel.pcb_defects_use_input_mask_check_box.setToolTip(
        str(
            t.get(
                'pcb_defects_use_input_mask_tip',
                'Use the input binary mask to place defects on valid copper structures when available.',
            )
        )
    )
    panel.pcb_defects_use_defect_mask_as_label_check_box.setText(
        str(t.get('pcb_defects_use_defect_mask_as_label', 'Use defect mask as training label'))
    )
    panel.pcb_defects_use_defect_mask_as_label_check_box.setToolTip(
        str(
            t.get(
                'pcb_defects_use_defect_mask_as_label_tip',
                'Return the generated defect mask as the supervision target for synthetic-defect segmentation.',
            )
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
    panel.tech_aug_min_operations_spinbox.setToolTip(str(t.get('tech_aug_min_operations_tip', '')))
    panel.tech_aug_max_operations_spinbox.setToolTip(str(t.get('tech_aug_max_operations_tip', '')))
    panel.tech_aug_max_changed_pixels_ratio_spinbox.setToolTip(
        str(t.get('tech_aug_max_changed_pixels_ratio_tip', ''))
    )
    panel.tech_aug_max_foreground_ratio_delta_spinbox.setToolTip(
        str(t.get('tech_aug_max_foreground_ratio_delta_tip', ''))
    )
    panel.tech_aug_global_width_probability_spinbox.setToolTip(
        str(t.get('tech_aug_global_width_probability_tip', ''))
    )
    panel.tech_aug_scale_rethreshold_probability_spinbox.setToolTip(
        str(t.get('tech_aug_scale_rethreshold_probability_tip', ''))
    )
    panel.tech_aug_blur_threshold_probability_spinbox.setToolTip(
        str(t.get('tech_aug_blur_threshold_probability_tip', ''))
    )
    panel.tech_aug_boundary_aware_probability_spinbox.setToolTip(
        str(t.get('tech_aug_boundary_aware_probability_tip', ''))
    )
    panel.tech_aug_local_morphology_probability_spinbox.setToolTip(
        str(t.get('tech_aug_local_morphology_probability_tip', ''))
    )
    panel.tech_aug_gap_variation_probability_spinbox.setToolTip(
        str(t.get('tech_aug_gap_variation_probability_tip', ''))
    )
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
    panel.pcb_defects_groupbox.setToolTip(
        str(t.get('pcb_defects_group_tip', 'Synthetic topology defects applied only during training.'))
    )
    for defect_name, text_key in (
        ('break', 'pcb_break_weight_tip'),
        ('short', 'pcb_short_weight_tip'),
        ('missing_copper', 'pcb_missing_copper_weight_tip'),
        ('excess_copper', 'pcb_excess_copper_weight_tip'),
        ('pinhole', 'pcb_pinhole_weight_tip'),
        ('spurious_copper', 'pcb_spurious_copper_weight_tip'),
        ('via', 'pcb_via_weight_tip'),
        ('misalignment', 'pcb_misalignment_weight_tip'),
    ):
        panel.pcb_defect_type_spinboxes[defect_name].setToolTip(str(t.get(text_key, '')))
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
    panel.pcb_defects_groupbox.setTitle(str(t.get('pcb_defects_group', 'Synthetic PCB defects')))
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
    panel.recognition_groupbox.setTitle(str(t.get('recognition_group', 'Recognition')))
    panel.runtime_groupbox.setTitle(str(t.get('runtime_group', 'Runtime and filtering')))
    panel.warmup_groupbox.setTitle(str(t.get('warmup_group', 'Warmup')))
    panel.scheduler_groupbox.setTitle(str(t.get('scheduler_group', 'Scheduler')))
    panel.hard_mining_groupbox.setTitle(str(t.get('hard_mining_group', 'Hard mining')))
    panel.early_stopping_groupbox.setTitle(str(t.get('early_stopping_group', 'Early stopping')))
    if hasattr(panel, 'settings_tabs'):
        panel.settings_tabs.setTabText(panel._page_indexes.get('base', 0), str(t.get('tab_base', 'Базовые')))
        panel.settings_tabs.setTabText(panel._page_indexes.get('training', 1), str(t.get('tab_training', 'Обучение')))
        panel.settings_tabs.setTabText(
            panel._page_indexes.get('recognition', 2),
            str(t.get('tab_recognition', 'Распознавание')),
        )
        panel.settings_tabs.setTabText(panel._page_indexes.get('expert', 3), str(t.get('tab_expert', 'Эксперт')))
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
    panel._sync_validation_path_labels()


