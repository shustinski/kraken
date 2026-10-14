"""apply_settings_panel_texts section 1."""
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

def _apply_settings_panel_texts_section_1(panel):
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
    recursive_file_search_key = 'recursive_file_search'
    compression_factor_key = 'compression_factor'
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
    labels_map.setdefault(
        recursive_file_search_key,
        str(t.get('recursive_file_search', 'Search in subfolders')),
    )
    descriptions.setdefault(
        recursive_file_search_key,
        str(t.get('recursive_file_search_tip', 'Find source, training, and label files recursively under selected folders.')),
    )
    labels_map.setdefault(
        compression_factor_key,
        str(t.get('compression_factor', 'Compression factor')),
    )
    descriptions.setdefault(
        compression_factor_key,
        str(t.get('compression_factor_tip', 'Downscale each frame by this factor and restore it to the original size before training or recognition.')),
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
    return augmentation_blur_probability_key, augmentation_blur_radius_key, augmentation_gamma_strength_key, crops_per_image_key, cutout_holes_key, cutout_probability_key, cutout_size_ratio_key, descriptions, labels_map, mixup_alpha_key, mixup_probability_key, pcb_break_weight_key, pcb_defects_max_count_key, pcb_defects_min_count_key, pcb_defects_probability_key, pcb_excess_copper_weight_key, pcb_misalignment_weight_key, pcb_missing_copper_weight_key, pcb_pinhole_weight_key, pcb_short_weight_key, pcb_spurious_copper_weight_key, pcb_via_weight_key, random_artifacts_count_key, random_artifacts_probability_key, random_artifacts_size_ratio_key, rare_patch_oversampling_factor_key, recognition_postprocess_kernel_size_key, recognition_threshold_key, scale_augmentation_strength_key, scheduler_cosine_eta_min_key, scheduler_cosine_t_max_key, scheduler_one_cycle_anneal_strategy_key, scheduler_one_cycle_div_factor_key, scheduler_one_cycle_final_div_factor_key, scheduler_one_cycle_max_lr_key, scheduler_one_cycle_pct_start_key, scheduler_one_cycle_three_phase_key, scheduler_plateau_cooldown_key, scheduler_plateau_min_lr_key, scheduler_step_lr_gamma_key, scheduler_step_lr_step_size_key, t, tech_aug_blur_threshold_probability_key, tech_aug_boundary_aware_probability_key, tech_aug_gap_variation_probability_key, tech_aug_global_width_probability_key, tech_aug_local_morphology_probability_key, tech_aug_max_changed_pixels_ratio_key, tech_aug_max_foreground_ratio_delta_key, tech_aug_max_operations_key, tech_aug_min_operations_key, tech_aug_scale_rethreshold_probability_key, validation_image_path_key, validation_label_path_key, validation_source_key
