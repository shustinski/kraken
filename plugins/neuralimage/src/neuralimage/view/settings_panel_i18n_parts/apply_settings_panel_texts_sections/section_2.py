"""apply_settings_panel_texts section 2."""
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

def _apply_settings_panel_texts_section_2(augmentation_blur_probability_key, augmentation_blur_radius_key, augmentation_gamma_strength_key, crops_per_image_key, cutout_holes_key, cutout_probability_key, cutout_size_ratio_key, descriptions, labels_map, mixup_alpha_key, mixup_probability_key, pcb_defects_max_count_key, pcb_defects_min_count_key, pcb_defects_probability_key, random_artifacts_count_key, random_artifacts_probability_key, random_artifacts_size_ratio_key, scale_augmentation_strength_key, scheduler_cosine_eta_min_key, scheduler_cosine_t_max_key, scheduler_one_cycle_anneal_strategy_key, scheduler_one_cycle_div_factor_key, scheduler_one_cycle_final_div_factor_key, scheduler_one_cycle_max_lr_key, scheduler_one_cycle_pct_start_key, scheduler_one_cycle_three_phase_key, scheduler_plateau_cooldown_key, scheduler_plateau_min_lr_key, scheduler_step_lr_gamma_key, scheduler_step_lr_step_size_key, t, tech_aug_blur_threshold_probability_key, tech_aug_boundary_aware_probability_key, tech_aug_gap_variation_probability_key, tech_aug_global_width_probability_key, tech_aug_local_morphology_probability_key, tech_aug_max_changed_pixels_ratio_key, tech_aug_max_foreground_ratio_delta_key, tech_aug_max_operations_key, tech_aug_min_operations_key, tech_aug_scale_rethreshold_probability_key):
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
