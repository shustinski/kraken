"""state_store: build settings state."""
from __future__ import annotations
import json
from neuralimage.application.dto import SettingsState
from neuralimage.lib.data_interfaces import (
    normalize_confidence_save_mode,
    normalize_multi_gpu_mode,
    normalize_patch_batch_sync_mode,
    normalize_scheduler_name,
    normalize_validation_source,
)
from neuralimage.lib.loss_config import (
    deserialize_loss_term_weights,
    dominant_loss_function,
    normalize_loss_term_name,
    resolve_loss_term_weights,
)

def _build_settings_state(
    *,
    read_bool,
    read_int,
    read_float,
    read_str,
) -> SettingsState:
    from neuralimage.infrastructure.config.state_store_parts.q_settings_state_store import _coerce_json_object
    defaults = SettingsState()
    legacy_additional_processing = read_bool('additional_processing', False)
    legacy_use_multi_gpu = read_bool('use_multi_gpu', defaults.use_multi_gpu)
    multi_gpu_mode = normalize_multi_gpu_mode(
        read_str('multi_gpu_mode', defaults.multi_gpu_mode),
        use_multi_gpu_fallback=legacy_use_multi_gpu,
    )
    legacy_sample_x = read_int('sample_x_size', defaults.sample_size[0])
    legacy_sample_y = read_int('sample_y_size', defaults.sample_size[1])
    legacy_batch_size = read_int('batch_size', defaults.batch_size)
    legacy_shuffle = read_bool('shuffle', defaults.shuffle)
    patch_batch_sync_mode = normalize_patch_batch_sync_mode(
        read_str('patch_batch_sync_mode', defaults.patch_batch_sync_mode)
    )
    sync_patch_sizes = bool(
        read_bool('sync_patch_sizes', patch_batch_sync_mode in ('patch', 'patch_and_batch'))
    )
    rare_patch_oversampling_factor = max(
        2,
        read_int(
            'rare_patch_oversampling_factor',
            getattr(defaults, 'rare_patch_oversampling_factor', 2),
        ),
    )
    legacy_loss_function = read_str('loss_function', defaults.loss_function)
    dice_loss_weight = read_float('dice_loss_weight', defaults.dice_loss_weight)
    iou_loss_weight = read_float('iou_loss_weight', defaults.iou_loss_weight)
    loss_term_weights_raw = read_str('loss_term_weights_json', '')
    loss_term_weights = resolve_loss_term_weights(
        deserialize_loss_term_weights(loss_term_weights_raw),
        fallback_loss_function=legacy_loss_function,
        dice_weight=dice_loss_weight,
        iou_weight=iou_loss_weight,
    )
    try:
        sem_segmentation_config = json.loads(read_str('sem_segmentation_config_json', '{}'))
    except (TypeError, json.JSONDecodeError):
        sem_segmentation_config = {}
    if not isinstance(sem_segmentation_config, dict):
        sem_segmentation_config = {}
    return SettingsState(
        step=read_int('cut_step', defaults.step),
        vertical_rotation=read_bool('vertical_rotation', defaults.vertical_rotation),
        horizontal_rotation=read_bool('horizontal_rotation', defaults.horizontal_rotation),
        flip_x=read_bool('flip_x', getattr(defaults, 'flip_x', False)),
        flip_y=read_bool('flip_y', getattr(defaults, 'flip_y', False)),
        additional_augmentation=read_bool('additional_augmentation', defaults.additional_augmentation),
        augmentation_multiplier=read_float('augmentation_multiplier', getattr(defaults, 'augmentation_multiplier', 0.0)),
        augmentation_brightness_strength=read_float(
            'augmentation_brightness_strength', defaults.augmentation_brightness_strength
        ),
        augmentation_contrast_strength=read_float(
            'augmentation_contrast_strength', defaults.augmentation_contrast_strength
        ),
        augmentation_gamma_strength=read_float(
            'augmentation_gamma_strength',
            getattr(defaults, 'augmentation_gamma_strength', 0.15),
        ),
        augmentation_noise_probability=read_float(
            'augmentation_noise_probability', defaults.augmentation_noise_probability
        ),
        augmentation_noise_sigma=read_float('augmentation_noise_sigma', defaults.augmentation_noise_sigma),
        augmentation_blur_probability=read_float(
            'augmentation_blur_probability',
            getattr(defaults, 'augmentation_blur_probability', 0.25),
        ),
        augmentation_blur_radius=read_float(
            'augmentation_blur_radius',
            getattr(defaults, 'augmentation_blur_radius', 1.0),
        ),
        training_augmentation=_coerce_json_object(
            read_str('training_augmentation_json', ''),
            default=getattr(defaults, 'training_augmentation', {}),
        ),
        sample_size=(
            legacy_sample_x,
            legacy_sample_y,
        ),
        train_patch_size=(
            read_int('train_patch_x_size', legacy_sample_x),
            read_int('train_patch_y_size', legacy_sample_y),
        ),
        recognition_patch_size=(
            read_int('recognition_patch_x_size', legacy_sample_x),
            read_int('recognition_patch_y_size', legacy_sample_y),
        ),
        model=read_str('model', defaults.model),
        color_mode=read_str('color_mode', defaults.color_mode),
        shuffle=legacy_shuffle,
        shuffle_patches_in_frame=read_bool(
            'shuffle_patches_in_frame',
            legacy_shuffle,
        ),
        random_crop=read_bool('random_crop', defaults.random_crop),
        crops_per_image=read_int('crops_per_image', defaults.crops_per_image),
        scale_augmentation=read_bool('scale_augmentation', defaults.scale_augmentation),
        scale_augmentation_strength=read_float(
            'scale_augmentation_strength',
            defaults.scale_augmentation_strength,
        ),
        use_validation=read_bool('validation', defaults.use_validation),
        validation_percent=read_int('validation_percent', defaults.validation_percent),
        validation_source=normalize_validation_source(
            read_str('validation_source', getattr(defaults, 'validation_source', 'split'))
        ),
        validation_image_folder=read_str(
            'validation_image_folder',
            getattr(defaults, 'validation_image_folder', ''),
        ),
        validation_label_folder=read_str(
            'validation_label_folder',
            getattr(defaults, 'validation_label_folder', ''),
        ),
        save_validation_binary_images=read_bool(
            'save_validation_binary_images',
            getattr(defaults, 'save_validation_binary_images', False),
        ),
        sample_cut_mode=read_str('sample_cut_mode', defaults.sample_cut_mode),
        batch_size=legacy_batch_size,
        dataloader_num_workers=read_int(
            'dataloader_num_workers',
            getattr(defaults, 'dataloader_num_workers', -1),
        ),
        train_batch_size=read_int('train_batch_size', legacy_batch_size),
        recognition_batch_size=read_int('recognition_batch_size', legacy_batch_size),
        sync_patch_sizes=sync_patch_sizes,
        patch_batch_sync_mode=patch_batch_sync_mode,
        overlap=read_int('overlap', defaults.overlap),
        recognition_jpeg_quality=read_int('recognition_jpeg_quality', defaults.recognition_jpeg_quality),
        recognition_multiprocessing_enabled=read_bool(
            'recognition_multiprocessing_enabled',
            getattr(defaults, 'recognition_multiprocessing_enabled', True),
        ),
        recognition_binarize_output=read_bool(
            'recognition_binarize_output',
            getattr(defaults, 'recognition_binarize_output', True),
        ),
        recognition_use_auto_threshold=read_bool(
            'recognition_use_auto_threshold',
            getattr(defaults, 'recognition_use_auto_threshold', True),
        ),
        recognition_threshold=read_float(
            'recognition_threshold',
            getattr(defaults, 'recognition_threshold', 0.5),
        ),
        recognition_tta_enabled=read_bool(
            'recognition_tta_enabled',
            getattr(defaults, 'recognition_tta_enabled', False),
        ),
        confidence_tta_enabled=read_bool(
            'confidence_tta_enabled',
            getattr(defaults, 'confidence_tta_enabled', False),
        ),
        recognition_postprocess=read_bool(
            'recognition_postprocess',
            getattr(defaults, 'recognition_postprocess', False),
        ),
        recognition_postprocess_kernel_size=max(
            1,
            read_int(
                'recognition_postprocess_kernel_size',
                getattr(defaults, 'recognition_postprocess_kernel_size', 3),
            ),
        ),
        confidence_save_mode=normalize_confidence_save_mode(
            read_str(
                'confidence_save_mode',
                getattr(defaults, 'confidence_save_mode', 'off'),
            )
        ),
        log_update_frequency=read_int('log_update_frequency', defaults.log_update_frequency),
        crop_enabled=read_bool('crop_enabled', legacy_additional_processing),
        resize_enabled=read_bool('resize_enabled', legacy_additional_processing),
        edge_cut_size=read_int('edge_cut_size', defaults.edge_cut_size),
        target_size=(
            read_int('target_x_size', defaults.target_size[0]),
            read_int('target_y_size', defaults.target_size[1]),
        ),
        compression_factor=max(1, read_int('compression_factor', getattr(defaults, 'compression_factor', 1))),
        recursive_file_search=read_bool(
            'recursive_file_search',
            getattr(defaults, 'recursive_file_search', False),
        ),
        optimizer_name=read_str('optimizer_name', defaults.optimizer_name),
        mixed_precision=read_str('mixed_precision', defaults.mixed_precision),
        loss_function=normalize_loss_term_name(legacy_loss_function)
        or dominant_loss_function(loss_term_weights, fallback=defaults.loss_function),
        loss_term_weights=loss_term_weights,
        dice_loss_weight=dice_loss_weight,
        iou_loss_weight=iou_loss_weight,
        topograph_enabled=read_bool('topograph_enabled', getattr(defaults, 'topograph_enabled', False)),
        topograph_loss_weight=read_float(
            'topograph_loss_weight',
            getattr(defaults, 'topograph_loss_weight', 0.1),
        ),
        topograph_debug_viz=read_bool(
            'topograph_debug_viz',
            getattr(defaults, 'topograph_debug_viz', False),
        ),
        topograph_num_processes=read_int(
            'topograph_num_processes',
            getattr(defaults, 'topograph_num_processes', 1),
        ),
        topograph_use_c=read_bool('topograph_use_c', getattr(defaults, 'topograph_use_c', False)),
        learning_rate=read_float('learning_rate', defaults.learning_rate),
        weight_decay=read_float('weight_decay', defaults.weight_decay),
        early_stopping_enabled=read_bool('early_stopping_enabled', defaults.early_stopping_enabled),
        early_stopping_patience=read_int(
            'early_stopping_patience', getattr(defaults, 'early_stopping_patience', 10)
        ),
        early_stopping_min_delta=read_float(
            'early_stopping_min_delta', getattr(defaults, 'early_stopping_min_delta', 0.0)
        ),
        early_stopping_restore_best_weights=read_bool(
            'early_stopping_restore_best_weights',
            getattr(defaults, 'early_stopping_restore_best_weights', True),
        ),
        warmup_enabled=read_bool('warmup_enabled', defaults.warmup_enabled),
        deep_supervision=read_bool(
            'deep_supervision',
            getattr(defaults, 'deep_supervision', False),
        ),
        warmup_epochs=read_int('warmup_epochs', defaults.warmup_epochs),
        warmup_start_factor=read_float('warmup_start_factor', defaults.warmup_start_factor),
        scheduler_name=normalize_scheduler_name(
            read_str('scheduler_name', getattr(defaults, 'scheduler_name', 'off'))
        ),
        scheduler_plateau_factor=read_float(
            'scheduler_plateau_factor',
            getattr(defaults, 'scheduler_plateau_factor', 0.5),
        ),
        scheduler_plateau_patience=read_int(
            'scheduler_plateau_patience',
            getattr(defaults, 'scheduler_plateau_patience', 3),
        ),
        scheduler_plateau_threshold=read_float(
            'scheduler_plateau_threshold',
            getattr(defaults, 'scheduler_plateau_threshold', 1e-4),
        ),
        scheduler_plateau_min_lr=read_float(
            'scheduler_plateau_min_lr',
            getattr(defaults, 'scheduler_plateau_min_lr', 1e-6),
        ),
        scheduler_plateau_cooldown=read_int(
            'scheduler_plateau_cooldown',
            getattr(defaults, 'scheduler_plateau_cooldown', 0),
        ),
        scheduler_cosine_t_max=read_int(
            'scheduler_cosine_t_max',
            getattr(defaults, 'scheduler_cosine_t_max', 10),
        ),
        scheduler_cosine_eta_min=read_float(
            'scheduler_cosine_eta_min',
            getattr(defaults, 'scheduler_cosine_eta_min', 1e-6),
        ),
        scheduler_one_cycle_max_lr=read_float(
            'scheduler_one_cycle_max_lr',
            getattr(defaults, 'scheduler_one_cycle_max_lr', 1e-3),
        ),
        scheduler_one_cycle_pct_start=read_float(
            'scheduler_one_cycle_pct_start',
            getattr(defaults, 'scheduler_one_cycle_pct_start', 0.3),
        ),
        scheduler_one_cycle_anneal_strategy=read_str(
            'scheduler_one_cycle_anneal_strategy',
            getattr(defaults, 'scheduler_one_cycle_anneal_strategy', 'cos'),
        ),
        scheduler_one_cycle_div_factor=read_float(
            'scheduler_one_cycle_div_factor',
            getattr(defaults, 'scheduler_one_cycle_div_factor', 25.0),
        ),
        scheduler_one_cycle_final_div_factor=read_float(
            'scheduler_one_cycle_final_div_factor',
            getattr(defaults, 'scheduler_one_cycle_final_div_factor', 10000.0),
        ),
        scheduler_one_cycle_three_phase=read_bool(
            'scheduler_one_cycle_three_phase',
            getattr(defaults, 'scheduler_one_cycle_three_phase', False),
        ),
        scheduler_step_lr_step_size=read_int(
            'scheduler_step_lr_step_size',
            getattr(defaults, 'scheduler_step_lr_step_size', 10),
        ),
        scheduler_step_lr_gamma=read_float(
            'scheduler_step_lr_gamma',
            getattr(defaults, 'scheduler_step_lr_gamma', 0.1),
        ),
        hard_mining_enabled=read_bool('hard_mining_enabled', defaults.hard_mining_enabled),
        hard_mining_strength=read_float(
            'hard_mining_strength', getattr(defaults, 'hard_mining_strength', 2.0)
        ),
        hard_mining_ema_alpha=read_float(
            'hard_mining_ema_alpha', getattr(defaults, 'hard_mining_ema_alpha', 0.3)
        ),
        sem_segmentation_config=sem_segmentation_config,
        random_patch_size_enabled=read_bool(
            'random_patch_size_enabled', getattr(defaults, 'random_patch_size_enabled', False)
        ),
        random_patch_min_size=(
            read_int('random_patch_min_width', getattr(defaults, 'random_patch_min_size', (128, 128))[0]),
            read_int('random_patch_min_height', getattr(defaults, 'random_patch_min_size', (128, 128))[1]),
        ),
        random_patch_max_size=(
            read_int('random_patch_max_width', getattr(defaults, 'random_patch_max_size', (512, 512))[0]),
            read_int('random_patch_max_height', getattr(defaults, 'random_patch_max_size', (512, 512))[1]),
        ),
        hard_pixel_mining_enabled=read_bool(
            'hard_pixel_mining_enabled',
            getattr(defaults, 'hard_pixel_mining_enabled', False),
        ),
        hard_pixel_mining_ratio=read_float(
            'hard_pixel_mining_ratio',
            getattr(defaults, 'hard_pixel_mining_ratio', 0.25),
        ),
        cutout_enabled=read_bool(
            'cutout_enabled',
            getattr(defaults, 'cutout_enabled', False),
        ),
        cutout_probability=read_float(
            'cutout_probability',
            getattr(defaults, 'cutout_probability', 1.0),
        ),
        cutout_holes=max(
            1,
            read_int(
                'cutout_holes',
                getattr(defaults, 'cutout_holes', 1),
            ),
        ),
        cutout_size_ratio=read_float(
            'cutout_size_ratio',
            getattr(defaults, 'cutout_size_ratio', 0.25),
        ),
        random_artifacts_enabled=read_bool(
            'random_artifacts_enabled',
            getattr(defaults, 'random_artifacts_enabled', False),
        ),
        random_artifacts_probability=read_float(
            'random_artifacts_probability',
            getattr(defaults, 'random_artifacts_probability', 1.0),
        ),
        random_artifacts_count=max(
            1,
            read_int(
                'random_artifacts_count',
                getattr(defaults, 'random_artifacts_count', 1),
            ),
        ),
        random_artifacts_size_ratio=read_float(
            'random_artifacts_size_ratio',
            getattr(defaults, 'random_artifacts_size_ratio', 0.25),
        ),
        random_artifacts_dust_enabled=read_bool(
            'random_artifacts_dust_enabled',
            getattr(defaults, 'random_artifacts_dust_enabled', True),
        ),
        random_artifacts_resist_residue_enabled=read_bool(
            'random_artifacts_resist_residue_enabled',
            getattr(defaults, 'random_artifacts_resist_residue_enabled', True),
        ),
        random_artifacts_etch_residue_enabled=read_bool(
            'random_artifacts_etch_residue_enabled',
            getattr(defaults, 'random_artifacts_etch_residue_enabled', True),
        ),
        random_artifacts_particle_cluster_enabled=read_bool(
            'random_artifacts_particle_cluster_enabled',
            getattr(defaults, 'random_artifacts_particle_cluster_enabled', True),
        ),
        random_artifacts_flake_enabled=read_bool(
            'random_artifacts_flake_enabled',
            getattr(defaults, 'random_artifacts_flake_enabled', True),
        ),
        mixup_enabled=read_bool(
            'mixup_enabled',
            getattr(defaults, 'mixup_enabled', False),
        ),
        mixup_probability=read_float(
            'mixup_probability',
            getattr(defaults, 'mixup_probability', 1.0),
        ),
        mixup_alpha=read_float(
            'mixup_alpha',
            getattr(defaults, 'mixup_alpha', 0.2),
        ),
        skip_uniform_labels=read_bool('skip_uniform_labels', defaults.skip_uniform_labels),
        rare_patch_oversampling_enabled=read_bool(
            'rare_patch_oversampling_enabled',
            getattr(defaults, 'rare_patch_oversampling_enabled', False),
        ),
        rare_patch_oversampling_factor=rare_patch_oversampling_factor,
        use_multi_gpu=multi_gpu_mode != 'off',
        multi_gpu_mode=multi_gpu_mode,
        torch_compile_enabled=read_bool('torch_compile_enabled', defaults.torch_compile_enabled),
        show_batch_preview=read_bool('show_batch_preview', defaults.show_batch_preview),
        synthetic_defect_generator=_coerce_json_object(
            read_str(
                'synthetic_defect_generator_json',
                read_str('pcb_defects_json', ''),
            ),
            default=getattr(defaults, 'synthetic_defect_generator', {}),
        ),
        tech_aug=_coerce_json_object(
            read_str('tech_aug_json', ''),
            default=getattr(defaults, 'tech_aug', {}),
        ),
        pcb_defects=_coerce_json_object(
            read_str('pcb_defects_json', ''),
            default=getattr(defaults, 'pcb_defects', {}),
        ),
    )
