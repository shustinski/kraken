"""state_store: settings state storage dict 2."""
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
    dominant_loss_function,
    normalize_loss_term_name,
    serialize_loss_term_weights,
)

def _settings_state_to_storage_dict(state: SettingsState) -> dict[str, str | int | float | bool]:
    from neuralimage.infrastructure.config.state_store_parts.q_settings_state_store import _jsonify_value
    multi_gpu_mode = normalize_multi_gpu_mode(
        getattr(state, 'multi_gpu_mode', ''),
        use_multi_gpu_fallback=bool(getattr(state, 'use_multi_gpu', False)),
    )
    train_patch_size = tuple(getattr(state, 'train_patch_size', None) or state.sample_size)
    recognition_patch_size = tuple(getattr(state, 'recognition_patch_size', None) or state.sample_size)
    train_batch_size = int(getattr(state, 'train_batch_size', None) or state.batch_size)
    recognition_batch_size = int(getattr(state, 'recognition_batch_size', None) or state.batch_size)
    patch_batch_sync_mode = normalize_patch_batch_sync_mode(getattr(state, 'patch_batch_sync_mode', ''))
    sync_patch_sizes = bool(
        getattr(state, 'sync_patch_sizes', patch_batch_sync_mode in ('patch', 'patch_and_batch'))
    )
    patch_batch_sync_mode = 'patch' if sync_patch_sizes else 'off'
    return {
        'cut_step': int(state.step),
        'horizontal_rotation': bool(state.horizontal_rotation),
        'vertical_rotation': bool(state.vertical_rotation),
        'flip_x': bool(getattr(state, 'flip_x', False)),
        'flip_y': bool(getattr(state, 'flip_y', False)),
        'additional_augmentation': bool(state.additional_augmentation),
        'augmentation_multiplier': float(getattr(state, 'augmentation_multiplier', 0.0)),
        'augmentation_brightness_strength': float(state.augmentation_brightness_strength),
        'augmentation_contrast_strength': float(state.augmentation_contrast_strength),
        'augmentation_gamma_strength': float(getattr(state, 'augmentation_gamma_strength', 0.15)),
        'augmentation_noise_probability': float(state.augmentation_noise_probability),
        'augmentation_noise_sigma': float(state.augmentation_noise_sigma),
        'augmentation_blur_probability': float(getattr(state, 'augmentation_blur_probability', 0.25)),
        'augmentation_blur_radius': float(getattr(state, 'augmentation_blur_radius', 1.0)),
        'training_augmentation_json': json.dumps(
            getattr(state, 'training_augmentation', {}), ensure_ascii=False, sort_keys=True
        ),
        'model': state.model,
        'color_mode': state.color_mode,
        'shuffle': bool(state.shuffle),
        'shuffle_patches_in_frame': bool(getattr(state, 'shuffle_patches_in_frame', state.shuffle)),
        'random_crop': bool(getattr(state, 'random_crop', False)),
        'crops_per_image': int(getattr(state, 'crops_per_image', 64)),
        'scale_augmentation': bool(getattr(state, 'scale_augmentation', False)),
        'scale_augmentation_strength': float(getattr(state, 'scale_augmentation_strength', 0.2)),
        'validation': bool(state.use_validation),
        'validation_percent': int(state.validation_percent),
        'validation_source': normalize_validation_source(
            getattr(state, 'validation_source', 'split')
        ),
        'validation_image_folder': str(getattr(state, 'validation_image_folder', '')),
        'validation_label_folder': str(getattr(state, 'validation_label_folder', '')),
        'save_validation_binary_images': bool(
            getattr(state, 'save_validation_binary_images', False)
        ),
        'sample_cut_mode': state.sample_cut_mode,
        'batch_size': int(train_batch_size),
        'dataloader_num_workers': int(getattr(state, 'dataloader_num_workers', -1)),
        'train_batch_size': int(train_batch_size),
        'recognition_batch_size': int(recognition_batch_size),
        'sync_patch_sizes': bool(sync_patch_sizes),
        'patch_batch_sync_mode': patch_batch_sync_mode,
        'overlap': int(state.overlap),
        'recognition_jpeg_quality': int(getattr(state, 'recognition_jpeg_quality', 95)),
        'recognition_multiprocessing_enabled': bool(
            getattr(state, 'recognition_multiprocessing_enabled', True)
        ),
        'recognition_binarize_output': bool(getattr(state, 'recognition_binarize_output', True)),
        'recognition_use_auto_threshold': bool(getattr(state, 'recognition_use_auto_threshold', True)),
        'recognition_threshold': float(getattr(state, 'recognition_threshold', 0.5)),
        'recognition_tta_enabled': bool(getattr(state, 'recognition_tta_enabled', False)),
        'confidence_tta_enabled': bool(getattr(state, 'confidence_tta_enabled', False)),
        'recognition_postprocess': bool(getattr(state, 'recognition_postprocess', False)),
        'recognition_postprocess_kernel_size': int(
            max(1, int(getattr(state, 'recognition_postprocess_kernel_size', 3)))
        ),
        'confidence_save_mode': normalize_confidence_save_mode(
            getattr(state, 'confidence_save_mode', 'off')
        ),
        'log_update_frequency': int(state.log_update_frequency),
        'crop_enabled': bool(state.crop_enabled),
        'resize_enabled': bool(state.resize_enabled),
        'additional_processing': bool(state.crop_enabled or state.resize_enabled),
        'edge_cut_size': int(state.edge_cut_size),
        'compression_factor': int(max(1, int(getattr(state, 'compression_factor', 1)))),
        'recursive_file_search': bool(getattr(state, 'recursive_file_search', False)),
        'sample_x_size': int(train_patch_size[0]),
        'sample_y_size': int(train_patch_size[1]),
        'train_patch_x_size': int(train_patch_size[0]),
        'train_patch_y_size': int(train_patch_size[1]),
        'recognition_patch_x_size': int(recognition_patch_size[0]),
        'recognition_patch_y_size': int(recognition_patch_size[1]),
        'optimizer_name': state.optimizer_name,
        'mixed_precision': state.mixed_precision,
        'loss_function': normalize_loss_term_name(state.loss_function)
        or dominant_loss_function(
            getattr(state, 'loss_term_weights', None),
            fallback='bce',
        ),
        'loss_term_weights_json': serialize_loss_term_weights(getattr(state, 'loss_term_weights', None)),
        'dice_loss_weight': float(state.dice_loss_weight),
        'iou_loss_weight': float(state.iou_loss_weight),
        'topograph_enabled': bool(getattr(state, 'topograph_enabled', False)),
        'topograph_loss_weight': float(getattr(state, 'topograph_loss_weight', 0.1)),
        'topograph_debug_viz': bool(getattr(state, 'topograph_debug_viz', False)),
        'topograph_num_processes': int(getattr(state, 'topograph_num_processes', 1)),
        'topograph_use_c': bool(getattr(state, 'topograph_use_c', False)),
        'learning_rate': float(state.learning_rate),
        'weight_decay': float(state.weight_decay),
        'early_stopping_enabled': bool(state.early_stopping_enabled),
        'early_stopping_patience': int(getattr(state, 'early_stopping_patience', 10)),
        'early_stopping_min_delta': float(getattr(state, 'early_stopping_min_delta', 0.0)),
        'early_stopping_restore_best_weights': bool(
            getattr(state, 'early_stopping_restore_best_weights', True)
        ),
        'warmup_enabled': bool(state.warmup_enabled),
        'deep_supervision': bool(getattr(state, 'deep_supervision', False)),
        'warmup_epochs': int(state.warmup_epochs),
        'warmup_start_factor': float(state.warmup_start_factor),
        'scheduler_name': normalize_scheduler_name(getattr(state, 'scheduler_name', 'off')),
        'scheduler_plateau_factor': float(getattr(state, 'scheduler_plateau_factor', 0.5)),
        'scheduler_plateau_patience': int(getattr(state, 'scheduler_plateau_patience', 3)),
        'scheduler_plateau_threshold': float(getattr(state, 'scheduler_plateau_threshold', 1e-4)),
        'scheduler_plateau_min_lr': float(getattr(state, 'scheduler_plateau_min_lr', 1e-6)),
        'scheduler_plateau_cooldown': int(getattr(state, 'scheduler_plateau_cooldown', 0)),
        'scheduler_cosine_t_max': int(getattr(state, 'scheduler_cosine_t_max', 10)),
        'scheduler_cosine_eta_min': float(getattr(state, 'scheduler_cosine_eta_min', 1e-6)),
        'scheduler_one_cycle_max_lr': float(getattr(state, 'scheduler_one_cycle_max_lr', 1e-3)),
        'scheduler_one_cycle_pct_start': float(getattr(state, 'scheduler_one_cycle_pct_start', 0.3)),
        'scheduler_one_cycle_anneal_strategy': str(
            getattr(state, 'scheduler_one_cycle_anneal_strategy', 'cos')
        ),
        'scheduler_one_cycle_div_factor': float(getattr(state, 'scheduler_one_cycle_div_factor', 25.0)),
        'scheduler_one_cycle_final_div_factor': float(
            getattr(state, 'scheduler_one_cycle_final_div_factor', 10000.0)
        ),
        'scheduler_one_cycle_three_phase': bool(getattr(state, 'scheduler_one_cycle_three_phase', False)),
        'scheduler_step_lr_step_size': int(getattr(state, 'scheduler_step_lr_step_size', 10)),
        'scheduler_step_lr_gamma': float(getattr(state, 'scheduler_step_lr_gamma', 0.1)),
        'hard_mining_enabled': bool(state.hard_mining_enabled),
        'hard_mining_strength': float(getattr(state, 'hard_mining_strength', 2.0)),
        'hard_mining_ema_alpha': float(getattr(state, 'hard_mining_ema_alpha', 0.3)),
        'sem_segmentation_config_json': json.dumps(
            getattr(state, 'sem_segmentation_config', {}), ensure_ascii=False, sort_keys=True
        ),
        'random_patch_size_enabled': bool(getattr(state, 'random_patch_size_enabled', False)),
        'random_patch_min_width': int(getattr(state, 'random_patch_min_size', (128, 128))[0]),
        'random_patch_min_height': int(getattr(state, 'random_patch_min_size', (128, 128))[1]),
        'random_patch_max_width': int(getattr(state, 'random_patch_max_size', (512, 512))[0]),
        'random_patch_max_height': int(getattr(state, 'random_patch_max_size', (512, 512))[1]),
        'hard_pixel_mining_enabled': bool(getattr(state, 'hard_pixel_mining_enabled', False)),
        'hard_pixel_mining_ratio': float(getattr(state, 'hard_pixel_mining_ratio', 0.25)),
        'cutout_enabled': bool(getattr(state, 'cutout_enabled', False)),
        'cutout_probability': float(getattr(state, 'cutout_probability', 1.0)),
        'cutout_holes': int(max(1, int(getattr(state, 'cutout_holes', 1)))),
        'cutout_size_ratio': float(getattr(state, 'cutout_size_ratio', 0.25)),
        'random_artifacts_enabled': bool(getattr(state, 'random_artifacts_enabled', False)),
        'random_artifacts_probability': float(getattr(state, 'random_artifacts_probability', 1.0)),
        'random_artifacts_count': int(max(1, int(getattr(state, 'random_artifacts_count', 1)))),
        'random_artifacts_size_ratio': float(getattr(state, 'random_artifacts_size_ratio', 0.25)),
        'random_artifacts_dust_enabled': bool(getattr(state, 'random_artifacts_dust_enabled', True)),
        'random_artifacts_resist_residue_enabled': bool(
            getattr(state, 'random_artifacts_resist_residue_enabled', True)
        ),
        'random_artifacts_etch_residue_enabled': bool(
            getattr(state, 'random_artifacts_etch_residue_enabled', True)
        ),
        'random_artifacts_particle_cluster_enabled': bool(
            getattr(state, 'random_artifacts_particle_cluster_enabled', True)
        ),
        'random_artifacts_flake_enabled': bool(getattr(state, 'random_artifacts_flake_enabled', True)),
        'mixup_enabled': bool(getattr(state, 'mixup_enabled', False)),
        'mixup_probability': float(getattr(state, 'mixup_probability', 1.0)),
        'mixup_alpha': float(getattr(state, 'mixup_alpha', 0.2)),
        'skip_uniform_labels': bool(state.skip_uniform_labels),
        'rare_patch_oversampling_enabled': bool(
            getattr(state, 'rare_patch_oversampling_enabled', False)
        ),
        'rare_patch_oversampling_factor': int(
            max(2, int(getattr(state, 'rare_patch_oversampling_factor', 2)))
        ),
        'use_multi_gpu': bool(multi_gpu_mode != 'off'),
        'multi_gpu_mode': multi_gpu_mode,
        'torch_compile_enabled': bool(state.torch_compile_enabled),
        'show_batch_preview': bool(state.show_batch_preview),
        'synthetic_defect_generator_json': json.dumps(
            _jsonify_value(getattr(state, 'synthetic_defect_generator', {})),
            ensure_ascii=False,
            sort_keys=True,
        ),
        'tech_aug_json': json.dumps(
            _jsonify_value(getattr(state, 'tech_aug', {})),
            ensure_ascii=False,
            sort_keys=True,
        ),
        'pcb_defects_json': json.dumps(
            _jsonify_value(getattr(state, 'pcb_defects', {})),
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
