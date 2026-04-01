import configparser
import json
import os
from dataclasses import is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QSettings

from application.dto import MainWindowState, SettingsState
from application.ports import StateStore
from application.services.workflow_mapper import build_workflow_parameters
from lib.data_interfaces import (
    WorkMode,
    normalize_confidence_save_mode,
    normalize_multi_gpu_mode,
    normalize_patch_batch_sync_mode,
    normalize_scheduler_name,
    normalize_validation_source,
    normalize_work_mode,
)
from lib.loss_config import (
    deserialize_loss_term_weights,
    dominant_loss_function,
    normalize_loss_term_name,
    resolve_loss_term_weights,
    serialize_loss_term_weights,
)


MAIN_WINDOW_ORG = 'NeuralImage'
MAIN_WINDOW_APP = 'MainWindow'
SETTINGS_ORG = 'NeuralImage'
SETTINGS_APP = 'Settings'

INI_SECTION_MAIN = 'main_window'
INI_SECTION_SETTINGS = 'settings'
WORKFLOW_SNAPSHOT_FILENAME = 'neuralimage_workflow.json'
WORKFLOW_SNAPSHOT_FORMAT_VERSION = 1


def _build_main_window_state(
    *,
    read_str,
    read_int,
) -> MainWindowState:
    defaults = MainWindowState()
    return MainWindowState(
        work_mode=normalize_work_mode(read_str('work_mode', defaults.work_mode)),
        source_folder=read_str('source_path', defaults.source_folder),
        result_folder=read_str('result_path', defaults.result_folder),
        model_path=read_str('model_path', defaults.model_path),
        label_folder=read_str('label_path', defaults.label_folder),
        sample_folder=read_str('sample_path', defaults.sample_folder),
        epochs=read_int('epochs', defaults.epochs),
        ui_mode=read_str('ui_mode', defaults.ui_mode),
    )


def _main_window_state_to_storage_dict(state: MainWindowState) -> dict[str, str | int]:
    return {
        'work_mode': normalize_work_mode(state.work_mode),
        'source_path': state.source_folder,
        'result_path': state.result_folder,
        'model_path': state.model_path,
        'label_path': state.label_folder,
        'sample_path': state.sample_folder,
        'epochs': int(state.epochs),
        'ui_mode': str(getattr(state, 'ui_mode', 'simple') or 'simple'),
    }


def _main_window_state_from_storage_dict(data: dict[str, Any]) -> MainWindowState:
    return _build_main_window_state(
        read_str=lambda key, default: _coerce_str(data.get(key), default),
        read_int=lambda key, default: _coerce_int(data.get(key), default),
    )


def _build_settings_state(
    *,
    read_bool,
    read_int,
    read_float,
    read_str,
) -> SettingsState:
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
    return SettingsState(
        step=read_int('cut_step', defaults.step),
        vertical_rotation=read_bool('vertical_rotation', defaults.vertical_rotation),
        horizontal_rotation=read_bool('horizontal_rotation', defaults.horizontal_rotation),
        flip_x=read_bool('flip_x', getattr(defaults, 'flip_x', False)),
        flip_y=read_bool('flip_y', getattr(defaults, 'flip_y', False)),
        additional_augmentation=read_bool('additional_augmentation', defaults.additional_augmentation),
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
        optimizer_name=read_str('optimizer_name', defaults.optimizer_name),
        mixed_precision=read_str('mixed_precision', defaults.mixed_precision),
        loss_function=normalize_loss_term_name(legacy_loss_function)
        or dominant_loss_function(loss_term_weights, fallback=defaults.loss_function),
        loss_term_weights=loss_term_weights,
        dice_loss_weight=dice_loss_weight,
        iou_loss_weight=iou_loss_weight,
        learning_rate=read_float('learning_rate', defaults.learning_rate),
        weight_decay=read_float('weight_decay', defaults.weight_decay),
        early_stopping_enabled=read_bool('early_stopping_enabled', defaults.early_stopping_enabled),
        early_stopping_patience=read_int('early_stopping_patience', defaults.early_stopping_patience),
        early_stopping_min_delta=read_float('early_stopping_min_delta', defaults.early_stopping_min_delta),
        early_stopping_restore_best_weights=read_bool(
            'early_stopping_restore_best_weights', defaults.early_stopping_restore_best_weights
        ),
        warmup_enabled=read_bool('warmup_enabled', defaults.warmup_enabled),
        deep_supervision=read_bool(
            'deep_supervision',
            getattr(defaults, 'deep_supervision', True),
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
        hard_mining_strength=read_float('hard_mining_strength', defaults.hard_mining_strength),
        hard_mining_ema_alpha=read_float('hard_mining_ema_alpha', defaults.hard_mining_ema_alpha),
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


def _settings_state_from_storage_dict(data: dict[str, Any]) -> SettingsState:
    return _build_settings_state(
        read_bool=lambda key, default: _coerce_bool(data.get(key), default),
        read_int=lambda key, default: _coerce_int(data.get(key), default),
        read_float=lambda key, default: _coerce_float(data.get(key), default),
        read_str=lambda key, default: _coerce_str(data.get(key), default),
    )


def _settings_state_to_storage_dict(state: SettingsState) -> dict[str, str | int | float | bool]:
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
        'augmentation_brightness_strength': float(state.augmentation_brightness_strength),
        'augmentation_contrast_strength': float(state.augmentation_contrast_strength),
        'augmentation_gamma_strength': float(getattr(state, 'augmentation_gamma_strength', 0.15)),
        'augmentation_noise_probability': float(state.augmentation_noise_probability),
        'augmentation_noise_sigma': float(state.augmentation_noise_sigma),
        'augmentation_blur_probability': float(getattr(state, 'augmentation_blur_probability', 0.25)),
        'augmentation_blur_radius': float(getattr(state, 'augmentation_blur_radius', 1.0)),
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
        'sample_x_size': int(train_patch_size[0]),
        'sample_y_size': int(train_patch_size[1]),
        'train_patch_x_size': int(train_patch_size[0]),
        'train_patch_y_size': int(train_patch_size[1]),
        'recognition_patch_x_size': int(recognition_patch_size[0]),
        'recognition_patch_y_size': int(recognition_patch_size[1]),
        'target_x_size': int(state.target_size[0]),
        'target_y_size': int(state.target_size[1]),
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
        'learning_rate': float(state.learning_rate),
        'weight_decay': float(state.weight_decay),
        'early_stopping_enabled': bool(state.early_stopping_enabled),
        'early_stopping_patience': int(state.early_stopping_patience),
        'early_stopping_min_delta': float(state.early_stopping_min_delta),
        'early_stopping_restore_best_weights': bool(state.early_stopping_restore_best_weights),
        'warmup_enabled': bool(state.warmup_enabled),
        'deep_supervision': bool(getattr(state, 'deep_supervision', True)),
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
        'hard_mining_strength': float(state.hard_mining_strength),
        'hard_mining_ema_alpha': float(state.hard_mining_ema_alpha),
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


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'1', 'true', 'yes', 'on'}:
            return True
        if normalized in {'0', 'false', 'no', 'off'}:
            return False
    if value is None:
        return default
    return bool(value)


def _coerce_str(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_json_object(value: Any, default: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    raw = _coerce_str(value, '')
    if not raw.strip():
        return dict(default)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return dict(default)
    if not isinstance(payload, dict):
        return dict(default)
    return payload


def _jsonify_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _jsonify_value(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonify_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify_value(item) for item in value]
    return value


def _sanitize_workflow_snapshot_payload(training: Any, recognition: Any) -> tuple[Any, Any]:
    serialized_training = _jsonify_value(training)
    serialized_recognition = _jsonify_value(recognition)
    if isinstance(serialized_recognition, dict):
        serialized_recognition.pop('source_files', None)
    return serialized_training, serialized_recognition


def resolve_workflow_snapshot_path(main_state: MainWindowState) -> Path:
    sample_folder = Path(main_state.sample_folder)
    label_folder = Path(main_state.label_folder)
    fallback_parent = sample_folder.parent if str(sample_folder) else Path.cwd()
    if not str(sample_folder) or not str(label_folder):
        return fallback_parent / WORKFLOW_SNAPSHOT_FILENAME
    target_dir = sample_folder.parent
    if sample_folder.parent == label_folder.parent:
        target_dir = sample_folder.parent
    return target_dir / WORKFLOW_SNAPSHOT_FILENAME


def create_workflow_snapshot_payload(
    main_state: MainWindowState,
    settings_state: SettingsState,
    workflow_snapshot: tuple[WorkMode | None, Any, Any] | None = None,
) -> dict[str, Any]:
    work_mode, training, recognition = workflow_snapshot or build_workflow_parameters(main_state, settings_state)
    serialized_training, serialized_recognition = _sanitize_workflow_snapshot_payload(training, recognition)
    return {
        'format_version': WORKFLOW_SNAPSHOT_FORMAT_VERSION,
        'saved_at': datetime.now(timezone.utc).isoformat(),
        'main_window_state': _main_window_state_to_storage_dict(main_state),
        'settings_state': _settings_state_to_storage_dict(settings_state),
        'workflow': {
            'work_mode': work_mode.value if work_mode is not None else None,
            'training': serialized_training,
            'recognition': serialized_recognition,
        },
    }


def save_workflow_snapshot(
    main_state: MainWindowState,
    settings_state: SettingsState,
    destination: Path | None = None,
    workflow_snapshot: tuple[WorkMode | None, Any, Any] | None = None,
) -> Path:
    snapshot_path = Path(destination) if destination is not None else resolve_workflow_snapshot_path(main_state)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    payload = create_workflow_snapshot_payload(main_state, settings_state, workflow_snapshot=workflow_snapshot)
    with snapshot_path.open('w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return snapshot_path


def load_workflow_snapshot(snapshot_path: Path | str) -> tuple[MainWindowState, SettingsState]:
    path = Path(snapshot_path)
    try:
        with path.open('r', encoding='utf-8') as file:
            payload = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f'Некорректный JSON: {error}') from error

    if not isinstance(payload, dict):
        raise ValueError('Файл параметров должен содержать JSON-объект.')
    if payload.get('format_version') != WORKFLOW_SNAPSHOT_FORMAT_VERSION:
        raise ValueError('Неподдерживаемая версия файла параметров.')

    main_payload = payload.get('main_window_state')
    settings_payload = payload.get('settings_state')
    if not isinstance(main_payload, dict) or not isinstance(settings_payload, dict):
        raise ValueError('В файле параметров отсутствуют main_window_state или settings_state.')

    main_state = _main_window_state_from_storage_dict(main_payload)
    settings_state = _settings_state_from_storage_dict(settings_payload)
    return main_state, settings_state


class QSettingsStateStore:
    def _settings(self, organization: str, application: str) -> QSettings:
        root = os.getenv('NEURALIMAGE_SETTINGS_DIR')
        if root:
            settings_root = Path(root)
            settings_root.mkdir(parents=True, exist_ok=True)
            return QSettings(
                str(settings_root / f'{organization}_{application}.ini'),
                QSettings.Format.IniFormat,
            )
        return QSettings(organization, application)

    def load_main_window_state(self) -> MainWindowState:
        settings = self._settings(MAIN_WINDOW_ORG, MAIN_WINDOW_APP)
        state = _build_main_window_state(
            read_str=lambda key, default: settings.value(key, default, type=str),
            read_int=lambda key, default: settings.value(key, default, type=int),
        )
        settings.sync()
        return state

    def save_main_window_state(self, state: MainWindowState) -> None:
        settings = self._settings(MAIN_WINDOW_ORG, MAIN_WINDOW_APP)
        for key, value in _main_window_state_to_storage_dict(state).items():
            settings.setValue(key, value)
        settings.sync()

    def load_settings_state(self) -> SettingsState:
        settings = self._settings(SETTINGS_ORG, SETTINGS_APP)
        state = _build_settings_state(
            read_bool=lambda key, default: settings.value(key, default, type=bool),
            read_int=lambda key, default: settings.value(key, default, type=int),
            read_float=lambda key, default: settings.value(key, default, type=float),
            read_str=lambda key, default: settings.value(key, default, type=str),
        )
        settings.sync()
        return state

    def save_settings_state(self, state: SettingsState) -> None:
        settings = self._settings(SETTINGS_ORG, SETTINGS_APP)
        for key, value in _settings_state_to_storage_dict(state).items():
            settings.setValue(key, value)
        settings.sync()


class IniStateStore:
    def __init__(self, ini_path: Path | None = None):
        self._ini_path = ini_path or self._resolve_ini_path()
        self._ini_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_ini_path() -> Path:
        explicit_path = os.getenv('NEURALIMAGE_INI_PATH')
        if explicit_path:
            return Path(explicit_path)

        settings_root = os.getenv('NEURALIMAGE_SETTINGS_DIR')
        if settings_root:
            return Path(settings_root) / 'neuralimage_state.ini'

        return Path.cwd() / 'neuralimage_state.ini'

    def _load_parser(self) -> configparser.ConfigParser:
        parser = configparser.ConfigParser()
        if self._ini_path.exists():
            parser.read(self._ini_path, encoding='utf-8')
        return parser

    def _write_parser(self, parser: configparser.ConfigParser) -> None:
        with self._ini_path.open('w', encoding='utf-8') as file:
            parser.write(file)

    @staticmethod
    def _get_bool(parser: configparser.ConfigParser, section: str, key: str, default: bool) -> bool:
        try:
            return parser.getboolean(section, key)
        except (ValueError, configparser.NoOptionError, configparser.NoSectionError):
            return default

    @staticmethod
    def _get_int(parser: configparser.ConfigParser, section: str, key: str, default: int) -> int:
        try:
            return parser.getint(section, key)
        except (ValueError, configparser.NoOptionError, configparser.NoSectionError):
            return default

    @staticmethod
    def _get_float(parser: configparser.ConfigParser, section: str, key: str, default: float) -> float:
        try:
            return parser.getfloat(section, key)
        except (ValueError, configparser.NoOptionError, configparser.NoSectionError):
            return default

    @staticmethod
    def _get_str(parser: configparser.ConfigParser, section: str, key: str, default: str) -> str:
        try:
            return parser.get(section, key)
        except (configparser.NoOptionError, configparser.NoSectionError):
            return default

    def load_main_window_state(self) -> MainWindowState:
        parser = self._load_parser()
        return _build_main_window_state(
            read_str=lambda key, default: self._get_str(parser, INI_SECTION_MAIN, key, default),
            read_int=lambda key, default: self._get_int(parser, INI_SECTION_MAIN, key, default),
        )

    def save_main_window_state(self, state: MainWindowState) -> None:
        parser = self._load_parser()
        parser[INI_SECTION_MAIN] = {
            key: str(value) for key, value in _main_window_state_to_storage_dict(state).items()
        }
        self._write_parser(parser)

    def load_settings_state(self) -> SettingsState:
        parser = self._load_parser()
        return _build_settings_state(
            read_bool=lambda key, default: self._get_bool(parser, INI_SECTION_SETTINGS, key, default),
            read_int=lambda key, default: self._get_int(parser, INI_SECTION_SETTINGS, key, default),
            read_float=lambda key, default: self._get_float(parser, INI_SECTION_SETTINGS, key, default),
            read_str=lambda key, default: self._get_str(parser, INI_SECTION_SETTINGS, key, default),
        )

    def save_settings_state(self, state: SettingsState) -> None:
        parser = self._load_parser()
        parser[INI_SECTION_SETTINGS] = {
            key: str(value) for key, value in _settings_state_to_storage_dict(state).items()
        }
        self._write_parser(parser)


def create_state_store(*, default_backend: str = 'qsettings') -> StateStore:
    backend = os.getenv('NEURALIMAGE_STATE_BACKEND', default_backend).strip().lower()
    if backend == 'ini':
        return IniStateStore()
    return QSettingsStateStore()


def load_main_window_state() -> MainWindowState:
    return create_state_store().load_main_window_state()


def save_main_window_state(state: MainWindowState) -> None:
    create_state_store().save_main_window_state(state)


def load_settings_state() -> SettingsState:
    return create_state_store().load_settings_state()


def save_settings_state(state: SettingsState) -> None:
    create_state_store().save_settings_state(state)
