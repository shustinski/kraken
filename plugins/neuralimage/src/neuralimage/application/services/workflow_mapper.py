from pathlib import Path

from neuralimage.application.dto import MainWindowState, SettingsState
from neuralimage.lib.data_interfaces import (
    build_pcb_defect_parameters,
    build_synthetic_defect_generator_parameters,
    build_tech_augmentation_config,
    normalize_confidence_save_mode,
    CutoutParameters,
    EarlyStoppingParameters,
    HardMiningParameters,
    MixupParameters,
    MixedPrecisionMode,
    OptimizerName,
    OptimizerParameters,
    RandomPatchSizeParameters,
    RandomArtifactsParameters,
    RecognitionParameters,
    SampleCutMode,
    SampleGenerationSettings,
    SamplePrepareSettings,
    SchedulerName,
    SchedulerParameters,
    TrainingParameters,
    ValidationSource,
    WarmupParameters,
    WorkMode,
    normalize_multi_gpu_mode,
    normalize_patch_batch_sync_mode,
    normalize_scheduler_name,
    normalize_validation_source,
    parse_work_mode,
)
from neuralimage.lib.loss_config import resolve_loss_term_weights
from neuralimage.configuration import build_sem_segmentation_config


def resolve_work_mode(value: str) -> WorkMode | None:
    return parse_work_mode(value)


def build_workflow_parameters(
    main_window: MainWindowState,
    settings: SettingsState,
) -> tuple[WorkMode | None, TrainingParameters, RecognitionParameters]:
    work_mode = resolve_work_mode(main_window.work_mode)
    channels = 3 if settings.color_mode == 'RGB' else 1
    cut_mode = SampleCutMode.online if settings.sample_cut_mode == SampleCutMode.online.value else SampleCutMode.disk
    train_patch_size = tuple(getattr(settings, 'train_patch_size', None) or settings.sample_size)
    recognition_patch_size = tuple(getattr(settings, 'recognition_patch_size', None) or settings.sample_size)
    train_batch_size = int(getattr(settings, 'train_batch_size', None) or settings.batch_size)
    recognition_batch_size = int(getattr(settings, 'recognition_batch_size', None) or settings.batch_size)
    patch_batch_sync_mode = normalize_patch_batch_sync_mode(getattr(settings, 'patch_batch_sync_mode', ''))
    sync_patch_sizes = bool(
        getattr(settings, 'sync_patch_sizes', patch_batch_sync_mode in ('patch', 'patch_and_batch'))
    )
    if sync_patch_sizes:
        recognition_patch_size = train_patch_size
    local_crop_size = tuple(getattr(settings, 'local_crop_size', None) or train_patch_size)
    context_crop_size = getattr(settings, 'context_crop_size', None)
    context_input_size = getattr(settings, 'context_input_size', None)
    requested_context_branch = getattr(settings, 'use_context_branch', None)
    sem_raw = getattr(settings, 'sem_segmentation_config', {})
    sem_config = build_sem_segmentation_config(sem_raw)
    sem_enabled = bool(sem_raw)
    if sem_enabled:
        requested_context_branch = sem_config.context.enabled
    if requested_context_branch is None:
        requested_context_branch = settings.model in {'quasi_dual_scale_unet', 'UNetWithContextBranch'}
    model = (
        settings.model
        if work_mode in (WorkMode.train_only, WorkMode.train_and_recognition)
        else Path(main_window.model_path)
    )

    prep = SamplePrepareSettings(
        enable_crop=settings.crop_enabled,
        enable_resize=settings.resize_enabled,
        edge_cut=(settings.edge_cut_size, settings.edge_cut_size),
        target_size=(
            settings.target_size
            if max(1, int(getattr(settings, 'compression_factor', 1))) <= 1
            else None
        ),
        compression_factor=max(1, int(getattr(settings, 'compression_factor', 1))),
    )

    generation = SampleGenerationSettings(
        step=settings.step,
        segment_size=train_patch_size,
        vertical_rotation=settings.vertical_rotation,
        horizontal_rotation=settings.horizontal_rotation,
        channels=channels,
        flip_x=bool(getattr(settings, 'flip_x', False)),
        flip_y=bool(getattr(settings, 'flip_y', False)),
        additional_augmentation=settings.additional_augmentation,
        augmentation_brightness_strength=settings.augmentation_brightness_strength,
        augmentation_contrast_strength=settings.augmentation_contrast_strength,
        augmentation_gamma_strength=float(getattr(settings, 'augmentation_gamma_strength', 0.15)),
        augmentation_noise_probability=settings.augmentation_noise_probability,
        augmentation_noise_sigma=settings.augmentation_noise_sigma,
        augmentation_blur_probability=float(getattr(settings, 'augmentation_blur_probability', 0.25)),
        augmentation_blur_radius=float(getattr(settings, 'augmentation_blur_radius', 1.0)),
        shuffle_patches_in_frame=bool(
            getattr(settings, 'shuffle_patches_in_frame', getattr(settings, 'shuffle', True))
        ),
        random_crop=bool(getattr(settings, 'random_crop', False)),
        crops_per_image=int(getattr(settings, 'crops_per_image', 64)),
        scale_augmentation=bool(getattr(settings, 'scale_augmentation', False)),
        scale_augmentation_strength=float(getattr(settings, 'scale_augmentation_strength', 0.2)),
        tech_aug=build_tech_augmentation_config(getattr(settings, 'tech_aug', None)),
    )
    raw_synthetic_defect_generator = getattr(settings, 'synthetic_defect_generator', None)
    if not raw_synthetic_defect_generator:
        raw_synthetic_defect_generator = getattr(settings, 'pcb_defects', None)
    synthetic_defect_generator = build_synthetic_defect_generator_parameters(
        raw_synthetic_defect_generator
    )

    try:
        optimizer_name = OptimizerName(settings.optimizer_name)
    except ValueError:
        optimizer_name = OptimizerName.adam
    try:
        mixed_precision = MixedPrecisionMode(settings.mixed_precision)
    except ValueError:
        mixed_precision = MixedPrecisionMode.off
    try:
        scheduler_name = SchedulerName(
            normalize_scheduler_name(getattr(settings, 'scheduler_name', SchedulerName.off.value))
        )
    except ValueError:
        scheduler_name = SchedulerName.off
    multi_gpu_mode = normalize_multi_gpu_mode(
        getattr(settings, 'multi_gpu_mode', ''),
        use_multi_gpu_fallback=bool(getattr(settings, 'use_multi_gpu', False)),
    )

    pcb_defects = build_pcb_defect_parameters(
        getattr(settings, 'pcb_defects', None) or synthetic_defect_generator.pcb_defects
    )
    pcb_defects.use_defect_mask_as_label = False

    training = TrainingParameters(
        image_path=Path(main_window.sample_folder),
        label_path=Path(main_window.label_folder),
        shuffle=settings.shuffle,
        validation=settings.use_validation,
        validation_percent=settings.validation_percent,
        batch_size=train_batch_size,
        cut_mode=cut_mode,
        colors=channels,
        epochs=main_window.epochs,
        generation=generation,
        prepare=prep,
        validation_source=normalize_validation_source(
            getattr(settings, 'validation_source', ValidationSource.split.value)
        ),
        validation_image_path=(
            Path(getattr(settings, 'validation_image_folder', ''))
            if str(getattr(settings, 'validation_image_folder', '')).strip()
            else None
        ),
        validation_label_path=(
            Path(getattr(settings, 'validation_label_folder', ''))
            if str(getattr(settings, 'validation_label_folder', '')).strip()
            else None
        ),
        save_validation_binary_images=bool(
            getattr(settings, 'save_validation_binary_images', False)
        ),
        optimizer=OptimizerParameters(
            name=optimizer_name,
            learning_rate=settings.learning_rate,
            weight_decay=settings.weight_decay,
        ),
        mixed_precision=mixed_precision,
        loss_function=settings.loss_function,
        loss_term_weights=resolve_loss_term_weights(
            getattr(settings, 'loss_term_weights', None),
            fallback_loss_function=settings.loss_function,
        ),
        dice_loss_weight=settings.dice_loss_weight,
        iou_loss_weight=settings.iou_loss_weight,
        early_stopping=EarlyStoppingParameters(
            enabled=settings.early_stopping_enabled,
            patience=int(getattr(settings, 'early_stopping_patience', 10)),
            min_delta=float(getattr(settings, 'early_stopping_min_delta', 0.0)),
            restore_best_weights=bool(
                getattr(settings, 'early_stopping_restore_best_weights', True)
            ),
        ),
        warmup=WarmupParameters(
            enabled=settings.warmup_enabled,
            epochs=settings.warmup_epochs,
            start_factor=settings.warmup_start_factor,
        ),
        scheduler=SchedulerParameters(
            name=scheduler_name,
            plateau_factor=float(getattr(settings, 'scheduler_plateau_factor', 0.5)),
            plateau_patience=max(0, int(getattr(settings, 'scheduler_plateau_patience', 3))),
            plateau_threshold=max(0.0, float(getattr(settings, 'scheduler_plateau_threshold', 1e-4))),
            plateau_min_lr=max(0.0, float(getattr(settings, 'scheduler_plateau_min_lr', 1e-6))),
            plateau_cooldown=max(0, int(getattr(settings, 'scheduler_plateau_cooldown', 0))),
            cosine_t_max=max(1, int(getattr(settings, 'scheduler_cosine_t_max', 10))),
            cosine_eta_min=max(0.0, float(getattr(settings, 'scheduler_cosine_eta_min', 1e-6))),
            one_cycle_max_lr=max(0.0, float(getattr(settings, 'scheduler_one_cycle_max_lr', 1e-3))),
            one_cycle_pct_start=float(
                min(max(getattr(settings, 'scheduler_one_cycle_pct_start', 0.3), 0.0), 1.0)
            ),
            one_cycle_anneal_strategy=str(
                getattr(settings, 'scheduler_one_cycle_anneal_strategy', 'cos') or 'cos'
            ).strip().lower(),
            one_cycle_div_factor=max(1.0, float(getattr(settings, 'scheduler_one_cycle_div_factor', 25.0))),
            one_cycle_final_div_factor=max(
                1.0,
                float(getattr(settings, 'scheduler_one_cycle_final_div_factor', 10000.0)),
            ),
            one_cycle_three_phase=bool(getattr(settings, 'scheduler_one_cycle_three_phase', False)),
            step_lr_step_size=max(1, int(getattr(settings, 'scheduler_step_lr_step_size', 10))),
            step_lr_gamma=float(min(max(getattr(settings, 'scheduler_step_lr_gamma', 0.1), 0.0), 1.0)),
        ),
        hard_mining=HardMiningParameters(
            enabled=(settings.hard_mining_enabled or (sem_enabled and sem_config.hard_mining.mode != 'off')),
            strength=float(getattr(settings, 'hard_mining_strength', 2.0)),
            pixel_enabled=getattr(settings, 'hard_pixel_mining_enabled', False),
            pixel_keep_ratio=float(getattr(settings, 'hard_pixel_mining_ratio', 0.25)),
            mode=sem_config.hard_mining.mode if sem_enabled else 'online',
            geometry_weight=sem_config.hard_mining.geometry_weight if sem_enabled else 0.5,
            loss_weight=sem_config.hard_mining.loss_weight if sem_enabled else 0.5,
            exploration_floor=sem_config.hard_mining.exploration_floor if sem_enabled else 0.1,
            ema_alpha=(
                sem_config.hard_mining.ema_alpha
                if sem_enabled
                else float(getattr(settings, 'hard_mining_ema_alpha', 0.3))
            ),
            score_clip=sem_config.hard_mining.score_clip if sem_enabled else 5.0,
            refresh_epochs=sem_config.hard_mining.refresh_epochs if sem_enabled else 1,
            offline_manifest=sem_config.hard_mining.offline_manifest if sem_enabled else None,
        ),
        random_patch_size=RandomPatchSizeParameters(
            enabled=(
                bool(getattr(settings, 'random_patch_size_enabled', False))
                and settings.sample_cut_mode == SampleCutMode.online.value
            ),
            min_size=tuple(getattr(settings, 'random_patch_min_size', (128, 128))),
            max_size=tuple(getattr(settings, 'random_patch_max_size', (512, 512))),
        ),
        cutout=CutoutParameters(
            enabled=bool(getattr(settings, 'cutout_enabled', False)),
            probability=float(getattr(settings, 'cutout_probability', 1.0)),
            holes=max(1, int(getattr(settings, 'cutout_holes', 1))),
            size_ratio=float(getattr(settings, 'cutout_size_ratio', 0.25)),
        ),
        random_artifacts=RandomArtifactsParameters(
            enabled=bool(getattr(settings, 'random_artifacts_enabled', False)),
            probability=float(getattr(settings, 'random_artifacts_probability', 1.0)),
            count=max(1, int(getattr(settings, 'random_artifacts_count', 1))),
            size_ratio=float(getattr(settings, 'random_artifacts_size_ratio', 0.25)),
            dust_enabled=bool(getattr(settings, 'random_artifacts_dust_enabled', True)),
            resist_residue_enabled=bool(getattr(settings, 'random_artifacts_resist_residue_enabled', True)),
            etch_residue_enabled=bool(getattr(settings, 'random_artifacts_etch_residue_enabled', True)),
            particle_cluster_enabled=bool(getattr(settings, 'random_artifacts_particle_cluster_enabled', True)),
            flake_enabled=bool(getattr(settings, 'random_artifacts_flake_enabled', True)),
        ),
        mixup=MixupParameters(
            enabled=bool(getattr(settings, 'mixup_enabled', False)),
            probability=float(getattr(settings, 'mixup_probability', 1.0)),
            alpha=float(getattr(settings, 'mixup_alpha', 0.2)),
        ),
        skip_uniform_labels=settings.skip_uniform_labels,
        rare_patch_oversampling_enabled=bool(
            getattr(settings, 'rare_patch_oversampling_enabled', False)
        ),
        rare_patch_oversampling_factor=max(
            2,
            int(getattr(settings, 'rare_patch_oversampling_factor', 2)),
        ),
        use_multi_gpu=multi_gpu_mode != 'off',
        multi_gpu_mode=multi_gpu_mode,
        show_batch_preview=settings.show_batch_preview,
        log_update_frequency=settings.log_update_frequency,
        local_crop_size=local_crop_size,
        context_crop_size=tuple(context_crop_size) if context_crop_size is not None else None,
        context_input_size=tuple(context_input_size) if context_input_size is not None else None,
        context_branch_channels=tuple(getattr(settings, 'context_branch_channels', (16, 32, 64, 128))),
        fusion_type=(sem_config.context.fusion_type if sem_enabled else str(getattr(settings, 'fusion_type', 'concat'))),
        use_context_branch=bool(requested_context_branch),
        use_cross_attention=(sem_config.context.cross_attention if sem_enabled else True),
        attention_dim=(sem_config.context.attention_dim if sem_enabled else 128),
        attention_heads=(sem_config.context.attention_heads if sem_enabled else 4),
        attention_max_global_tokens=(sem_config.context.max_global_tokens if sem_enabled else 1024),
        deep_supervision=bool(getattr(settings, 'deep_supervision', False)),
        dataloader_num_workers=int(getattr(settings, 'dataloader_num_workers', -1)),
        recursive_file_search=bool(getattr(settings, 'recursive_file_search', False)),
        pcb_defects=pcb_defects,
        synthetic_defect_generator=synthetic_defect_generator,
        supervision_targets=sem_config.targets,
        preprocessing=sem_config.preprocessing,
        sem_augmentation=sem_config.augmentation,
        uncertainty=sem_config.uncertainty,
        active_learning=sem_config.active_learning,
        advanced_validation=sem_config.validation.enabled,
        advanced_validation_full_frame=sem_config.validation.full_frame,
        advanced_validation_boundary_tolerance=sem_config.validation.boundary_tolerance,
        advanced_validation_include_hd95=sem_config.validation.include_hd95,
        advanced_validation_confidence_bins=sem_config.validation.confidence_bins,
        loss_weighting_strategy=sem_config.losses.weighting_strategy,
        mask_loss_weight_floor=sem_config.losses.mask_weight_floor,
    )

    recognition = RecognitionParameters(
        source_files=[],
        source_folder=(
            Path(main_window.source_folder)
            if str(main_window.source_folder).strip()
            else None
        ),
        result_folder=Path(main_window.result_folder),
        model=model,
        batch_size=recognition_batch_size,
        part_size=recognition_patch_size,
        overlap=settings.overlap,
        jpeg_quality=int(getattr(settings, 'recognition_jpeg_quality', 95)),
        recognition_multiprocessing_enabled=bool(
            getattr(settings, 'recognition_multiprocessing_enabled', True)
        ),
        binarize_output=bool(getattr(settings, 'recognition_binarize_output', True)),
        use_auto_threshold=bool(getattr(settings, 'recognition_use_auto_threshold', True)),
        threshold=float(getattr(settings, 'recognition_threshold', 0.5)),
        postprocess_enabled=bool(getattr(settings, 'recognition_postprocess', False)),
        postprocess_kernel_size=max(
            1,
            int(getattr(settings, 'recognition_postprocess_kernel_size', 3)),
        ),
        recognition_tta_enabled=bool(getattr(settings, 'recognition_tta_enabled', False)),
        confidence_tta_enabled=bool(getattr(settings, 'confidence_tta_enabled', False)),
        confidence_save_mode=normalize_confidence_save_mode(
            getattr(settings, 'confidence_save_mode', 'off')
        ),
        use_context_branch=(
            bool(requested_context_branch)
            if getattr(settings, 'use_context_branch', None) is not None
            else None
        ),
        use_cross_attention=(sem_config.context.cross_attention if sem_enabled else None),
        context_crop_size=tuple(context_crop_size) if context_crop_size is not None else None,
        context_input_size=tuple(context_input_size) if context_input_size is not None else None,
        recursive_file_search=bool(getattr(settings, 'recursive_file_search', False)),
        compression_factor=max(1, int(getattr(settings, 'compression_factor', 1))),
        preprocessing=sem_config.preprocessing,
        uncertainty=sem_config.uncertainty,
        active_learning=sem_config.active_learning,
        sem_config_hash=sem_config.stable_hash() if sem_enabled else None,
    )

    return work_mode, training, recognition
