from pathlib import Path

from application.dto import MainWindowState, SettingsState
from lib.data_interfaces import (
    CutoutParameters,
    EarlyStoppingParameters,
    HardMiningParameters,
    MixupParameters,
    MixedPrecisionMode,
    OptimizerName,
    OptimizerParameters,
    RecognitionParameters,
    SampleCutMode,
    SampleGenerationSettings,
    SamplePrepareSettings,
    TrainingParameters,
    WarmupParameters,
    WorkMode,
    normalize_multi_gpu_mode,
    normalize_patch_batch_sync_mode,
    parse_work_mode,
)
from lib.file_func import filter_images
from lib.loss_config import resolve_loss_term_weights


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
    model = (
        settings.model
        if work_mode in (WorkMode.train_only, WorkMode.train_and_recognition)
        else Path(main_window.model_path)
    )

    prep = SamplePrepareSettings(
        enable_crop=settings.crop_enabled,
        enable_resize=settings.resize_enabled,
        edge_cut=(settings.edge_cut_size, settings.edge_cut_size),
        target_size=settings.target_size,
    )

    generation = SampleGenerationSettings(
        step=settings.step,
        segment_size=train_patch_size,
        vertical_rotation=settings.vertical_rotation,
        horizontal_rotation=settings.horizontal_rotation,
        channels=channels,
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
    )

    try:
        optimizer_name = OptimizerName(settings.optimizer_name)
    except ValueError:
        optimizer_name = OptimizerName.adam
    try:
        mixed_precision = MixedPrecisionMode(settings.mixed_precision)
    except ValueError:
        mixed_precision = MixedPrecisionMode.bf16
    multi_gpu_mode = normalize_multi_gpu_mode(
        getattr(settings, 'multi_gpu_mode', ''),
        use_multi_gpu_fallback=bool(getattr(settings, 'use_multi_gpu', False)),
    )

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
            patience=settings.early_stopping_patience,
            min_delta=settings.early_stopping_min_delta,
            restore_best_weights=settings.early_stopping_restore_best_weights,
        ),
        warmup=WarmupParameters(
            enabled=settings.warmup_enabled,
            epochs=settings.warmup_epochs,
            start_factor=settings.warmup_start_factor,
        ),
        hard_mining=HardMiningParameters(
            enabled=settings.hard_mining_enabled,
            strength=settings.hard_mining_strength,
            ema_alpha=settings.hard_mining_ema_alpha,
            pixel_enabled=getattr(settings, 'hard_pixel_mining_enabled', False),
            pixel_keep_ratio=float(getattr(settings, 'hard_pixel_mining_ratio', 0.25)),
        ),
        cutout=CutoutParameters(
            enabled=bool(getattr(settings, 'cutout_enabled', False)),
            probability=float(getattr(settings, 'cutout_probability', 1.0)),
            holes=max(1, int(getattr(settings, 'cutout_holes', 1))),
            size_ratio=float(getattr(settings, 'cutout_size_ratio', 0.25)),
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
    )

    recognition = RecognitionParameters(
        source_files=filter_images(Path(main_window.source_folder)),
        result_folder=Path(main_window.result_folder),
        model=model,
        batch_size=recognition_batch_size,
        part_size=recognition_patch_size,
        overlap=settings.overlap,
        jpeg_quality=int(getattr(settings, 'recognition_jpeg_quality', 95)),
        binarize_output=bool(getattr(settings, 'recognition_binarize_output', True)),
        use_auto_threshold=bool(getattr(settings, 'recognition_use_auto_threshold', True)),
        threshold=float(getattr(settings, 'recognition_threshold', 0.5)),
        postprocess_enabled=bool(getattr(settings, 'recognition_postprocess', False)),
        postprocess_kernel_size=max(
            1,
            int(getattr(settings, 'recognition_postprocess_kernel_size', 3)),
        ),
    )

    return work_mode, training, recognition
