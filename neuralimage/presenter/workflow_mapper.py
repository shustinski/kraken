from pathlib import Path

from lib.data_interfaces import (
    WorkMode,
    parse_work_mode,
    SampleCutMode,
    OptimizerName,
    MixedPrecisionMode,
    OptimizerParameters,
    EarlyStoppingParameters,
    WarmupParameters,
    HardMiningParameters,
    SamplePrepareSettings,
    SampleGenerationSettings,
    TrainingParameters,
    RecognitionParameters,
)
from lib.file_func import filter_images
from view.window_dataclasses import MainWindowState, SettingsState


def resolve_work_mode(value: str) -> WorkMode | None:
    return parse_work_mode(value)


def build_workflow_parameters(
    main_window: MainWindowState, settings: SettingsState
) -> tuple[WorkMode | None, TrainingParameters, RecognitionParameters]:
    work_mode = resolve_work_mode(main_window.work_mode)
    channels = 3 if settings.color_mode == 'RGB' else 1
    cut_mode = SampleCutMode.online if settings.sample_cut_mode == SampleCutMode.online.value else SampleCutMode.disk
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
        segment_size=settings.sample_size,
        vertical_rotation=settings.vertical_rotation,
        horizontal_rotation=settings.horizontal_rotation,
        channels=channels,
        additional_augmentation=settings.additional_augmentation,
        augmentation_brightness_strength=settings.augmentation_brightness_strength,
        augmentation_contrast_strength=settings.augmentation_contrast_strength,
        augmentation_noise_probability=settings.augmentation_noise_probability,
        augmentation_noise_sigma=settings.augmentation_noise_sigma,
    )

    try:
        optimizer_name = OptimizerName(settings.optimizer_name)
    except ValueError:
        optimizer_name = OptimizerName.adam
    try:
        mixed_precision = MixedPrecisionMode(settings.mixed_precision)
    except ValueError:
        mixed_precision = MixedPrecisionMode.bf16

    training = TrainingParameters(
        image_path=Path(main_window.sample_folder),
        label_path=Path(main_window.label_folder),
        shuffle=settings.shuffle,
        validation=settings.use_validation,
        validation_percent=settings.validation_percent,
        batch_size=settings.batch_size,
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
        ),
        skip_uniform_labels=settings.skip_uniform_labels,
        use_multi_gpu=settings.use_multi_gpu,
        show_batch_preview=settings.show_batch_preview,
        log_update_frequency=settings.log_update_frequency,
    )

    recognition = RecognitionParameters(
        source_files=filter_images(Path(main_window.source_folder)),
        result_folder=Path(main_window.result_folder),
        model=model,
        batch_size=settings.batch_size,
        part_size=settings.sample_size,
        overlap=settings.overlap,
    )

    return work_mode, training, recognition
