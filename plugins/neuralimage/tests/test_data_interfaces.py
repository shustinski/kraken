from pathlib import Path

from neuralimage.lib.data_interfaces import (
    PCBDefectParameters,
    WorkMode,
    ValidationSource,
    normalize_work_mode,
    normalize_validation_source,
    parse_work_mode,
    SampleCutMode,
    OptimizerName,
    OptimizerParameters,
    SampleGenerationSettings,
    SamplePrepareSettings,
    SchedulerName,
    TrainingParameters,
    RecognitionParameters,
)


def test_work_mode_values():
    assert WorkMode.train_only.value == 'train_only'
    assert WorkMode.recognition_only.value == 'recognition_only'
    assert WorkMode.further_training.value == 'further_training'


def test_work_mode_legacy_aliases_are_normalized():
    assert normalize_work_mode('recognintion_only') == WorkMode.recognition_only.value
    assert normalize_work_mode('futher_training') == WorkMode.further_training.value
    assert parse_work_mode('recognintion_only') == WorkMode.recognition_only
    assert parse_work_mode('futher_training') == WorkMode.further_training


def test_validation_source_defaults_to_split():
    assert normalize_validation_source('external') == ValidationSource.external.value
    assert normalize_validation_source(None) == ValidationSource.split.value


def test_dataclass_instantiation():
    gen = SampleGenerationSettings(1, (16, 16), True, False, 1)
    prep = SamplePrepareSettings()
    train = TrainingParameters(
        image_path=Path('a'),
        label_path=Path('b'),
        shuffle=True,
        validation=False,
        validation_percent=20,
        batch_size=4,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=gen,
        prepare=prep,
    )
    rec = RecognitionParameters(
        source_files=[Path('x')],
        result_folder=Path('y'),
        model='m',
        part_size=(16, 16),
        batch_size=2,
        overlap=0,
    )

    assert train.generation.segment_size == (16, 16)
    assert train.validation_source == ValidationSource.split.value
    assert train.validation_image_path is None
    assert train.validation_label_path is None
    assert train.save_validation_binary_images is False
    assert train.dataloader_num_workers == -1
    assert train.deep_supervision is False
    assert train.generation.tech_aug.enabled is False
    assert rec.source_files == [Path('x')]
    assert rec.recognition_multiprocessing_enabled is True
    assert rec.recognition_tta_enabled is False
    assert rec.confidence_tta_enabled is False
    assert rec.confidence_save_mode == 'off'


def test_training_parameters_default_optimizer():
    gen = SampleGenerationSettings(1, (16, 16), True, False, 1)
    prep = SamplePrepareSettings()
    train = TrainingParameters(
        image_path=Path('a'),
        label_path=Path('b'),
        shuffle=True,
        validation=False,
        validation_percent=20,
        batch_size=4,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=gen,
        prepare=prep,
    )
    assert train.optimizer.name == OptimizerName.adam
    assert train.optimizer.learning_rate == 1e-3
    assert train.optimizer.weight_decay == 0.0
    assert train.mixed_precision.value == 'fp16'
    assert train.warmup.enabled is False
    assert train.scheduler.name == SchedulerName.off
    assert train.scheduler.plateau_factor == 0.5
    assert train.scheduler.cosine_t_max == 10
    assert train.scheduler.one_cycle_anneal_strategy == 'cos'
    assert train.scheduler.step_lr_gamma == 0.1
    assert train.early_stopping.enabled is False
    assert train.cutout.enabled is False
    assert train.cutout.probability == 1.0
    assert train.cutout.holes == 1
    assert train.cutout.size_ratio == 0.25
    assert train.random_artifacts.enabled is False
    assert train.random_artifacts.probability == 1.0
    assert train.random_artifacts.count == 1
    assert train.random_artifacts.size_ratio == 0.25
    assert train.mixup.enabled is False
    assert train.mixup.probability == 1.0
    assert train.mixup.alpha == 0.2
    assert train.pcb_defects.enabled is False
    assert train.pcb_defects.use_defect_mask_as_label is False


def test_pcb_defect_parameters_defaults():
    params = PCBDefectParameters()

    assert params.enabled is False
    assert params.defect_probability == 0.5
    assert params.min_defects == 1
    assert params.max_defects == 3
    assert params.use_input_mask is True
    assert params.use_defect_mask_as_label is False
    assert set(params.defect_probabilities) == {
        'break',
        'short',
        'missing_copper',
        'excess_copper',
        'pinhole',
        'spurious_copper',
        'via',
        'misalignment',
    }


def test_tech_augmentation_defaults_are_visibly_stronger():
    gen = SampleGenerationSettings(1, (16, 16), True, False, 1)

    assert gen.tech_aug.max_changed_pixels_ratio == 0.32
    assert gen.tech_aug.max_foreground_ratio_delta == 0.2
    assert gen.tech_aug.global_width.kernel_size_range == (2, 3)
    assert gen.tech_aug.scale_rethreshold.scale_range == (0.82, 1.18)
    assert gen.tech_aug.blur_threshold.blur_radius_range == (0.6, 1.8)
    assert gen.tech_aug.boundary_aware.band_width_range == (2, 4)
    assert gen.tech_aug.local_morphology.kernel_size_range == (2, 3)
    assert gen.tech_aug.gap_variation.kernel_size_range == (2, 3)

