from pathlib import Path

from lib.data_interfaces import (
    WorkMode,
    SampleCutMode,
    OptimizerName,
    OptimizerParameters,
    SampleGenerationSettings,
    SamplePrepareSettings,
    TrainingParameters,
    RecognitionParameters,
)


def test_work_mode_values():
    assert WorkMode.train_only.value == 'train_only'
    assert WorkMode.recognition_only.value == 'recognintion_only'


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
    assert rec.source_files == [Path('x')]


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
    assert train.mixed_precision.value == 'bf16'
    assert train.warmup.enabled is False
    assert train.early_stopping.enabled is False

