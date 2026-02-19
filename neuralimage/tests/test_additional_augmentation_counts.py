import random

import numpy as np

from lib.data_interfaces import CutSettings, SampleGenerationSettings
from lib.images import SampleCalculator, SampleFastCutter, SampleWorker


def test_sample_calculator_doubles_parts_when_additional_augmentation_enabled():
    base_params = SampleGenerationSettings(
        step=16,
        segment_size=(16, 16),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
        additional_augmentation=False,
    )
    augmented_params = SampleGenerationSettings(
        step=16,
        segment_size=(16, 16),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
        additional_augmentation=True,
    )

    base_count = len(SampleCalculator((32, 32), base_params))
    augmented_count = len(SampleCalculator((32, 32), augmented_params))

    assert base_count == 4
    assert augmented_count == 8


def test_sample_worker_count_doubles_when_additional_augmentation_enabled():
    base_settings = CutSettings(
        vertical_rotation=False,
        horizontal_rotation=False,
        step=16,
        color_mode='RGB',
        x_size=16,
        y_size=16,
        model='M 720k',
        additional_augmentation=False,
    )
    augmented_settings = CutSettings(
        vertical_rotation=False,
        horizontal_rotation=False,
        step=16,
        color_mode='RGB',
        x_size=16,
        y_size=16,
        model='M 720k',
        additional_augmentation=True,
    )

    base_worker = SampleWorker(paramns=base_settings)
    augmented_worker = SampleWorker(paramns=augmented_settings)

    base_count = base_worker.calculate_image_parts((32, 32))
    augmented_count = augmented_worker.calculate_image_parts((32, 32))

    assert base_count == 4
    assert augmented_count == 8


def test_sample_fast_cutter_returns_original_and_augmented_pair():
    random.seed(7)
    np.random.seed(7)

    image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(1, 32, 32)
    label = (image > 0.5).astype(np.float32)
    params = SampleGenerationSettings(
        step=16,
        segment_size=(16, 16),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
        additional_augmentation=True,
        augmentation_brightness_strength=0.4,
        augmentation_contrast_strength=0.4,
        augmentation_noise_probability=0.0,
        augmentation_noise_sigma=0.0,
    )

    cutter = SampleFastCutter((image, label), params, shuffle=False)

    original_image, original_label = cutter[0]
    augmented_image, augmented_label = cutter[1]

    assert len(cutter) == 8
    assert np.array_equal(original_label, augmented_label)
    assert not np.allclose(original_image, augmented_image)
