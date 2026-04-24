import random

import numpy as np

from neuralimage.lib.data_interfaces import CutSettings, SampleGenerationSettings
from neuralimage.lib.images import SampleCalculator, SampleFastCutter, SampleWorker


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


def test_sample_calculator_doubles_parts_when_scale_augmentation_enabled():
    base_params = SampleGenerationSettings(
        step=16,
        segment_size=(16, 16),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
        scale_augmentation=False,
    )
    scaled_params = SampleGenerationSettings(
        step=16,
        segment_size=(16, 16),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
        scale_augmentation=True,
        scale_augmentation_strength=0.4,
    )

    base_count = len(SampleCalculator((32, 32), base_params))
    scaled_count = len(SampleCalculator((32, 32), scaled_params))

    assert base_count == 4
    assert scaled_count == 8


def test_sample_worker_count_doubles_when_scale_augmentation_enabled():
    base_settings = CutSettings(
        vertical_rotation=False,
        horizontal_rotation=False,
        step=16,
        color_mode='RGB',
        x_size=16,
        y_size=16,
        model='M 720k',
        scale_augmentation=False,
    )
    scaled_settings = CutSettings(
        vertical_rotation=False,
        horizontal_rotation=False,
        step=16,
        color_mode='RGB',
        x_size=16,
        y_size=16,
        model='M 720k',
        scale_augmentation=True,
        scale_augmentation_strength=0.4,
    )

    base_worker = SampleWorker(paramns=base_settings)
    scaled_worker = SampleWorker(paramns=scaled_settings)

    base_count = base_worker.calculate_image_parts((32, 32))
    scaled_count = scaled_worker.calculate_image_parts((32, 32))

    assert base_count == 4
    assert scaled_count == 8


def test_sample_worker_random_crop_uses_crops_per_image_instead_of_step():
    small_step_settings = CutSettings(
        vertical_rotation=False,
        horizontal_rotation=False,
        step=8,
        color_mode='RGB',
        x_size=16,
        y_size=16,
        model='M 720k',
        random_crop=True,
        crops_per_image=7,
    )
    large_step_settings = CutSettings(
        vertical_rotation=False,
        horizontal_rotation=False,
        step=64,
        color_mode='RGB',
        x_size=16,
        y_size=16,
        model='M 720k',
        random_crop=True,
        crops_per_image=7,
    )

    small_step_worker = SampleWorker(paramns=small_step_settings)
    large_step_worker = SampleWorker(paramns=large_step_settings)

    assert small_step_worker.calculate_image_parts((64, 64)) == 7
    assert large_step_worker.calculate_image_parts((64, 64)) == 7


def test_sample_worker_path_and_settings_updates_are_lazy(monkeypatch, tmp_path):
    worker = SampleWorker()

    def _unexpected_scan(*_args, **_kwargs):
        raise AssertionError('set_path/set_settings must not scan files synchronously')

    monkeypatch.setattr('lib.images.filter_files', _unexpected_scan)

    worker.set_path(tmp_path)
    worker.set_settings(
        CutSettings(
            vertical_rotation=False,
            horizontal_rotation=False,
            step=16,
            color_mode='RGB',
            x_size=16,
            y_size=16,
            model='M 720k',
        )
    )


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


def test_sample_fast_cutter_returns_original_and_scaled_pair(monkeypatch):
    monkeypatch.setattr(random, 'uniform', lambda _min, _max: 0.5)

    image = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
    label = (image > 16).astype(np.float32)
    params = SampleGenerationSettings(
        step=4,
        segment_size=(4, 4),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
        scale_augmentation=True,
        scale_augmentation_strength=0.8,
    )

    cutter = SampleFastCutter((image, label), params, shuffle=False)

    original_image, original_label = cutter[0]
    scaled_image, scaled_label = cutter[1]

    assert len(cutter) == 8
    assert original_image.shape == (1, 4, 4)
    assert scaled_image.shape == (1, 4, 4)
    assert original_label.shape == (1, 4, 4)
    assert scaled_label.shape == (1, 4, 4)
    assert not np.allclose(original_image, scaled_image)


def test_sample_fast_cutter_random_crop_differs_from_fixed_grid():
    random.seed(7)

    image = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    label = image.copy()
    grid_params = SampleGenerationSettings(
        step=2,
        segment_size=(2, 2),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
        random_crop=False,
    )
    random_params = SampleGenerationSettings(
        step=2,
        segment_size=(2, 2),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
        random_crop=True,
        crops_per_image=4,
    )

    grid_cutter = SampleFastCutter((image, label), grid_params, shuffle=False)
    random_cutter = SampleFastCutter((image, label), random_params, shuffle=False)

    grid_image, grid_label = grid_cutter[0]
    random_image, random_label = random_cutter[0]

    assert np.array_equal(grid_image, grid_label)
    assert np.array_equal(random_image, random_label)
    assert not np.array_equal(random_image, grid_image)


def test_sample_fast_cutter_random_crop_reuses_same_base_patch_for_augmented_pair():
    random.seed(7)
    np.random.seed(7)

    image = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
    label = image.copy()
    params = SampleGenerationSettings(
        step=4,
        segment_size=(4, 4),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
        additional_augmentation=True,
        augmentation_brightness_strength=0.4,
        augmentation_contrast_strength=0.4,
        augmentation_noise_probability=0.0,
        augmentation_noise_sigma=0.0,
        random_crop=True,
        crops_per_image=4,
    )

    cutter = SampleFastCutter((image, label), params, shuffle=False)

    original_image, original_label = cutter[0]
    augmented_image, augmented_label = cutter[1]

    assert np.array_equal(original_label, augmented_label)
    assert original_image.shape == (1, 4, 4)
    assert augmented_image.shape == (1, 4, 4)
    assert not np.allclose(original_image, augmented_image)


def test_sample_fast_cutter_skips_uniform_label_locations():
    image = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    label = np.zeros((1, 4, 4), dtype=np.float32)
    label[:, 2:, 2:] = 1.0
    label[:, 0:2, 2:] = 1.0
    label[:, 2, 1] = 1.0
    params = SampleGenerationSettings(
        step=2,
        segment_size=(2, 2),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
    )

    cutter = SampleFastCutter((image, label), params, shuffle=False, skip_uniform_labels=True)

    assert len(cutter) == 1
    _, filtered_label = cutter[0]
    assert np.any(filtered_label > 0.5)
    assert np.any(filtered_label <= 0.5)


def test_sample_calculator_random_crop_uses_crops_per_image_instead_of_step():
    small_step_params = SampleGenerationSettings(
        step=8,
        segment_size=(16, 16),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
        random_crop=True,
        crops_per_image=9,
    )
    large_step_params = SampleGenerationSettings(
        step=64,
        segment_size=(16, 16),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
        random_crop=True,
        crops_per_image=9,
    )

    assert len(SampleCalculator((64, 64), small_step_params)) == 9
    assert len(SampleCalculator((64, 64), large_step_params)) == 9
