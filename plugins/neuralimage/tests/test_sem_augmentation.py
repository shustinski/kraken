import numpy as np
import random

from neuralimage.augmentations.sem import SemAugmentor
from neuralimage.augmentations.sem_config import SemAugmentationConfig


def test_sem_augmentor_disabled_is_identity():
    image = np.full((32, 32), 128, dtype=np.uint8)
    augmentor = SemAugmentor(SemAugmentationConfig(enabled=False))
    augmented, label = augmentor.apply(image, np.ones((32, 32), dtype=np.float32))
    assert np.array_equal(augmented, image.astype(np.float32) / 255.0)


def test_sem_augmentor_can_modify_image_when_enabled():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(32, 32), dtype=np.uint8)
    config = SemAugmentationConfig(
        enabled=True,
        charging_artifacts=True,
        charging_probability=1.0,
        detector_noise=True,
        detector_noise_probability=1.0,
    )
    augmentor = SemAugmentor(config)
    augmented, _ = augmentor.apply(image, np.zeros((32, 32), dtype=np.float32))
    assert not np.array_equal(augmented, image.astype(np.float32) / 255.0)


def test_sem_v2_is_seed_deterministic_bounded_and_does_not_change_label():
    image = np.linspace(0.0, 1.0, 48 * 48, dtype=np.float32).reshape(48, 48)
    label = (image > 0.5).astype(np.float32)
    config = SemAugmentationConfig(
        enabled=True,
        plan='sem_v2',
        charging_probability=1.0,
        scan_drift_probability=1.0,
        focus_variation_probability=1.0,
        detector_noise_probability=1.0,
        brightness_gradient_probability=1.0,
        realistic_defect_probability=1.0,
    )
    random.seed(7)
    np.random.seed(7)
    first, first_label = SemAugmentor(config).apply(image, label)
    random.seed(7)
    np.random.seed(7)
    second, second_label = SemAugmentor(config).apply(image, label)
    assert np.array_equal(first, second)
    assert np.array_equal(first_label, label)
    assert np.array_equal(second_label, label)
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0
