import numpy as np

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
