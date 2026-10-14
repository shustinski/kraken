"""dataset: apply binary tech augmentation."""
from __future__ import annotations
import random
import numpy as np
from neuralimage.augmentations import (
    TechVariationAugmentor,
)

from neuralimage.model.NeuralNetwork.dataset_parts.extract_binary_single_channel import (
    _broadcast_augmented_mask,
    _extract_binary_single_channel,
    _is_binary_like_plane,
    _unwrap_tech_augmented_mask,
)

def _apply_binary_tech_augmentation(
    image: np.ndarray,
    augmentor: TechVariationAugmentor | None,
    *,
    binary_tolerance: float,
) -> np.ndarray:
    if augmentor is None:
        return image.astype(np.float32, copy=False)
    if image.ndim != 3 or image.shape[0] <= 0:
        return image.astype(np.float32, copy=False)

    single_channel = image[:1]
    if image.shape[0] > 1 and not np.allclose(image, np.repeat(single_channel, image.shape[0], axis=0), atol=1e-6):
        return image.astype(np.float32, copy=False)
    if not _is_binary_like_plane(single_channel[0], tolerance=binary_tolerance):
        return image.astype(np.float32, copy=False)

    augmented = _unwrap_tech_augmented_mask(augmentor(single_channel.astype(np.float32, copy=False)))
    if augmented.ndim == 2:
        augmented = augmented[None, :, :]
    augmented = augmented.astype(np.float32, copy=False)
    if image.shape[0] == 1:
        return augmented
    return np.repeat(augmented[:1], image.shape[0], axis=0).astype(np.float32, copy=False)

def _apply_binary_tech_augmentation_to_pair(
    image: np.ndarray,
    label: np.ndarray,
    augmentor: TechVariationAugmentor | None,
    *,
    binary_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    image_array = np.asarray(image, dtype=np.float32)
    label_array = np.asarray(label, dtype=np.float32)
    if augmentor is None:
        return image_array.astype(np.float32, copy=False), label_array.astype(np.float32, copy=False)

    source_mask = _extract_binary_single_channel(label_array, binary_tolerance=binary_tolerance)
    if source_mask is None:
        source_mask = _extract_binary_single_channel(image_array, binary_tolerance=binary_tolerance)
    if source_mask is None:
        return image_array.astype(np.float32, copy=False), label_array.astype(np.float32, copy=False)

    augmented_mask = _unwrap_tech_augmented_mask(augmentor(source_mask.astype(np.float32, copy=False)))
    augmented_label = _broadcast_augmented_mask(label_array, augmented_mask)
    if _extract_binary_single_channel(image_array, binary_tolerance=binary_tolerance) is None:
        return image_array.astype(np.float32, copy=False), augmented_label
    augmented_image = _broadcast_augmented_mask(image_array, augmented_mask)
    return augmented_image, augmented_label

def _apply_binary_tech_augmentation_with_seed(
    image: np.ndarray,
    augmentor: TechVariationAugmentor | None,
    *,
    binary_tolerance: float,
    seed: int,
) -> np.ndarray:
    if augmentor is None:
        return image.astype(np.float32, copy=False)
    random_state = random.getstate()
    np_random_state = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        return _apply_binary_tech_augmentation(
            image,
            augmentor,
            binary_tolerance=binary_tolerance,
        )
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)

def _apply_binary_tech_augmentation_to_pair_with_seed(
    image: np.ndarray,
    label: np.ndarray,
    augmentor: TechVariationAugmentor | None,
    *,
    binary_tolerance: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if augmentor is None:
        return (
            np.asarray(image, dtype=np.float32).astype(np.float32, copy=False),
            np.asarray(label, dtype=np.float32).astype(np.float32, copy=False),
        )
    random_state = random.getstate()
    np_random_state = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        return _apply_binary_tech_augmentation_to_pair(
            image,
            label,
            augmentor,
            binary_tolerance=binary_tolerance,
        )
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)
