from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from neuralimage.augmentations.sem import SemAugmentor
from neuralimage.augmentations.sem_config import SemAugmentationConfig
from neuralimage.preprocessing.config import PreprocessingConfig
from neuralimage.preprocessing.pipeline import SemPreprocessingPipeline
from neuralimage.targets.config import SupervisionTargetsParameters
from neuralimage.targets.registry import generate_supervision_targets


def _to_numpy_label(label: Any) -> np.ndarray:
    if torch.is_tensor(label):
        array = label.detach().cpu().numpy()
    else:
        array = np.asarray(label)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    return array.astype(np.float32)


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 2:
        values = values[None, ...]
    return torch.from_numpy(values)


def apply_dataset_preprocessing(image: Any, config: PreprocessingConfig | None) -> Any:
    if config is None or not config.any_enabled():
        return image
    pipeline = SemPreprocessingPipeline(config)
    if torch.is_tensor(image):
        array = image.detach().cpu().numpy()
        if array.ndim == 3:
            processed_planes = [pipeline.apply(plane) for plane in array]
            processed = np.stack(processed_planes, axis=0)
        else:
            processed = pipeline.apply(array)
        return torch.from_numpy(processed.astype(np.float32))
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in {1, 3}:
        processed = np.stack([pipeline.apply(plane) for plane in array], axis=0)
    else:
        processed = pipeline.apply(array)
    return processed.astype(np.float32)


def apply_dataset_sem_augmentation(
    image: Any,
    label: Any,
    config: SemAugmentationConfig | None,
) -> tuple[Any, Any]:
    if config is None or not config.enabled:
        return image, label
    augmentor = SemAugmentor(config)
    if torch.is_tensor(image):
        array = image.detach().cpu().numpy()
        label_array = _to_numpy_label(label)
        if array.ndim == 3:
            augmented_planes = []
            for plane in array:
                augmented_plane, _ = augmentor.apply(plane, label_array)
                augmented_planes.append(augmented_plane)
            augmented = np.stack(augmented_planes, axis=0)
        else:
            augmented, _ = augmentor.apply(array, label_array)
        return torch.from_numpy(augmented.astype(np.float32)), label
    array = np.asarray(image)
    label_array = _to_numpy_label(label)
    if array.ndim == 3 and array.shape[0] in {1, 3}:
        augmented = np.stack(
            [augmentor.apply(plane, label_array)[0] for plane in array],
            axis=0,
        )
    else:
        augmented, _ = augmentor.apply(array, label_array)
    return augmented.astype(np.float32), label


def maybe_build_supervision_target(
    label: Any,
    config: SupervisionTargetsParameters | None,
) -> Any:
    if config is None or not config.any_enabled():
        return label
    label_array = _to_numpy_label(label)
    targets = generate_supervision_targets(
        label_array,
        basic_config=config.basic,
        geometry_config=config.geometry,
        cache=bool(getattr(config, 'cache_enabled', False)),
        cache_size=int(getattr(config, 'cache_size', 256)),
    )
    tensor_targets: dict[str, torch.Tensor] = {}
    for name, array in targets.items():
        values = np.asarray(array, dtype=np.float32)
        if values.ndim == 2:
            values = values[None, ...]
        elif values.ndim == 3 and values.shape[-1] in {2, 3}:
            values = np.transpose(values, (2, 0, 1))
        tensor_targets[name] = torch.from_numpy(values)
    return tensor_targets


def extract_primary_label(target: Any) -> torch.Tensor:
    if isinstance(target, Mapping):
        mask = target.get('mask')
        if mask is None:
            raise KeyError('Supervision target mapping must include "mask".')
        return mask if torch.is_tensor(mask) else torch.from_numpy(np.asarray(mask, dtype=np.float32))
    return target if torch.is_tensor(target) else torch.from_numpy(np.asarray(target, dtype=np.float32))
