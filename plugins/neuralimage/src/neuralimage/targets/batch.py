from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def extract_mask_from_target(target: Any) -> torch.Tensor | np.ndarray:
    if isinstance(target, Mapping):
        mask = target.get('mask')
        if mask is None:
            raise KeyError('Supervision target mapping must contain a "mask" entry.')
        return mask
    return target


def collate_supervision_targets(batch_targets: list[Any]) -> Any:
    if not batch_targets:
        return batch_targets
    first = batch_targets[0]
    if not isinstance(first, Mapping):
        return torch.utils.data.default_collate(batch_targets)

    collated: dict[str, torch.Tensor] = {}
    for key in first:
        values = [item[key] for item in batch_targets]
        first_value = values[0]
        if isinstance(first_value, np.ndarray):
            arrays = [np.asarray(value, dtype=np.float32) for value in values]
            collated[str(key)] = torch.from_numpy(np.stack(arrays, axis=0))
        elif torch.is_tensor(first_value):
            collated[str(key)] = torch.stack(values, dim=0)
        else:
            collated[str(key)] = torch.utils.data.default_collate(values)
    return collated
