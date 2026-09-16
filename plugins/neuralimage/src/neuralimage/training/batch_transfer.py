from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def move_batch_to_device(value: Any, device: torch.device, *, non_blocking: bool = True) -> Any:
    """Recursively move tensors while preserving mapping and sequence structure."""
    if torch.is_tensor(value):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return {
            str(key): move_batch_to_device(item, device, non_blocking=non_blocking)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(move_batch_to_device(item, device, non_blocking=non_blocking) for item in value)
    if isinstance(value, list):
        return [move_batch_to_device(item, device, non_blocking=non_blocking) for item in value]
    return value
