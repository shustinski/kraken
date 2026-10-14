"""model_train_and_recognition: validation no cut frame."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import torch

@dataclass
class _ValidationNoCutFrameExportCache:
    frame_index: int
    image_path: Path
    baseim_size: tuple[int, int]
    overlap: int
    parts_count: int
    part_lookup: list[int] | None
    patches: dict[int, np.ndarray]
    label_path: Path | None = None
    confidence_patches: dict[int, np.ndarray] = field(default_factory=dict)

@dataclass
class _ValidationExportCache:
    mode: str
    dataset: Any
    frame_predictions: dict[int, _ValidationNoCutFrameExportCache] | None = None

def _release_torch_memory() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        # Best-effort release; ignore cleanup failures.
        pass
