"""general_neural_handler: estimate label foreground ratio."""
from __future__ import annotations
import sys
import hashlib
from pathlib import Path
from neuralimage.lib.images import open_image
from kraken_core.asset_source import AssetSource

from neuralimage.model.general_neural_handler_parts.module_state import (
    _VALIDATION_FOREGROUND_BUCKETS,
    _VALIDATION_SPLIT_SEED,
)

def _validation_grid_step(segment_size: tuple[int, int]) -> int:
    """Choose a deterministic covering stride without dense training overlap."""
    return max(1, min(int(segment_size[0]), int(segment_size[1])))

def _is_debugger_attached() -> bool:
    gettrace = getattr(sys, 'gettrace', None)
    if not callable(gettrace):
        return False
    try:
        return bool(gettrace())
    except Exception:
        return False

def _stable_sample_sort_key(sample: tuple[Path, Path], *, seed: int = _VALIDATION_SPLIT_SEED) -> str:
    image_path, label_path = sample
    payload = f'{seed}:{image_path.stem}:{label_path.stem}'.encode('utf-8', errors='ignore')
    return hashlib.sha1(payload).hexdigest()

def _estimate_label_foreground_ratio(label_path: Path, asset_source: AssetSource | None = None) -> float:
    with open_image(label_path, asset_source) as image:
        grayscale = image.convert('L')
        histogram = grayscale.histogram()
    if not histogram:
        return 0.0
    total_pixels = int(sum(histogram))
    if total_pixels <= 0:
        return 0.0
    foreground_pixels = int(sum(histogram[1:]))
    return min(max(float(foreground_pixels) / float(total_pixels), 0.0), 1.0)

def _label_ratio_bucket(ratio: float) -> int:
    normalized = min(max(float(ratio), 0.0), 1.0)
    for bucket_index, upper_bound in enumerate(_VALIDATION_FOREGROUND_BUCKETS):
        if normalized <= upper_bound:
            return bucket_index
    return len(_VALIDATION_FOREGROUND_BUCKETS)
