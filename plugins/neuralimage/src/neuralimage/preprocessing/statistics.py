from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from neuralimage.preprocessing.pipeline import to_float01


MIN_DATASET_STD = 1e-6


@dataclass(frozen=True)
class DatasetStatistics:
    mean: float
    std: float
    pixel_count: int


def compute_dataset_statistics(images: Iterable[np.ndarray]) -> DatasetStatistics:
    """Compute population mean/std over training pixels using stable batches."""
    count = 0
    mean = 0.0
    m2 = 0.0
    for image in images:
        values = to_float01(np.asarray(image)).astype(np.float64, copy=False)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        batch_count = int(finite.size)
        batch_mean = float(finite.mean(dtype=np.float64))
        batch_m2 = float(np.square(finite - batch_mean).sum(dtype=np.float64))
        if count == 0:
            count = batch_count
            mean = batch_mean
            m2 = batch_m2
            continue
        combined = count + batch_count
        delta = batch_mean - mean
        mean += delta * batch_count / combined
        m2 += batch_m2 + delta * delta * count * batch_count / combined
        count = combined
    if count == 0:
        raise ValueError('Cannot calculate dataset statistics from an empty training dataset.')
    std = float(np.sqrt(max(m2 / count, 0.0)))
    if std < MIN_DATASET_STD:
        std = 1.0
    return DatasetStatistics(mean=float(mean), std=std, pixel_count=count)
