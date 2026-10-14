"""general_neural_handler: module state."""
from __future__ import annotations

_VALIDATION_SPLIT_SEED = 1337

_VALIDATION_FOREGROUND_BUCKETS: tuple[float, ...] = (0.0, 0.001, 0.01, 0.05, 0.2, 1.0)
