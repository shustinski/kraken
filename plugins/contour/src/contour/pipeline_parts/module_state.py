"""pipeline: module state."""
from __future__ import annotations
from collections.abc import Callable
from typing import Any
import numpy as np

OperationCallable = Callable[[np.ndarray, dict[str, Any]], np.ndarray]

OperationTimingCallback = Callable[[int, str, float], None]

_OPERATIONS: dict[str, OperationDescriptor] = {}

_COMBINED_PRESETS: dict[str, tuple[str, ...]] = {
    "default": ("scharr", "log", "phase_congruency"),
    "fine_detail": ("scharr", "laplacian"),
    "robust": ("scharr", "log", "phase_congruency", "ridge"),
    "illumination_invariant": ("phase_congruency", "log"),
    "all_classical": ("sobel", "scharr", "laplacian", "log", "auto_canny"),
}
