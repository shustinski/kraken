"""Shared contracts for interchangeable conductor segmentation backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class StrategyUnavailableError(RuntimeError):
    """A selected backend cannot run in the current installation."""


class StrategyConfigurationError(ValueError):
    """A selected backend cannot safely run with the supplied parameters."""


@dataclass(slots=True)
class StrategySegmentation:
    binary_mask: np.ndarray
    instance_labels: np.ndarray | None = None
    boundary_map: np.ndarray | None = None
    confidence_map: np.ndarray | None = None
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)
    debug_data: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)


__all__ = [
    "StrategyConfigurationError",
    "StrategySegmentation",
    "StrategyUnavailableError",
]
