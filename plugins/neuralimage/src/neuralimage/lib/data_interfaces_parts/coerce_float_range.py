"""data_interfaces: coerce float range."""
from __future__ import annotations
from typing import Any

from neuralimage.lib.data_interfaces_parts.ic_defect_parameters import _coerce_range

def _coerce_float_range(
    value: Any,
    default: tuple[float, float],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, float]:
    resolved = _coerce_range(value, default, cast_type=float)
    low = float(resolved[0])
    high = float(resolved[1])
    if minimum is not None:
        low = max(float(minimum), low)
        high = max(float(minimum), high)
    if maximum is not None:
        low = min(float(maximum), low)
        high = min(float(maximum), high)
    if low > high:
        low, high = high, low
    return low, high

def _coerce_int_range(
    value: Any,
    default: tuple[int, int],
    *,
    minimum: int = 0,
) -> tuple[int, int]:
    resolved = _coerce_range(value, default, cast_type=int)
    low = max(int(minimum), int(resolved[0]))
    high = max(int(minimum), int(resolved[1]))
    if low > high:
        low, high = high, low
    return low, high
