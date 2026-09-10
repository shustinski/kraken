"""Numeric conversions for untyped settings and signal payloads.

Unsupported inputs raise TypeError, just as the corresponding built-in does.
No defaults or lossy string round-trips are introduced here.
"""

from collections.abc import Buffer
from typing import SupportsFloat, SupportsIndex, SupportsInt


def to_int(value: object) -> int:
    if not isinstance(value, (str, Buffer, SupportsInt, SupportsIndex)):
        raise TypeError(f"Cannot convert {type(value).__name__} to int")
    return int(value)


def to_float(value: object) -> float:
    if not isinstance(value, (str, Buffer, SupportsFloat, SupportsIndex)):
        raise TypeError(f"Cannot convert {type(value).__name__} to float")
    return float(value)
