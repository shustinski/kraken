"""Compact per-cell containers for grid results.

A frame keeps thousands of cells. Tuples of point and feature pairs cost
about 100 bytes per item; these keep the same values packed in bytes and
still read as the same sequences of pairs.
"""

from __future__ import annotations

from array import array
from typing import Any, Iterable, Iterator

import numpy as np

_KEY_SETS: dict[tuple[str, ...], tuple[str, ...]] = {}


class PackedPoints:
    """Immutable sequence of ``(x, y)`` int pairs stored as int32 bytes."""

    __slots__ = ("_data",)

    def __init__(self, points: Iterable[Any] = ()) -> None:
        if isinstance(points, PackedPoints):
            data = points._data
        else:
            values = np.asarray(points if isinstance(points, np.ndarray) else list(points), dtype=np.int64)
            data = np.ascontiguousarray(values.reshape(-1, 2), dtype=np.int32).tobytes() if values.size else b""
        object.__setattr__(self, "_data", data)

    @classmethod
    def _from_bytes(cls, data: bytes) -> "PackedPoints":
        packed = cls.__new__(cls)
        object.__setattr__(packed, "_data", bytes(data))
        return packed

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("PackedPoints is immutable")

    def _array(self) -> np.ndarray:
        return np.frombuffer(self._data, dtype=np.int32).reshape(-1, 2)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        values = self._array()
        return values.astype(dtype) if dtype is not None else values.copy()

    def __len__(self) -> int:
        return len(self._data) // 8

    def __bool__(self) -> bool:
        return bool(self._data)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(tuple(map(tuple, self._array().tolist())))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return PackedPoints(self._array()[index])
        x, y = self._array()[index].tolist()
        return (x, y)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PackedPoints):
            return self._data == other._data
        if isinstance(other, (tuple, list)):
            return tuple(self) == tuple(tuple(item) for item in other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __reduce__(self):
        return (PackedPoints._from_bytes, (self._data,))

    def __copy__(self) -> "PackedPoints":
        return self

    def __deepcopy__(self, memo) -> "PackedPoints":
        return self

    def __repr__(self) -> str:
        return f"PackedPoints({tuple(self)!r})"


class PackedFeatures:
    """Immutable sequence of ``(name, value)`` pairs; names shared, values float64 bytes."""

    __slots__ = ("_keys", "_values")

    def __init__(self, pairs: Iterable[tuple[str, float]] = ()) -> None:
        if isinstance(pairs, PackedFeatures):
            keys, values = pairs._keys, pairs._values
        else:
            items = [(str(key), float(value)) for key, value in pairs]
            key_tuple = tuple(key for key, _value in items)
            keys = _KEY_SETS.setdefault(key_tuple, key_tuple)
            values = array("d", (value for _key, value in items)).tobytes()
        object.__setattr__(self, "_keys", keys)
        object.__setattr__(self, "_values", values)

    @classmethod
    def _from_parts(cls, keys: tuple[str, ...], values: bytes) -> "PackedFeatures":
        packed = cls.__new__(cls)
        key_tuple = tuple(keys)
        object.__setattr__(packed, "_keys", _KEY_SETS.setdefault(key_tuple, key_tuple))
        object.__setattr__(packed, "_values", bytes(values))
        return packed

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("PackedFeatures is immutable")

    def _floats(self) -> array:
        values = array("d")
        values.frombytes(self._values)
        return values

    def __len__(self) -> int:
        return len(self._keys)

    def __bool__(self) -> bool:
        return bool(self._keys)

    def __iter__(self) -> Iterator[tuple[str, float]]:
        return iter(tuple(zip(self._keys, self._floats())))

    def __getitem__(self, index):
        pairs = tuple(self)
        if isinstance(index, slice):
            return PackedFeatures(pairs[index])
        return pairs[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PackedFeatures):
            return self._keys == other._keys and self._values == other._values
        if isinstance(other, (tuple, list)):
            return tuple(self) == tuple(tuple(item) for item in other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __reduce__(self):
        return (PackedFeatures._from_parts, (self._keys, self._values))

    def __copy__(self) -> "PackedFeatures":
        return self

    def __deepcopy__(self, memo) -> "PackedFeatures":
        return self

    def __repr__(self) -> str:
        return f"PackedFeatures({tuple(self)!r})"


def pack_points(points: Any) -> Any:
    """Pack a non-empty point sequence; empty and already packed values pass through."""

    if isinstance(points, PackedPoints) or not points:
        return points
    return PackedPoints(points)


def pack_features(pairs: Any) -> Any:
    """Pack a non-empty feature-pair sequence; anything not name/number pairs stays as is."""

    if isinstance(pairs, PackedFeatures) or not pairs:
        return pairs
    try:
        return PackedFeatures(pairs)
    except (TypeError, ValueError):
        return pairs
