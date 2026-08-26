"""Kraken frame-coordinate adapter. Cartograph stays 0-based (row, col)."""

from __future__ import annotations

from cartograph.domain.coordinates import GridCoordinate


def to_kraken_xy(coord: GridCoordinate) -> tuple[int, int]:
    """Return 1-based Kraken ``(x, y)``."""

    return coord.to_kraken_xy()


def from_kraken_xy(x: int, y: int) -> GridCoordinate:
    return GridCoordinate.from_kraken_xy(x, y)
