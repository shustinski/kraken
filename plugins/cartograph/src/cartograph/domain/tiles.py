"""Tile and grid models. Image payloads are grayscale buffers, never OpenCV types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .coordinates import GridCoordinate, StageCoordinate
from .errors import GridLoadError


GrayImage = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class ImageBuffer:
    """Domain grayscale image. Values are typically 0..255 as float32."""

    pixels: GrayImage

    def __post_init__(self) -> None:
        array = np.asarray(self.pixels)
        if array.ndim != 2:
            raise ValueError("ImageBuffer must be a 2D grayscale array")
        if array.size == 0:
            raise ValueError("ImageBuffer must not be empty")
        object.__setattr__(self, "pixels", np.ascontiguousarray(array, dtype=np.float32))

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def std(self) -> float:
        return float(self.pixels.std())

    @property
    def mean(self) -> float:
        return float(self.pixels.mean())


@dataclass(frozen=True, slots=True)
class Tile:
    """One SEM frame. Grid and stage coordinates are stored independently."""

    coord: GridCoordinate
    source_id: str
    width: int
    height: int
    path: Path | None = None
    stage: StageCoordinate | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("tile width and height must be positive")
        object.__setattr__(self, "source_id", str(self.source_id))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class TileGrid:
    """Sparse logical grid of SEM tiles. Missing cells are allowed."""

    tiles: Mapping[GridCoordinate, Tile]
    overlap_x: float
    overlap_y: float
    name: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.overlap_x < 1.0 or not 0.0 <= self.overlap_y < 1.0:
            raise GridLoadError("overlap fractions must be in [0, 1)")
        if not self.tiles:
            raise GridLoadError("tile grid is empty")
        object.__setattr__(self, "tiles", dict(self.tiles))

    def get(self, coord: GridCoordinate) -> Tile | None:
        return self.tiles.get(coord)

    def require(self, coord: GridCoordinate) -> Tile:
        tile = self.tiles.get(coord)
        if tile is None:
            raise GridLoadError(f"no tile at row={coord.row} col={coord.col}")
        return tile

    @property
    def row_count(self) -> int:
        return max(tile.coord.row for tile in self.tiles.values()) + 1

    @property
    def col_count(self) -> int:
        return max(tile.coord.col for tile in self.tiles.values()) + 1

    @property
    def typical_width(self) -> int:
        return next(iter(self.tiles.values())).width

    @property
    def typical_height(self) -> int:
        return next(iter(self.tiles.values())).height

    def contains(self, coord: GridCoordinate) -> bool:
        return coord in self.tiles
