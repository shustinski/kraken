"""Independent coordinate systems used by Cartograph.

Kraken project frames are 1-based ``(x=column, y=row)``. Cartograph stores a
0-based ``GridCoordinate(row, col)`` and converts at the Kraken adapter.
Stage coordinates are a prior, not metric ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .errors import PlacementError


class NominalPlacementMode(StrEnum):
    SEM_STAGE = "sem_stage"
    REGULAR_GRID = "regular_grid"
    HYBRID = "hybrid"


class TransformKind(StrEnum):
    TRANSLATION = "translation"
    RIGID = "rigid"
    SIMILARITY = "similarity"
    AFFINE = "affine"


@dataclass(frozen=True, slots=True, order=True)
class GridCoordinate:
    """Logical tile index in the SEM frame array (0-based)."""

    row: int
    col: int

    def __post_init__(self) -> None:
        if isinstance(self.row, bool) or isinstance(self.col, bool):
            raise ValueError("grid row and col must be integers")
        if self.row < 0 or self.col < 0:
            raise ValueError("grid row and col must be >= 0")

    def offset(self, d_row: int, d_col: int) -> GridCoordinate:
        return GridCoordinate(self.row + d_row, self.col + d_col)

    def to_kraken_xy(self) -> tuple[int, int]:
        """Return Kraken 1-based ``(x, y)`` where x is column and y is row."""

        return self.col + 1, self.row + 1

    @classmethod
    def from_kraken_xy(cls, x: int, y: int) -> GridCoordinate:
        if x < 1 or y < 1:
            raise ValueError("Kraken frame coordinates are 1-based")
        return cls(row=y - 1, col=x - 1)


@dataclass(frozen=True, slots=True)
class StageCoordinate:
    """Physical SEM stage position in instrument units (typically micrometres)."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class NominalCoordinate:
    """Approximate pixel position used for navigation, not high-precision stitching."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class PixelSize:
    """Stage units per pixel. X and Y may differ for a poorly calibrated SEM."""

    x: float
    y: float

    def __post_init__(self) -> None:
        if self.x <= 0.0 or self.y <= 0.0:
            raise PlacementError("pixel size must be positive")


@dataclass(frozen=True, slots=True)
class Translation2D:
    dx: float
    dy: float

    @property
    def magnitude(self) -> float:
        return (self.dx * self.dx + self.dy * self.dy) ** 0.5

    def plus(self, other: Translation2D) -> Translation2D:
        return Translation2D(self.dx + other.dx, self.dy + other.dy)

    def minus(self, other: Translation2D) -> Translation2D:
        return Translation2D(self.dx - other.dx, self.dy - other.dy)

    def scaled(self, factor: float) -> Translation2D:
        return Translation2D(self.dx * factor, self.dy * factor)

    def negated(self) -> Translation2D:
        return Translation2D(-self.dx, -self.dy)


@dataclass(frozen=True, slots=True)
class LocalTransform:
    """Precise local transform of a tile versus the block coordinate system or a neighbor."""

    kind: TransformKind
    dx: float
    dy: float

    def __post_init__(self) -> None:
        if self.kind is not TransformKind.TRANSLATION:
            raise ValueError(f"Cartograph v1 implements TRANSLATION only, got {self.kind}")

    @classmethod
    def translation(cls, dx: float, dy: float) -> LocalTransform:
        return cls(TransformKind.TRANSLATION, float(dx), float(dy))

    def as_translation(self) -> Translation2D:
        return Translation2D(self.dx, self.dy)


def regular_grid_position(
    coord: GridCoordinate,
    tile_width: int,
    tile_height: int,
    overlap_x: float,
    overlap_y: float,
) -> NominalCoordinate:
    """X_ij = j * W * (1 - o_x), Y_ij = i * H * (1 - o_y)."""

    if tile_width <= 0 or tile_height <= 0:
        raise PlacementError("tile dimensions must be positive")
    if not 0.0 <= overlap_x < 1.0 or not 0.0 <= overlap_y < 1.0:
        raise PlacementError("nominal overlap fractions must be in [0, 1)")
    stride_x = tile_width * (1.0 - overlap_x)
    stride_y = tile_height * (1.0 - overlap_y)
    return NominalCoordinate(x=coord.col * stride_x, y=coord.row * stride_y)


def stage_to_pixels(stage: StageCoordinate, pixel_size: PixelSize) -> NominalCoordinate:
    return NominalCoordinate(x=stage.x / pixel_size.x, y=stage.y / pixel_size.y)
