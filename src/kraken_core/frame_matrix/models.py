"""Domain-neutral models shared by frame-matrix views and data sources."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MatrixOrientation(StrEnum):
    Y_DOWN = "y_down"
    Y_UP = "y_up"


class LodBand(StrEnum):
    OVERVIEW = "overview"
    CELLS = "cells"
    PREVIEWS = "previews"
    DETAILS = "details"
    SUBCELLS = "subcells"


@dataclass(frozen=True, slots=True)
class MatrixBounds:
    """Inclusive, one-based matrix rectangle."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        values = tuple(int(value) for value in (self.x1, self.y1, self.x2, self.y2))
        if min(values) < 1:
            raise ValueError("matrix coordinates are one-based and must be positive")
        object.__setattr__(self, "x1", min(values[0], values[2]))
        object.__setattr__(self, "x2", max(values[0], values[2]))
        object.__setattr__(self, "y1", min(values[1], values[3]))
        object.__setattr__(self, "y2", max(values[1], values[3]))

    @property
    def width(self) -> int:
        return self.x2 - self.x1 + 1

    @property
    def height(self) -> int:
        return self.y2 - self.y1 + 1

    @property
    def frame_count(self) -> int:
        return self.width * self.height

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= int(x) <= self.x2 and self.y1 <= int(y) <= self.y2

    def intersects(self, other: MatrixBounds) -> bool:
        return not (
            self.x2 < other.x1
            or other.x2 < self.x1
            or self.y2 < other.y1
            or other.y2 < self.y1
        )

    def coordinates(self) -> Iterator[tuple[int, int]]:
        for y in range(self.y1, self.y2 + 1):
            for x in range(self.x1, self.x2 + 1):
                yield x, y


@dataclass(frozen=True, slots=True)
class MatrixSelection:
    rectangles: tuple[MatrixBounds, ...] = ()
    keys: frozenset[str] = frozenset()

    @classmethod
    def single(cls, x: int, y: int, *, key: str | None = None) -> MatrixSelection:
        return cls((MatrixBounds(x, y, x, y),), frozenset(() if key is None else (key,)))

    @property
    def is_empty(self) -> bool:
        return not self.rectangles and not self.keys

    def contains(self, x: int, y: int, *, key: str | None = None) -> bool:
        return (key is not None and key in self.keys) or any(rect.contains(x, y) for rect in self.rectangles)

    def coordinates(self, *, maximum: int | None = None) -> Iterator[tuple[int, int]]:
        seen: set[tuple[int, int]] = set()
        for rectangle in self.rectangles:
            for coordinate in rectangle.coordinates():
                if coordinate in seen:
                    continue
                seen.add(coordinate)
                if maximum is not None and len(seen) > maximum:
                    raise ValueError(f"selection contains more than {maximum} frames")
                yield coordinate


@dataclass(frozen=True, slots=True)
class MatrixAssetRef:
    source_key: str
    source_revision: str
    media_type: str = "application/octet-stream"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MatrixItem:
    key: str
    x: int
    y: int
    status: str = "empty"
    label: str = ""
    tooltip: str = ""
    asset: MatrixAssetRef | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.x) < 1 or int(self.y) < 1:
            raise ValueError("matrix coordinates are one-based and must be positive")
        object.__setattr__(self, "key", str(self.key))
        object.__setattr__(self, "x", int(self.x))
        object.__setattr__(self, "y", int(self.y))
        object.__setattr__(self, "status", str(self.status or "empty"))


@dataclass(frozen=True, slots=True)
class MatrixAggregate:
    bounds: MatrixBounds
    materialized_count: int
    status_counts: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MatrixViewportRequest:
    bounds: MatrixBounds
    lod: int = 0
    generation: int = 0
    asset_width: int = 0
    asset_height: int = 0

    def __post_init__(self) -> None:
        if int(self.lod) < 0:
            raise ValueError("LOD must not be negative")


@dataclass(frozen=True, slots=True)
class MatrixViewportResult:
    request: MatrixViewportRequest
    items: tuple[MatrixItem, ...] = ()
    aggregates: tuple[MatrixAggregate, ...] = ()
    source_revision: str = ""


@dataclass(frozen=True, slots=True)
class LodPolicy:
    overview_max_zoom: float = 0.18
    previews_min_zoom: float = 0.72
    details_min_zoom: float = 1.5
    subcells_min_zoom: float = 2.5

    def band_for_zoom(self, zoom: float) -> LodBand:
        value = max(0.0, float(zoom))
        if value < self.overview_max_zoom:
            return LodBand.OVERVIEW
        if value < self.previews_min_zoom:
            return LodBand.CELLS
        if value < self.details_min_zoom:
            return LodBand.PREVIEWS
        if value < self.subcells_min_zoom:
            return LodBand.DETAILS
        return LodBand.SUBCELLS


@dataclass(frozen=True, slots=True)
class MatrixConfig:
    cell_width: float = 48.0
    cell_height: float = 48.0
    gap_x: float = 3.0
    gap_y: float = 3.0
    overlap_x: float = 0.0
    overlap_y: float = 0.0
    prefetch_cells: int = 1
    lod_policy: LodPolicy = field(default_factory=LodPolicy)


@dataclass(frozen=True, slots=True)
class MatrixSession:
    namespace: str
    width: int
    height: int
    source_revision: str = ""
    orientation: MatrixOrientation = MatrixOrientation.Y_DOWN
    generation: int = 0

    def __post_init__(self) -> None:
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("matrix dimensions must be positive")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "orientation", MatrixOrientation(str(self.orientation)))


__all__ = [
    "LodBand",
    "LodPolicy",
    "MatrixAggregate",
    "MatrixAssetRef",
    "MatrixBounds",
    "MatrixConfig",
    "MatrixItem",
    "MatrixOrientation",
    "MatrixSelection",
    "MatrixSession",
    "MatrixViewportRequest",
    "MatrixViewportResult",
]
