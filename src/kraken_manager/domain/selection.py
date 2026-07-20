"""Compact, versioned frame-selection schema for very large sparse grids."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

from .common import DomainValidationError
from .project import FrameCoordinate


@dataclass(frozen=True, slots=True)
class FrameRectangle:
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        FrameCoordinate(self.x1, self.y1)
        FrameCoordinate(self.x2, self.y2)
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise DomainValidationError("rectangle end must not precede its start")

    def contains(self, coordinate: FrameCoordinate) -> bool:
        return self.x1 <= coordinate.x <= self.x2 and self.y1 <= coordinate.y <= self.y2


@dataclass(frozen=True, slots=True)
class FrameRowRange:
    y: int
    x_start: int
    x_end: int

    def __post_init__(self) -> None:
        FrameCoordinate(self.x_start, self.y)
        FrameCoordinate(self.x_end, self.y)
        if self.x_end < self.x_start:
            raise DomainValidationError("row range end must not precede its start")

    def contains(self, coordinate: FrameCoordinate) -> bool:
        return coordinate.y == self.y and self.x_start <= coordinate.x <= self.x_end


@dataclass(frozen=True, slots=True)
class FrameSelectionV1:
    """Union of rectangles and row ranges, minus explicit coordinates.

    This representation stays compact for million-frame selections and avoids
    placing a giant list of frame IDs into events and plugin manifests.
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    rectangles: tuple[FrameRectangle, ...] = ()
    row_ranges: tuple[FrameRowRange, ...] = ()
    exclusions: frozenset[FrameCoordinate] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rectangles", tuple(self.rectangles))
        object.__setattr__(self, "row_ranges", tuple(self.row_ranges))
        object.__setattr__(self, "exclusions", frozenset(self.exclusions))
        if not self.rectangles and not self.row_ranges:
            raise DomainValidationError("frame selection must include at least one rectangle or row range")
        for exclusion in self.exclusions:
            if not self._included_before_exclusion(exclusion):
                raise DomainValidationError("every excluded coordinate must first be included by the selection")

    @classmethod
    def rectangle(cls, x1: int, y1: int, x2: int, y2: int) -> Self:
        return cls(rectangles=(FrameRectangle(x1, y1, x2, y2),))

    def _included_before_exclusion(self, coordinate: FrameCoordinate) -> bool:
        return any(rectangle.contains(coordinate) for rectangle in self.rectangles) or any(
            row_range.contains(coordinate) for row_range in self.row_ranges
        )

    def contains(self, coordinate: FrameCoordinate) -> bool:
        return coordinate not in self.exclusions and self._included_before_exclusion(coordinate)

    def validate_bounds(self, *, width: int, height: int) -> Self:
        if width < 1 or height < 1:
            raise DomainValidationError("selection bounds must be positive")
        for rectangle in self.rectangles:
            if rectangle.x2 > width or rectangle.y2 > height:
                raise DomainValidationError("selection rectangle is outside the project grid")
        for row_range in self.row_ranges:
            if row_range.x_end > width or row_range.y > height:
                raise DomainValidationError("selection row range is outside the project grid")
        return self

    @property
    def min_y(self) -> int:
        return min(
            [rectangle.y1 for rectangle in self.rectangles]
            + [row_range.y for row_range in self.row_ranges]
        )

    @property
    def max_y(self) -> int:
        return max(
            [rectangle.y2 for rectangle in self.rectangles]
            + [row_range.y for row_range in self.row_ranges]
        )

    def intervals_for_row(self, y: int) -> tuple[tuple[int, int], ...]:
        """Return merged inclusive X intervals for a row before exclusions."""

        intervals = [
            (rectangle.x1, rectangle.x2)
            for rectangle in self.rectangles
            if rectangle.y1 <= y <= rectangle.y2
        ]
        intervals.extend(
            (row_range.x_start, row_range.x_end)
            for row_range in self.row_ranges
            if row_range.y == y
        )
        if not intervals:
            return ()
        intervals.sort()
        merged: list[tuple[int, int]] = []
        for start, end in intervals:
            if not merged or start > merged[-1][1] + 1:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return tuple(merged)

    def cardinality(self) -> int:
        """Count unique selected coordinates without materialising them."""

        total = 0
        # The selected X intervals only change at rectangle boundaries and at
        # explicit row ranges. A sweep over those breakpoints is independent of
        # a rectangle's height (important for grids with tens of millions of
        # rows).
        breakpoints = {self.min_y, self.max_y + 1}
        for rectangle in self.rectangles:
            breakpoints.update({rectangle.y1, rectangle.y2 + 1})
        for row_range in self.row_ranges:
            breakpoints.update({row_range.y, row_range.y + 1})
        ordered = sorted(breakpoints)
        for start_y, end_y in zip(ordered, ordered[1:]):
            row_width = sum(end - start + 1 for start, end in self.intervals_for_row(start_y))
            total += row_width * (end_y - start_y)
        return total - len(self.exclusions)

    def iter_coordinates(self, *, limit: int | None = None) -> Iterator[FrameCoordinate]:
        """Iterate in deterministic row-major order.

        ``limit`` is a safety valve for UI/adapters. It limits yielded items and
        does not mutate or truncate the selection schema itself.
        """

        if limit is not None and limit < 0:
            raise DomainValidationError("iteration limit must not be negative")
        emitted = 0
        for y in range(self.min_y, self.max_y + 1):
            for start, end in self.intervals_for_row(y):
                for x in range(start, end + 1):
                    coordinate = FrameCoordinate(x, y)
                    if coordinate in self.exclusions:
                        continue
                    if limit is not None and emitted >= limit:
                        return
                    yield coordinate
                    emitted += 1

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "rectangles": [
                {"x1": item.x1, "y1": item.y1, "x2": item.x2, "y2": item.y2}
                for item in self.rectangles
            ],
            "row_ranges": [
                {"y": item.y, "x_start": item.x_start, "x_end": item.x_end}
                for item in self.row_ranges
            ],
            "exclusions": [
                {"x": item.x, "y": item.y}
                for item in sorted(self.exclusions, key=lambda coordinate: (coordinate.y, coordinate.x))
            ],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> Self:
        if raw.get("schema_version") != cls.SCHEMA_VERSION:
            raise DomainValidationError("unsupported frame selection schema version")
        rectangles_raw = raw.get("rectangles", ())
        ranges_raw = raw.get("row_ranges", ())
        exclusions_raw = raw.get("exclusions", ())
        if not isinstance(rectangles_raw, Sequence) or isinstance(rectangles_raw, (str, bytes)):
            raise DomainValidationError("selection rectangles must be a sequence")
        if not isinstance(ranges_raw, Sequence) or isinstance(ranges_raw, (str, bytes)):
            raise DomainValidationError("selection row_ranges must be a sequence")
        if not isinstance(exclusions_raw, Sequence) or isinstance(exclusions_raw, (str, bytes)):
            raise DomainValidationError("selection exclusions must be a sequence")

        def item_mapping(item: object, field: str) -> Mapping[str, object]:
            if not isinstance(item, Mapping):
                raise DomainValidationError(f"{field} items must be mappings")
            return item

        def integer(item: object, key: str, field: str) -> int:
            value = item_mapping(item, field).get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise DomainValidationError(f"{field}.{key} must be an integer")
            return value

        try:
            rectangles = tuple(
                FrameRectangle(
                    x1=integer(item, "x1", "rectangles"),
                    y1=integer(item, "y1", "rectangles"),
                    x2=integer(item, "x2", "rectangles"),
                    y2=integer(item, "y2", "rectangles"),
                )
                for item in rectangles_raw
            )
            row_ranges = tuple(
                FrameRowRange(
                    y=integer(item, "y", "row_ranges"),
                    x_start=integer(item, "x_start", "row_ranges"),
                    x_end=integer(item, "x_end", "row_ranges"),
                )
                for item in ranges_raw
            )
            exclusions = frozenset(
                FrameCoordinate(
                    x=integer(item, "x", "exclusions"),
                    y=integer(item, "y", "exclusions"),
                )
                for item in exclusions_raw
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, DomainValidationError):
                raise
            raise DomainValidationError("invalid frame selection payload") from exc
        return cls(rectangles=rectangles, row_ranges=row_ranges, exclusions=exclusions)


__all__ = ["FrameRectangle", "FrameRowRange", "FrameSelectionV1"]
