"""qt: frame rect."""
from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class FrameRect:
    """Inclusive, one-based rectangle of project coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        values = tuple(int(value) for value in (self.x1, self.y1, self.x2, self.y2))
        if min(values) < 1:
            raise ValueError("frame coordinates are one-based and must be positive")
        left, right = sorted((values[0], values[2]))
        top, bottom = sorted((values[1], values[3]))
        object.__setattr__(self, "x1", left)
        object.__setattr__(self, "y1", top)
        object.__setattr__(self, "x2", right)
        object.__setattr__(self, "y2", bottom)

    @classmethod
    def from_points(cls, first: tuple[int, int], second: tuple[int, int]) -> FrameRect:
        return cls(first[0], first[1], second[0], second[1])

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
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def intersects(self, other: FrameRect) -> bool:
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
