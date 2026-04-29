from __future__ import annotations

import re
from collections.abc import Iterable

from .models import FrameRange, SelectionMode

_RANGE_SEPARATOR = re.compile(r"[;,]")
_BOUNDARY_SEPARATOR = re.compile(r"[-:]")


class FrameRangeError(ValueError):
    """Raised when the frame range expression is syntactically invalid."""


def parse_frame_ranges(expression: str) -> tuple[FrameRange, ...]:
    """Parse expressions like ``10-20;100:300,500`` into inclusive ranges."""
    text = expression.strip()
    if not text:
        raise FrameRangeError("Frame range is empty.")

    ranges: list[FrameRange] = []
    for raw_part in _RANGE_SEPARATOR.split(text):
        part = raw_part.strip()
        if not part:
            raise FrameRangeError("Frame range contains an empty part.")

        boundaries = [item.strip() for item in _BOUNDARY_SEPARATOR.split(part)]
        if len(boundaries) > 2:
            raise FrameRangeError(f"Too many boundaries in range: {part!r}.")
        if any(not item.isdigit() for item in boundaries):
            raise FrameRangeError(f"Frame range contains a non-numeric value: {part!r}.")

        start = int(boundaries[0])
        end = int(boundaries[-1])
        if start > end:
            start, end = end, start
        ranges.append(FrameRange(start=start, end=end))

    return tuple(ranges)


def expand_frames(
    ranges: Iterable[FrameRange],
    *,
    mode: SelectionMode,
    frames_per_row: int,
) -> tuple[int, ...]:
    if mode == SelectionMode.FULL_RANGE:
        return _unique_in_order(frame for item in ranges for frame in range(item.start, item.end + 1))

    if frames_per_row <= 0:
        raise FrameRangeError("Frames per row must be greater than zero for rectangle selection.")

    frames: list[int] = []
    for item in ranges:
        frames.extend(_expand_rectangle(item.start, item.end, frames_per_row))
    return _unique_in_order(frames)


def _expand_rectangle(first_frame: int, last_frame: int, frames_per_row: int) -> list[int]:
    rows = abs(last_frame // frames_per_row - first_frame // frames_per_row)
    if first_frame + frames_per_row * rows > last_frame:
        first_frame, last_frame = last_frame - rows * frames_per_row, first_frame + rows * frames_per_row

    row_start = first_frame
    row_end = last_frame - rows * frames_per_row
    frames: list[int] = []
    for _ in range(rows + 1):
        frames.extend(range(row_start, row_end + 1))
        row_start += frames_per_row
        row_end += frames_per_row
    return frames


def _unique_in_order(values: Iterable[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)
