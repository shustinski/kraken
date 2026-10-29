from __future__ import annotations

import re
from collections.abc import Iterable
from itertools import islice

from .models import FrameRange, SelectionMode

_RANGE_SEPARATOR = re.compile(r"[;,]")
_BOUNDARY_SEPARATOR = re.compile(r"[-:]")
MAX_EXPRESSION_LENGTH = 65536
MAX_RANGES = 4096
MAX_FRAME_NUMBER = 2**63 - 1
MAX_SELECTED_FRAMES = 1_000_000
MAX_EXPANSION_VISITS = 4_000_000


class FrameRangeError(ValueError):
    """Raised when the frame range expression is syntactically invalid."""


def parse_frame_ranges(expression: str) -> tuple[FrameRange, ...]:
    """Parse expressions like ``10-20;100:300,500`` into inclusive ranges."""
    if not isinstance(expression, str) or len(expression) > MAX_EXPRESSION_LENGTH:
        raise FrameRangeError(f"Frame expression must be text of at most {MAX_EXPRESSION_LENGTH} characters.")
    text = expression.strip()
    if not text:
        raise FrameRangeError("Frame range is empty.")

    ranges: list[FrameRange] = []
    for raw_part in _RANGE_SEPARATOR.split(text):
        if len(ranges) >= MAX_RANGES:
            raise FrameRangeError(f"At most {MAX_RANGES} ranges are supported.")
        part = raw_part.strip()
        if not part:
            raise FrameRangeError("Frame range contains an empty part.")

        boundaries = [item.strip() for item in _BOUNDARY_SEPARATOR.split(part)]
        if len(boundaries) > 2:
            raise FrameRangeError(f"Too many boundaries in range: {part!r}.")
        if any(not item.isdecimal() for item in boundaries):
            raise FrameRangeError(f"Frame range contains a non-numeric value: {part!r}.")

        try:
            start, end = int(boundaries[0]), int(boundaries[-1])
        except ValueError as exc:
            raise FrameRangeError("Frame number is too long.") from exc
        if max(start, end) > MAX_FRAME_NUMBER:
            raise FrameRangeError(f"Frame numbers must not exceed {MAX_FRAME_NUMBER}.")
        if start > end:
            start, end = end, start
        ranges.append(FrameRange(start=start, end=end))

    return tuple(ranges)


def expand_frames(
    ranges: Iterable[FrameRange],
    *,
    mode: SelectionMode,
    frames_per_row: int,
    max_frames: int = MAX_SELECTED_FRAMES,
    max_visits: int = MAX_EXPANSION_VISITS,
) -> tuple[int, ...]:
    if type(max_frames) is not int or max_frames <= 0 or type(max_visits) is not int or max_visits <= 0:
        raise FrameRangeError("Selection budgets must be positive integers.")
    try:
        mode = SelectionMode(mode)
    except ValueError as exc:
        raise FrameRangeError("Unknown selection mode.") from exc
    ranges = tuple(islice(ranges, MAX_RANGES + 1))
    if len(ranges) > MAX_RANGES:
        raise FrameRangeError(f"At most {MAX_RANGES} ranges are supported.")
    if mode is SelectionMode.RECTANGLE and (type(frames_per_row) is not int or frames_per_row <= 0):
        raise FrameRangeError("Frames per row must be greater than zero for rectangle selection.")
    visits = 0
    for item in ranges:
        if (not isinstance(item, FrameRange) or type(item.start) is not int or type(item.end) is not int
                or not 0 <= item.start <= item.end <= MAX_FRAME_NUMBER):
            raise FrameRangeError("Frame boundaries must be ordered nonnegative supported integers.")
        if mode is SelectionMode.FULL_RANGE:
            count = item.end - item.start + 1
        else:
            first, columns, rows = _rectangle_parameters(item.start, item.end, frames_per_row)
            count = columns * (rows + 1)
            if first + rows * frames_per_row + columns - 1 > MAX_FRAME_NUMBER:
                raise FrameRangeError("Rectangle extends beyond the supported frame numbers.")
        if count > max_frames:
            raise FrameRangeError(f"Selection exceeds {max_frames} unique frames.")
        visits += count
        if visits > max_visits:
            raise FrameRangeError(f"Selection exceeds {max_visits} expansion visits; reduce overlapping ranges.")
    if mode == SelectionMode.FULL_RANGE:
        values = (frame for item in ranges for frame in range(item.start, item.end + 1))
    else:
        values = (frame for item in ranges for frame in _expand_rectangle(item.start, item.end, frames_per_row))
    return _unique_in_order(values, max_frames=max_frames)


def _rectangle_parameters(first_frame: int, last_frame: int, frames_per_row: int) -> tuple[int, int, int]:
    rows = abs(last_frame // frames_per_row - first_frame // frames_per_row)
    if first_frame + frames_per_row * rows > last_frame:
        first_frame, last_frame = last_frame - rows * frames_per_row, first_frame + rows * frames_per_row

    return first_frame, last_frame - rows * frames_per_row - first_frame + 1, rows


def _expand_rectangle(first_frame: int, last_frame: int, frames_per_row: int) -> Iterable[int]:
    row_start, columns, rows = _rectangle_parameters(first_frame, last_frame, frames_per_row)
    for _ in range(rows + 1):
        yield from range(row_start, row_start + columns)
        row_start += frames_per_row


def _unique_in_order(values: Iterable[int], *, max_frames: int) -> tuple[int, ...]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value not in seen:
            if len(result) >= max_frames:
                raise FrameRangeError(f"Selection exceeds {max_frames} unique frames.")
            seen.add(value)
            result.append(value)
    return tuple(result)
