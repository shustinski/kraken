from __future__ import annotations

import pytest

from csliser.domain.models import FrameRange, SelectionMode
from csliser.domain.ranges import FrameRangeError, expand_frames, parse_frame_ranges


def test_parse_frame_ranges_accepts_commas_semicolons_colons_and_single_frames() -> None:
    assert parse_frame_ranges("10-12;20:21,30") == (
        FrameRange(10, 12),
        FrameRange(20, 21),
        FrameRange(30, 30),
    )


def test_parse_frame_ranges_normalizes_descending_ranges() -> None:
    assert parse_frame_ranges("7-3") == (FrameRange(3, 7),)


def test_parse_frame_ranges_rejects_invalid_values() -> None:
    with pytest.raises(FrameRangeError):
        parse_frame_ranges("1-a")


def test_expand_full_ranges_is_inclusive_and_deduplicated() -> None:
    frames = expand_frames(
        (FrameRange(1, 3), FrameRange(3, 4)),
        mode=SelectionMode.FULL_RANGE,
        frames_per_row=10,
    )
    assert frames == (1, 2, 3, 4)


def test_expand_rectangle_keeps_legacy_grid_selection_semantics() -> None:
    frames = expand_frames(
        (FrameRange(1, 12),),
        mode=SelectionMode.RECTANGLE,
        frames_per_row=10,
    )
    assert frames == (1, 2, 11, 12)


def test_expand_rectangle_normalizes_inverted_corners() -> None:
    frames = expand_frames(
        (FrameRange(8, 13),),
        mode=SelectionMode.RECTANGLE,
        frames_per_row=10,
    )
    assert frames == (3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18)
