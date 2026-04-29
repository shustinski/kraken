from __future__ import annotations

from kategb.application.dto import SampleRequest
from kategb.application.use_cases import BuildSample
from kategb.domain.frames import frames_from_range, parse_frame_range
from kategb.domain.models import CrystalInfo, FrameRange, LayerInfo


def test_area_frame_range_preserves_legacy_rectangular_selection() -> None:
    assert frames_from_range(FrameRange(3, 17), frames_in_row=10, mode="area") == (3, 4, 5, 6, 7, 13, 14, 15, 16, 17)


def test_parse_single_frame_as_range() -> None:
    assert parse_frame_range("42") == FrameRange(42, 42)


def test_build_sample_filters_by_author_percent_and_range() -> None:
    crystal = CrystalInfo(
        layers={
            "M1": LayerInfo(
                name="M1",
                author_frames={"Alice": (1, 2, 3, 4), "Bob": (5, 6, 7, 8)},
                frames_in_layer=8,
                frames_in_row=4,
            )
        }
    )

    frames = BuildSample().execute(
        crystal,
        SampleRequest(
            layer_name="M1",
            authors=("Alice", "Bob"),
            percent_per_author=50,
            frame_range_text="2-7",
            selection_mode="all",
            random_seed=1,
        ),
    )

    assert frames == (2, 4, 5, 6)
