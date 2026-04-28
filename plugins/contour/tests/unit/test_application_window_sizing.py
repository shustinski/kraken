from __future__ import annotations

from contour.application.model import ContourApplicationModel
from contour.application.view import _bounded_initial_window_size


def test_model_initial_size_matches_compact_window_minimum() -> None:
    assert ContourApplicationModel(width=1, height=1).initial_size == (640, 420)


def test_initial_window_size_is_bounded_by_available_screen() -> None:
    size = _bounded_initial_window_size(
        width=1680,
        height=980,
        available_width=1024,
        available_height=768,
    )

    assert size.width() == 992
    assert size.height() == 736


def test_initial_window_size_keeps_preferred_minimum_when_space_allows() -> None:
    size = _bounded_initial_window_size(
        width=320,
        height=240,
        available_width=1920,
        available_height=1080,
    )

    assert size.width() == 640
    assert size.height() == 420


def test_initial_window_size_can_fit_very_small_screens() -> None:
    size = _bounded_initial_window_size(
        width=1680,
        height=980,
        available_width=500,
        available_height=360,
    )

    assert size.width() == 468
    assert size.height() == 328
