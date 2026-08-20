from __future__ import annotations

from math import hypot

import numpy as np

from contour.graphics.gradient_field_arrows import (
    DEFAULT_ARROW_LENGTH_VIEW_PX,
    gradient_sample_step_image_px,
    sample_gradient_field_arrows,
)
from contour.vision.metal_recovery.pipeline_stages import render_gradient_field_bgr


def test_sample_step_shrinks_with_zoom() -> None:
    zoomed_out = gradient_sample_step_image_px(1.0)
    zoomed_in = gradient_sample_step_image_px(0.25)
    assert zoomed_in < zoomed_out
    assert zoomed_out == 18.0
    assert zoomed_in == 4.5


def test_higher_zoom_samples_more_arrows_in_same_image_rect() -> None:
    gradient_x = np.full((120, 160), 12.0, dtype=np.float32)
    gradient_y = np.zeros((120, 160), dtype=np.float32)
    visible = (0.0, 0.0, 80.0, 80.0)
    coarse = sample_gradient_field_arrows(gradient_x, gradient_y, visible, 1.0)
    fine = sample_gradient_field_arrows(gradient_x, gradient_y, visible, 0.25)
    assert fine
    assert len(fine) > len(coarse)
    xs = sorted({round(sample[0], 5) for sample in fine})
    assert len(xs) >= 2
    assert xs[1] - xs[0] < gradient_sample_step_image_px(1.0)


def test_arrow_length_matches_view_pixels_in_scene_units() -> None:
    gradient_x = np.full((40, 40), 8.0, dtype=np.float32)
    gradient_y = np.zeros((40, 40), dtype=np.float32)
    scene_units_per_view_px = 0.5
    arrows = sample_gradient_field_arrows(
        gradient_x,
        gradient_y,
        (0.0, 0.0, 40.0, 40.0),
        scene_units_per_view_px,
    )
    assert arrows
    expected = DEFAULT_ARROW_LENGTH_VIEW_PX * scene_units_per_view_px
    for _x, _y, delta_x, delta_y in arrows:
        assert abs(hypot(delta_x, delta_y) - expected) < 1e-6
        assert delta_x > 0
        assert abs(delta_y) < 1e-6


def test_weak_gradients_are_skipped() -> None:
    gradient_x = np.full((32, 32), 1.0, dtype=np.float32)
    gradient_y = np.zeros((32, 32), dtype=np.float32)
    gradient_x[12:21, 12:21] = 100.0
    arrows = sample_gradient_field_arrows(
        gradient_x,
        gradient_y,
        (0.0, 0.0, 32.0, 32.0),
        0.2,
        view_step_px=16.0,
    )
    assert arrows
    for origin_x, origin_y, _dx, _dy in arrows:
        pixel_x = int(round(origin_x))
        pixel_y = int(round(origin_y))
        assert float(gradient_x[pixel_y, pixel_x]) >= 8.0


def test_arrow_count_is_capped() -> None:
    gradient_x = np.ones((400, 400), dtype=np.float32)
    gradient_y = np.zeros((400, 400), dtype=np.float32)
    arrows = sample_gradient_field_arrows(
        gradient_x,
        gradient_y,
        (0.0, 0.0, 400.0, 400.0),
        0.05,
        max_arrows=200,
    )
    assert len(arrows) <= 200


def test_render_gradient_field_bgr_has_no_baked_arrows() -> None:
    gradient_x = np.zeros((24, 32), dtype=np.float32)
    gradient_y = np.zeros((24, 32), dtype=np.float32)
    gradient_x[:, 10:22] = 40.0
    field = render_gradient_field_bgr(gradient_x, gradient_y)
    assert field.shape == (24, 32, 3)
    assert not np.all(field > 250, axis=2).any()
