"""Unit tests for pure viewport/zoom helpers (see ``viewport_navigation``)."""

from __future__ import annotations

import unittest

from contour.graphics.viewport_navigation import (
    image_coordinate_under_cursor,
    pan_offset_after_zoom_to_cursor,
    viewport_scroll_correction_after_scale_reanchor,
    scroll_values_after_viewport_drag,
)


class ZoomReanchorScrollTests(unittest.TestCase):
    def test_zero_when_anchor_pixel_unchanged(self) -> None:
        self.assertEqual(
            viewport_scroll_correction_after_scale_reanchor((10, 20), (10, 20)),
            (0, 0),
        )

    def test_matches_movement_of_scene_pick_on_viewport(self) -> None:
        self.assertEqual(viewport_scroll_correction_after_scale_reanchor((100, 80), (92, 76)), (-8, -4))


class ScrollPanTests(unittest.TestCase):
    def test_scrollbars_move_opposite_drag(self) -> None:
        self.assertEqual(scroll_values_after_viewport_drag(100.0, 200.0, 10.0, 20.0), (90.0, 180.0))


class ZoomToCursorCoordinateTests(unittest.TestCase):
    def test_centered_image_zoom_in_keeps_cursor_image_coordinate_stable(self) -> None:
        cursor = (420.0, 300.0)
        viewport = (900.0, 700.0)
        image = (400.0, 300.0)
        before = image_coordinate_under_cursor(cursor, viewport_size=viewport, image_size=image, scale=1.0)
        pan = pan_offset_after_zoom_to_cursor(
            cursor,
            viewport_size=viewport,
            image_size=image,
            old_scale=1.0,
            new_scale=1.5,
        )
        after = image_coordinate_under_cursor(
            cursor,
            viewport_size=viewport,
            image_size=image,
            scale=1.5,
            pan_offset_xy=pan,
        )
        self.assertAlmostEqual(before[0], after[0], delta=1e-6)
        self.assertAlmostEqual(before[1], after[1], delta=1e-6)

    def test_panned_large_image_zoom_out_keeps_cursor_image_coordinate_stable(self) -> None:
        cursor = (250.0, 180.0)
        viewport = (500.0, 400.0)
        image = (1200.0, 900.0)
        old_pan = (-130.0, -70.0)
        before = image_coordinate_under_cursor(
            cursor,
            viewport_size=viewport,
            image_size=image,
            scale=0.8,
            pan_offset_xy=old_pan,
        )
        pan = pan_offset_after_zoom_to_cursor(
            cursor,
            viewport_size=viewport,
            image_size=image,
            old_scale=0.8,
            new_scale=0.5,
            old_pan_offset_xy=old_pan,
        )
        after = image_coordinate_under_cursor(
            cursor,
            viewport_size=viewport,
            image_size=image,
            scale=0.5,
            pan_offset_xy=pan,
        )
        self.assertAlmostEqual(before[0], after[0], delta=1e-6)
        self.assertAlmostEqual(before[1], after[1], delta=1e-6)


if __name__ == "__main__":
    unittest.main()
