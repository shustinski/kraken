"""Unit tests for pure viewport/zoom helpers (see ``viewport_navigation``)."""

from __future__ import annotations

import unittest

from contour.graphics.viewport_navigation import (
    polygon_overlay_visibility_after_space_toggle,
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


class VectorOverlaySpaceToggleTests(unittest.TestCase):
    def test_toggle_alternates_visibility_flag(self) -> None:
        hidden, visible = polygon_overlay_visibility_after_space_toggle(False)
        self.assertTrue(hidden)
        self.assertFalse(visible)
        hidden2, visible2 = polygon_overlay_visibility_after_space_toggle(hidden)
        self.assertFalse(hidden2)
        self.assertTrue(visible2)


if __name__ == "__main__":
    unittest.main()
