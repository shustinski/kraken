from __future__ import annotations

import unittest

from contour.graphics.minimap_geometry import (
    fitted_minimap_size,
    image_rect_in_minimap,
    minimap_point_to_scene,
    viewport_frame_in_minimap,
)


class MinimapGeometryTests(unittest.TestCase):
    def test_fitted_size_caps_long_side_and_keeps_aspect(self) -> None:
        width, height = fitted_minimap_size(400.0, 200.0, max_long_side=180.0)
        self.assertAlmostEqual(width, 180.0)
        self.assertAlmostEqual(height, 90.0)

    def test_fitted_size_for_tall_image(self) -> None:
        width, height = fitted_minimap_size(100.0, 400.0, max_long_side=180.0)
        self.assertAlmostEqual(width, 45.0)
        self.assertAlmostEqual(height, 180.0)

    def test_click_at_minimap_image_center_maps_to_image_center(self) -> None:
        image_scene = (0.0, 0.0, 400.0, 200.0)
        minimap_image = image_rect_in_minimap((400.0, 200.0), (180.0, 90.0))
        center_x = minimap_image[0] + minimap_image[2] / 2.0
        center_y = minimap_image[1] + minimap_image[3] / 2.0
        scene_x, scene_y = minimap_point_to_scene((center_x, center_y), image_scene, minimap_image)
        self.assertAlmostEqual(scene_x, 200.0)
        self.assertAlmostEqual(scene_y, 100.0)

    def test_viewport_frame_for_partial_zoom(self) -> None:
        image_scene = (0.0, 0.0, 400.0, 400.0)
        viewport_scene = (100.0, 100.0, 200.0, 200.0)
        minimap_image = (0.0, 0.0, 180.0, 180.0)
        frame = viewport_frame_in_minimap(image_scene, viewport_scene, minimap_image)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertAlmostEqual(frame[0], 45.0)
        self.assertAlmostEqual(frame[1], 45.0)
        self.assertAlmostEqual(frame[2], 90.0)
        self.assertAlmostEqual(frame[3], 90.0)

    def test_viewport_frame_none_when_view_misses_image(self) -> None:
        image_scene = (0.0, 0.0, 100.0, 100.0)
        viewport_scene = (200.0, 200.0, 50.0, 50.0)
        frame = viewport_frame_in_minimap(image_scene, viewport_scene, (0.0, 0.0, 180.0, 180.0))
        self.assertIsNone(frame)
