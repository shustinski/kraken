"""test_widget_interactions: test cutting shape hole."""
from __future__ import annotations
import numpy as np
from PyQt6.QtCore import QPointF
from contour.application.processing import (
    DisplaySettings,
)
from contour.graphics_items import EditablePolygonItem
from contour.utils import draw_polygon_overlay

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class CuttingShapeHoleMixin:
    def test_cutting_shape_with_hole_keeps_center_empty(self) -> None:
        self.view.set_polygons([])
        ring_points = [
            (30.0, 30.0),
            (70.0, 30.0),
            (70.0, 70.0),
            (30.0, 70.0),
            (30.0, 30.0),
        ]
        self.view._editor_scene.add_brush_stroke(ring_points, thickness=10.0)

        changed = self.view._editor_scene.add_rectangle_polygon(
            QPointF(22.0, 38.0),
            QPointF(30.0, 62.0),
            erase=True,
        )
        self._app.processEvents()

        self.assertTrue(changed)
        polygons = self.view.get_polygons()
        self.assertTrue(any(polygon.is_hole for polygon in polygons))
        outer_item = next(item for item in self.view._editor_scene._polygon_items.values() if not item.polygon.is_hole)
        self.assertFalse(outer_item.contains(QPointF(50.0, 50.0)))

    def test_box_and_via_items_display_as_ellipses(self) -> None:
        polygon = _rectangle_polygon(20, 20, 80, 60)
        polygon.category = "via"
        polygon.shape_hint = "box"

        item = EditablePolygonItem(polygon, DisplaySettings(show_vertices=True))

        self.assertGreater(item.path().elementCount(), len(polygon.points) + 1)
        self.assertEqual(len(item._handles), 0)

    def test_overlay_preview_draws_box_and_via_as_ellipse(self) -> None:
        polygon = _rectangle_polygon(10, 10, 30, 30)
        polygon.category = "via"
        polygon.shape_hint = "box"
        image = np.zeros((48, 48, 3), dtype=np.uint8)

        overlay = draw_polygon_overlay(
            image,
            [polygon],
            DisplaySettings(external_color="#00FF00", fill_opacity=1.0, show_vertices=False),
        )

        self.assertEqual(int(overlay[20, 20, 1]), 255)
        self.assertEqual(int(overlay[10, 10, 1]), 0)
