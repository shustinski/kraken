"""test_widget_interactions: test paste skips contacts."""
from __future__ import annotations
import numpy as np
from PyQt6.QtCore import QPointF
from contour.graphics_view import (
    PolygonEditorScene,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class PasteSkipsContactsMixin:
    def test_paste_clips_conductor_to_three_pixel_image_inset(self) -> None:
        scene = PolygonEditorScene()
        scene.set_image(np.zeros((100, 100), dtype=np.uint8))

        pasted = scene.add_cloned_polygons_at(
            [_rectangle_polygon(-10, -20, 110, 120)],
            QPointF(0.0, 0.0),
            QPointF(0.0, 0.0),
        )

        self.assertEqual(len(pasted), 1)
        polygon = scene.get_polygons()[0]
        self.assertEqual(min(x for x, _y in polygon.points), 3)
        self.assertEqual(max(x for x, _y in polygon.points), 97)
        self.assertEqual(min(y for _x, y in polygon.points), 3)
        self.assertEqual(max(y for _x, y in polygon.points), 97)

    def test_paste_rejects_conductor_fully_outside_image_inset(self) -> None:
        scene = PolygonEditorScene()
        scene.set_image(np.zeros((100, 100), dtype=np.uint8))
        undo_count = scene.undo_stack.count()

        pasted = scene.add_cloned_polygons_at(
            [_rectangle_polygon(-30, -30, -10, -10)],
            QPointF(0.0, 0.0),
            QPointF(0.0, 0.0),
        )

        self.assertEqual(pasted, [])
        self.assertEqual(scene.get_polygons(), [])
        self.assertEqual(scene.undo_stack.count(), undo_count)

    def test_paste_skips_contacts_closer_than_minimum_distance(self) -> None:
        scene = PolygonEditorScene()
        existing = _rectangle_polygon(0, 0, 10, 10)
        existing.category = "via"
        existing.shape_hint = "box"
        too_close = _rectangle_polygon(10, 0, 20, 10)
        too_close.category = "via"
        too_close.shape_hint = "box"
        far_enough = _rectangle_polygon(20, 0, 30, 10)
        far_enough.id = 2
        far_enough.category = "via"
        far_enough.shape_hint = "box"
        scene.set_polygons([existing])
        scene.set_minimum_contact_distance(20.0)

        pasted = scene.add_cloned_polygons_at(
            [too_close, far_enough],
            QPointF(0.0, 0.0),
            QPointF(0.0, 0.0),
        )

        self.assertEqual(len(pasted), 1)
        centers = sorted(
            (min(x for x, _y in polygon.points) + max(x for x, _y in polygon.points)) / 2.0
            for polygon in scene.get_polygons()
        )
        self.assertEqual(centers, [5.0, 25.0])
        scene.undo_stack.undo()
        self.assertEqual([polygon.id for polygon in scene.get_polygons()], [1])

    def test_paste_filters_contacts_against_contacts_accepted_in_same_batch(self) -> None:
        scene = PolygonEditorScene()
        first = _rectangle_polygon(0, 0, 10, 10)
        first.category = "via"
        first.shape_hint = "box"
        second = _rectangle_polygon(12, 0, 22, 10)
        second.id = 2
        second.category = "via"
        second.shape_hint = "box"
        scene.set_minimum_contact_distance(20.0)

        pasted = scene.add_cloned_polygons_at(
            [first, second],
            QPointF(0.0, 0.0),
            QPointF(0.0, 0.0),
        )

        self.assertEqual(len(pasted), 1)
        self.assertEqual(len(scene.get_polygons()), 1)
