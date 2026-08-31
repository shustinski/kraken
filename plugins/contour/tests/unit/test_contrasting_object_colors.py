"""Contrast-aware conductor coloring for connectivity inspection."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication

from contour.application.processing import DisplaySettings
from contour.application.vector_geometry_postprocess import VectorGeometrySettings
from contour.commands import MovePolygonCommand, ReplacePolygonsPatchCommand
from contour.domain import PolygonData, compute_polygon_metrics
from contour.graphics.editor_scene import PolygonEditorScene
from contour.graphics.geometry import _contrasting_object_colors


def _app() -> QApplication:
    instance = QApplication.instance()
    return instance if instance is not None else QApplication([])


def _rectangle(
    polygon_id: int,
    left: float,
    top: float,
    width: float = 10.0,
    height: float = 10.0,
    *,
    is_hole: bool = False,
    parent_id: int | None = None,
    category: str = "conductor",
    recognition_score: float | None = None,
) -> PolygonData:
    points = [
        (left, top),
        (left + width, top),
        (left + width, top + height),
        (left, top + height),
    ]
    area, perimeter, bbox = compute_polygon_metrics(points)
    return PolygonData(
        id=polygon_id,
        points=points,
        is_hole=is_hole,
        parent_id=parent_id,
        category=category,
        area=area,
        perimeter=perimeter,
        bbox=bbox,
        recognition_score=recognition_score,
    )


class ContrastingObjectColorTests(unittest.TestCase):
    def test_nearby_objects_with_same_preferred_color_are_distinct(self) -> None:
        first = _rectangle(1, 0.0, 0.0)
        second = _rectangle(13, 15.0, 0.0)

        colors = _contrasting_object_colors([first, second], (1000.0, 1000.0))

        self.assertNotEqual(colors[first.id], colors[second.id])

    def test_distant_objects_can_reuse_a_color(self) -> None:
        first = _rectangle(1, 0.0, 0.0)
        second = _rectangle(13, 100.0, 0.0)

        colors = _contrasting_object_colors([first, second], (1000.0, 1000.0))

        self.assertEqual(colors[first.id], colors[second.id])

    def test_result_is_independent_of_input_order(self) -> None:
        polygons = [
            _rectangle(1, 0.0, 0.0),
            _rectangle(13, 15.0, 0.0),
            _rectangle(25, 30.0, 0.0),
        ]

        forward = _contrasting_object_colors(polygons, (1000.0, 1000.0))
        backward = _contrasting_object_colors(reversed(polygons), (1000.0, 1000.0))

        self.assertEqual(forward, backward)

    def test_hole_inherits_parent_color_and_via_is_not_recolored(self) -> None:
        parent = _rectangle(1, 0.0, 0.0, width=40.0, height=40.0)
        hole = _rectangle(2, 10.0, 10.0, is_hole=True, parent_id=parent.id)
        via = _rectangle(3, 50.0, 0.0, category="via", recognition_score=70.0)

        colors = _contrasting_object_colors([via, hole, parent], (1000.0, 1000.0))

        self.assertEqual(colors[hole.id], colors[parent.id])
        self.assertNotIn(via.id, colors)


class ContrastingObjectColorSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def setUp(self) -> None:
        self.scene = PolygonEditorScene()
        self.scene.set_vector_geometry_settings(
            VectorGeometrySettings(
                min_outer_area_px2=1.0,
                min_hole_area_to_remove_px2=0.0,
                drop_three_vertex_triangle_artifacts=False,
            )
        )
        self.scene.set_image_pixmap(QPixmap(1000, 1000))

    def tearDown(self) -> None:
        self.scene.deleteLater()

    def test_toggle_changes_rendered_colors_and_restores_defaults(self) -> None:
        first = _rectangle(1, 100.0, 100.0, width=40.0, height=40.0)
        second = _rectangle(13, 145.0, 100.0, width=40.0, height=40.0)
        self.scene.set_polygons([first, second])

        self.scene.set_random_object_colors_enabled(True)

        self.assertNotEqual(
            self.scene._polygon_items[first.id].pen().color().name(),
            self.scene._polygon_items[second.id].pen().color().name(),
        )

        self.scene.set_random_object_colors_enabled(False)

        expected = DisplaySettings().external_color.lower()
        self.assertEqual(self.scene._polygon_items[first.id].pen().color().name(), expected)
        self.assertEqual(self.scene._polygon_items[second.id].pen().color().name(), expected)

    def test_geometry_edit_and_undo_redo_rebuild_neighbor_colors(self) -> None:
        first = _rectangle(1, 100.0, 100.0, width=40.0, height=40.0)
        second = _rectangle(13, 300.0, 100.0, width=40.0, height=40.0)
        moved_points = [(145.0, 100.0), (185.0, 100.0), (185.0, 140.0), (145.0, 140.0)]
        self.scene.set_polygons([first, second])
        self.scene.set_random_object_colors_enabled(True)
        self.assertEqual(self.scene._object_colors[first.id], self.scene._object_colors[second.id])

        self.scene.undo_stack.push(MovePolygonCommand(self.scene, second.id, second.points, moved_points))
        self.assertNotEqual(self.scene._object_colors[first.id], self.scene._object_colors[second.id])

        self.scene.undo_stack.undo()
        self.assertEqual(self.scene._object_colors[first.id], self.scene._object_colors[second.id])

        self.scene.undo_stack.redo()
        self.assertNotEqual(self.scene._object_colors[first.id], self.scene._object_colors[second.id])

    def test_incremental_replace_and_undo_rebuild_neighbor_colors(self) -> None:
        first = _rectangle(1, 100.0, 100.0, width=40.0, height=40.0)
        distant = _rectangle(13, 300.0, 100.0, width=40.0, height=40.0)
        nearby = _rectangle(25, 145.0, 100.0, width=40.0, height=40.0)
        self.scene.set_polygons([first, distant])
        self.scene.set_random_object_colors_enabled(True)
        self.assertEqual(self.scene._object_colors[first.id], self.scene._object_colors[distant.id])

        self.scene.undo_stack.push(
            ReplacePolygonsPatchCommand(
                self.scene,
                removed_polygons=[distant],
                added_polygons=[nearby],
                description="Replace conductor fragment",
            )
        )
        self.assertNotEqual(self.scene._object_colors[first.id], self.scene._object_colors[nearby.id])

        self.scene.undo_stack.undo()
        self.assertEqual(self.scene._object_colors[first.id], self.scene._object_colors[distant.id])

    def test_hole_uses_parent_color_and_via_score_keeps_precedence(self) -> None:
        parent = _rectangle(1, 100.0, 100.0, width=40.0, height=40.0)
        hole = _rectangle(2, 110.0, 110.0, is_hole=True, parent_id=parent.id)
        via = _rectangle(3, 150.0, 100.0, category="via", recognition_score=70.0)
        self.scene.set_polygons([parent, hole, via])

        self.scene.set_random_object_colors_enabled(True)

        self.assertEqual(
            self.scene._polygon_items[hole.id].pen().color().name(),
            self.scene._polygon_items[parent.id].pen().color().name(),
        )
        self.assertEqual(
            self.scene._polygon_items[via.id].pen().color().name(),
            self.scene._via_score_color(via),
        )

    def test_bulk_frame_load_rebuilds_color_and_z_caches_once(self) -> None:
        polygons = [
            _rectangle(
                polygon_id,
                100.0 + (polygon_id % 20) * 45.0,
                100.0 + (polygon_id // 20) * 45.0,
                width=40.0,
                height=40.0,
            )
            for polygon_id in range(1, 201)
        ]
        self.scene.set_random_object_colors_enabled(True)

        with (
            patch.object(
                self.scene,
                "_rebuild_object_colors",
                wraps=self.scene._rebuild_object_colors,
            ) as rebuild_colors,
            patch.object(
                self.scene,
                "_rebuild_outer_pick_z_ranks",
                wraps=self.scene._rebuild_outer_pick_z_ranks,
            ) as rebuild_z_ranks,
        ):
            self.scene.set_image_pixmap(QPixmap(1200, 900))
            self.scene.set_polygons(polygons, scan_repair=False)

        self.assertEqual(rebuild_colors.call_count, 1)
        self.assertEqual(rebuild_z_ranks.call_count, 1)


if __name__ == "__main__":
    unittest.main()
