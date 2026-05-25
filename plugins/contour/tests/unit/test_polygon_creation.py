"""Polygon creation / commit validation and editor scene wiring."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QApplication

from contour.application.processing import ImageProcessingState
from contour.application.services.workspace_session import WorkspaceSession
from contour.domain import PolygonData, compute_polygon_metrics
from contour.graphics.editor_scene import PolygonEditorScene
from contour.graphics.polygon_creation import (
    POLYGON_COMMIT_INVALID_RING,
    POLYGON_COMMIT_TOO_FEW_VERTICES,
    POLYGON_COMMIT_TOO_SMALL_AREA,
    polygon_commit_acceptability,
)


def _app() -> QApplication:
    instance = QApplication.instance()
    return instance if instance is not None else QApplication([])


def _triangle() -> list[tuple[float, float]]:
    return [(0.0, 0.0), (100.0, 0.0), (50.0, 80.0)]


def _polygon(
    polygon_id: int,
    points: list[tuple[float, float]],
    *,
    is_hole: bool = False,
    parent_id: int | None = None,
) -> PolygonData:
    area, perimeter, bbox = compute_polygon_metrics(points)
    return PolygonData(
        id=polygon_id,
        points=points,
        is_hole=is_hole,
        parent_id=parent_id,
        area=area,
        perimeter=perimeter,
        bbox=bbox,
    )


class PolygonCommitAcceptabilityTests(unittest.TestCase):
    def test_too_few_vertices(self) -> None:
        ok, reason = polygon_commit_acceptability([(0.0, 0.0), (1.0, 0.0)])
        self.assertFalse(ok)
        self.assertEqual(reason, POLYGON_COMMIT_TOO_FEW_VERTICES)

    def test_valid_triangle(self) -> None:
        ok, reason = polygon_commit_acceptability(_triangle())
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_bowtie_is_invalid_ring(self) -> None:
        bowtie = [(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)]
        ok, reason = polygon_commit_acceptability(bowtie)
        self.assertFalse(ok)
        self.assertEqual(reason, POLYGON_COMMIT_INVALID_RING)

    def test_near_zero_area_colinear(self) -> None:
        colinear = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        ok, reason = polygon_commit_acceptability(colinear)
        self.assertFalse(ok)
        self.assertEqual(reason, POLYGON_COMMIT_TOO_SMALL_AREA)

    def test_drop_triangle_keeps_manual_outline_three_vertex_polygons(self) -> None:
        from contour.application.vector_geometry_postprocess import drop_triangle_outer_artifacts

        pts = [(0.0, 0.0), (60.0, 0.0), (30.0, 50.0)]
        area, perim, bbox = compute_polygon_metrics(pts)
        manual = PolygonData(id=9, points=pts, shape_hint="manual_outline", area=area, perimeter=perim, bbox=bbox)
        noisy = PolygonData(id=99, points=[(100.0, 100.0), (103.0, 100.0), (101.5, 101.0)], area=0.1)
        survivors = drop_triangle_outer_artifacts([manual, noisy], enabled=True)
        self.assertEqual({p.id for p in survivors}, {9})


class PolygonEditorSceneCreationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def setUp(self) -> None:
        self.scene = PolygonEditorScene()

    def tearDown(self) -> None:
        self.scene.deleteLater()

    def _reset(self, initial: list[PolygonData] | None = None) -> None:
        self.scene.set_polygons([p.clone() for p in (initial or [])])

    def test_points_mode_finish_adds_polygon_selects_emits_polygon_changed(self) -> None:
        self._reset([])
        changed: list[int] = []

        def _bump() -> None:
            changed.append(1)

        self.scene.polygonsChanged.connect(_bump)

        active: list[int | None] = []
        self.scene.activePolygonChanged.connect(active.append)

        self.scene.append_pending_point(QPointF(10.0, 10.0))
        self.scene.append_pending_point(QPointF(40.0, 10.0))
        self.scene.append_pending_point(QPointF(25.0, 50.0))
        ok = self.scene.finish_pending_polygon()
        self.assertTrue(ok)
        data = self.scene.get_polygons()
        self.assertEqual(len(data), 1)
        new_id = data[0].id
        self.assertEqual(self.scene.selected_polygon_id(), new_id)
        self.assertTrue(changed)
        self.assertEqual(active[-1], new_id)

    def test_points_mode_rounds_vertices_to_integer_coordinates(self) -> None:
        self._reset([])

        self.scene.append_pending_point(QPointF(10.2, 10.6))
        self.scene.append_pending_point(QPointF(40.4, 10.1))
        self.scene.append_pending_point(QPointF(25.5, 50.5))
        ok = self.scene.finish_pending_polygon()

        self.assertTrue(ok)
        self.assertEqual(self.scene.get_polygons()[0].points, [(10, 11), (40, 10), (26, 50)])

    def test_rectangle_adds_polygon_selected(self) -> None:
        self._reset([])
        ok = self.scene.add_rectangle_polygon(QPointF(5.0, 5.0), QPointF(35.0, 28.0), erase=False)
        self.assertTrue(ok)
        ids = sorted(polygon.id for polygon in self.scene.get_polygons())
        self.assertEqual(len(ids), 1)
        self.assertEqual(self.scene.selected_polygon_id(), ids[0])

    def test_rectangle_rounds_vertices_to_integer_coordinates(self) -> None:
        self._reset([])

        ok = self.scene.add_rectangle_polygon(QPointF(5.2, 5.6), QPointF(35.4, 28.5), erase=False)

        self.assertTrue(ok)
        self.assertEqual(self.scene.get_polygons()[0].points, [(5, 6), (35, 6), (35, 28), (5, 28)])

    def test_delete_parent_polygon_removes_internal_contours(self) -> None:
        outer = _polygon(1, [(0.0, 0.0), (80.0, 0.0), (80.0, 80.0), (0.0, 80.0)])
        hole = _polygon(
            2,
            [(20.0, 20.0), (40.0, 20.0), (40.0, 40.0), (20.0, 40.0)],
            is_hole=True,
            parent_id=1,
        )
        unrelated = _polygon(3, [(100.0, 0.0), (130.0, 0.0), (130.0, 30.0), (100.0, 30.0)])
        self._reset([outer, hole, unrelated])

        self.assertTrue(self.scene.delete_polygon(1))

        remaining_ids = {polygon.id for polygon in self.scene.get_polygons()}
        self.assertEqual(remaining_ids, {3})

        self.scene.undo_stack.undo()
        restored = {polygon.id: polygon for polygon in self.scene.get_polygons()}
        self.assertEqual(set(restored), {1, 2, 3})
        self.assertTrue(restored[2].is_hole)
        self.assertEqual(restored[2].parent_id, 1)

    def test_invalid_polygon_not_committed_keeps_pending(self) -> None:
        self._reset([])
        self.scene.append_pending_point(QPointF(10.0, 10.0))
        self.scene.append_pending_point(QPointF(20.0, 10.0))
        self.scene.append_pending_point(QPointF(15.0, 10.0))
        ok = self.scene.finish_pending_polygon()
        self.assertFalse(ok)
        self.assertEqual(len(self.scene.get_polygons()), 0)
        self.assertTrue(self.scene.has_pending_polygon())

    def test_points_mode_preview_fills_valid_polygon_blue(self) -> None:
        self._reset([])
        self.scene.append_pending_point(QPointF(10.0, 10.0))
        self.scene.append_pending_point(QPointF(40.0, 10.0))
        self.scene.append_pending_point(QPointF(25.0, 50.0))

        self.assertEqual(self.scene._pending_path_item.brush().color().name().lower(), "#38bdf8")

    def test_points_mode_preview_fills_unfinishable_polygon_red(self) -> None:
        self._reset([])
        self.scene.append_pending_point(QPointF(10.0, 10.0))
        self.scene.append_pending_point(QPointF(20.0, 10.0))
        self.scene.append_pending_point(QPointF(15.0, 10.0))

        self.assertEqual(self.scene._pending_path_item.brush().color().name().lower(), "#ef4444")

    def test_finish_with_under_three_vertices_clears_pending(self) -> None:
        self._reset([])
        self.scene.append_pending_point(QPointF(1.0, 1.0))
        self.scene.append_pending_point(QPointF(2.0, 2.0))
        ok = self.scene.finish_pending_polygon()
        self.assertFalse(ok)
        self.assertFalse(self.scene.has_pending_polygon())

    def test_workspace_dirty_after_polygon_commit(self) -> None:
        self._reset([])
        key = str(Path("manual_test.png"))
        state = ImageProcessingState(image_path=key, polygons=[], reference_polygons=[])
        session = WorkspaceSession()
        session._state_cache[key] = state  # type: ignore[attr-defined]
        session._current_image_path = key  # type: ignore[attr-defined]
        session._current_state = state  # type: ignore[attr-defined]
        self.assertFalse(session.image_has_changes(key))

        self.scene.append_pending_point(QPointF(8.0, 8.0))
        self.scene.append_pending_point(QPointF(32.0, 8.0))
        self.scene.append_pending_point(QPointF(20.0, 40.0))
        self.scene.finish_pending_polygon()

        session.update_current_polygons(self.scene.get_polygons())
        self.assertTrue(session.image_has_changes(key))

    def test_vector_postprocess_preserves_primary_selection_when_id_survives(self) -> None:
        points = [(20.0, 20.0), (80.0, 20.0), (50.0, 70.0)]
        area, perimeter, bbox = compute_polygon_metrics(points)
        outer = PolygonData(id=10, points=points, area=area, perimeter=perimeter, bbox=bbox)
        self._reset([outer])
        self.scene.select_polygon(10)
        self.scene.append_pending_point(QPointF(110.0, 30.0))
        self.scene.append_pending_point(QPointF(170.0, 30.0))
        self.scene.append_pending_point(QPointF(140.0, 90.0))
        self.scene.finish_pending_polygon()

        polygons = self.scene.get_polygons()
        self.assertEqual(len(polygons), 2)
        self.assertEqual(self.scene.selected_polygon_id(), 11)
