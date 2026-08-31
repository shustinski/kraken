"""Polygon creation / commit validation and editor scene wiring."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QApplication

from contour.application.processing import ImageProcessingState
from contour.application.services.workspace_session import WorkspaceSession
from contour.application.vector_geometry_postprocess import VectorGeometrySettings
from contour.domain import PolygonData, compute_polygon_metrics
from contour.domain.polygon_ring import is_valid_closed_polygon_ring
from contour.graphics.editor_scene import PolygonEditorScene
from contour.graphics.polygon_creation import (
    POLYGON_COMMIT_INVALID_RING,
    POLYGON_COMMIT_TOO_FEW_VERTICES,
    POLYGON_COMMIT_TOO_SMALL_AREA,
    polygon_commit_acceptability,
)
from contour.graphics_items import EditablePolygonItem


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
        self.scene.set_vector_geometry_settings(
            VectorGeometrySettings(
                min_outer_area_px2=1.0,
                min_hole_area_to_remove_px2=0.0,
                drop_three_vertex_triangle_artifacts=False,
            )
        )

    def tearDown(self) -> None:
        self.scene.deleteLater()

    def _reset(self, initial: list[PolygonData] | None = None) -> None:
        self.scene.set_polygons([p.clone() for p in (initial or [])])

    def test_set_polygons_reuses_items_and_defers_surplus_cleanup(self) -> None:
        first = _polygon(1, [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)])
        second = _polygon(2, [(20.0, 0.0), (30.0, 0.0), (20.0, 10.0)])
        replacement = _polygon(9, [(40.0, 0.0), (50.0, 0.0), (40.0, 10.0)])
        self.scene.set_polygons([first, second])
        original_items = set(self.scene._polygon_items.values())

        self.scene.set_polygons([replacement])

        replacement_item = self.scene._polygon_items[9]
        self.assertIn(replacement_item, original_items)
        self.assertEqual(replacement_item.polygon_id, 9)
        self.assertEqual(len(self.scene._recycled_polygon_items), 1)
        self.assertTrue(self.scene._recycled_polygon_cleanup_timer.isActive())

        self.scene._drain_recycled_polygon_items()

        self.assertEqual(self.scene._recycled_polygon_items, [])
        self.assertFalse(self.scene._recycled_polygon_cleanup_timer.isActive())

    def test_set_polygons_paints_conductor_items_once(self) -> None:
        outer = _polygon(1, [(0.0, 0.0), (80.0, 0.0), (80.0, 80.0), (0.0, 80.0)])
        hole = _polygon(
            2,
            [(20.0, 20.0), (40.0, 20.0), (40.0, 40.0), (20.0, 40.0)],
            is_hole=True,
            parent_id=1,
        )
        with patch.object(
            EditablePolygonItem,
            "update_from_polygon",
            autospec=True,
            side_effect=EditablePolygonItem.update_from_polygon,
        ) as update:
            self.scene.set_polygons([outer, hole])

        self.assertEqual(update.call_count, 2)
        self.assertEqual(self.scene._polygon_items[1].polygon_id, 1)
        self.assertEqual(self.scene._polygon_items[2].polygon_id, 2)

    def test_set_polygons_can_skip_repair_geometry_scan(self) -> None:
        outer = _polygon(1, [(0.0, 0.0), (80.0, 0.0), (80.0, 80.0), (0.0, 80.0)])
        with patch(
            "contour.graphics.editor_scene.polygons_needing_repair",
            side_effect=AssertionError("frame switch must not scan repair geometry"),
        ):
            self.scene.set_polygons([outer], scan_repair=False)
        self.assertEqual(self.scene.polygons_needing_repair_map(), {})

    def test_set_polygons_applies_precomputed_repair_reasons(self) -> None:
        outer = _polygon(1, [(0.0, 0.0), (80.0, 0.0), (80.0, 80.0), (0.0, 80.0)])
        with patch(
            "contour.graphics.editor_scene.polygons_needing_repair",
            side_effect=AssertionError("precomputed repair must not rescan"),
        ):
            self.scene.set_polygons(
                [outer],
                repair_reasons={1: ["overlapping"]},
                scan_repair=False,
            )
        self.assertEqual(self.scene.polygons_needing_repair_map(), {1: ["overlapping"]})
        self.assertTrue(self.scene.polygon_needs_repair(1))

    def test_polygon_points_returns_empty_for_missing_id(self) -> None:
        self._reset([_polygon(2, [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)])])
        self.assertEqual(self.scene.polygon_points(1), [])
        self.assertFalse(self.scene.has_polygon(1))
        self.assertTrue(self.scene.has_polygon(2))

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

    def test_delete_vertex_merges_when_new_edge_crosses_existing(self) -> None:
        slot = _polygon(
            1,
            [
                (0.0, 0.0),
                (100.0, 0.0),
                (100.0, 20.0),
                (30.0, 20.0),
                (30.0, 40.0),
                (100.0, 40.0),
                (100.0, 60.0),
                (0.0, 60.0),
            ],
        )
        self._reset([slot])

        self.assertTrue(self.scene.delete_vertex_at(QPointF(0.0, 0.0), 2.0))
        remaining = self.scene.get_polygons()
        self.assertTrue(remaining)
        self.assertTrue(all(is_valid_closed_polygon_ring(polygon.points) for polygon in remaining))
        filled_area = sum(abs(float(polygon.area)) for polygon in remaining if not polygon.is_hole)
        self.assertGreater(filled_area, 1000.0)

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

    def test_delete_hole_merges_island_parented_to_hole(self) -> None:
        outer = _polygon(1, [(0.0, 0.0), (80.0, 0.0), (80.0, 80.0), (0.0, 80.0)])
        hole = _polygon(
            2,
            [(20.0, 20.0), (60.0, 20.0), (60.0, 60.0), (20.0, 60.0)],
            is_hole=True,
            parent_id=1,
        )
        island = _polygon(
            3,
            [(30.0, 30.0), (50.0, 30.0), (50.0, 50.0), (30.0, 50.0)],
            parent_id=2,
        )
        self._reset([outer, hole, island])

        self.assertTrue(self.scene.delete_polygon(2))

        remaining = self.scene.get_polygons()
        filled = [polygon for polygon in remaining if not polygon.is_hole]
        self.assertEqual(len(filled), 1)
        self.assertFalse(any(polygon.is_hole for polygon in remaining))

    def test_delete_hole_merges_independent_island_inside_hole(self) -> None:
        outer = _polygon(1, [(0.0, 0.0), (80.0, 0.0), (80.0, 80.0), (0.0, 80.0)])
        hole = _polygon(
            2,
            [(20.0, 20.0), (60.0, 20.0), (60.0, 60.0), (20.0, 60.0)],
            is_hole=True,
            parent_id=1,
        )
        island = _polygon(3, [(30.0, 30.0), (50.0, 30.0), (50.0, 50.0), (30.0, 50.0)])
        self._reset([outer, hole, island])

        self.assertTrue(self.scene.delete_polygon(2))

        remaining = self.scene.get_polygons()
        filled = [polygon for polygon in remaining if not polygon.is_hole]
        self.assertEqual(len(filled), 1)
        self.assertFalse(any(polygon.is_hole for polygon in remaining))

    def test_delete_hole_keeps_via_inside(self) -> None:
        outer = _polygon(1, [(0.0, 0.0), (80.0, 0.0), (80.0, 80.0), (0.0, 80.0)])
        hole = _polygon(
            2,
            [(20.0, 20.0), (60.0, 20.0), (60.0, 60.0), (20.0, 60.0)],
            is_hole=True,
            parent_id=1,
        )
        via = _polygon(
            3,
            [(30.0, 30.0), (50.0, 30.0), (50.0, 50.0), (30.0, 50.0)],
            parent_id=2,
        )
        via.category = "via"
        via.shape_hint = "box"
        self._reset([outer, hole, via])

        self.assertTrue(self.scene.delete_polygon(2))

        remaining = self.scene.get_polygons()
        vias = [polygon for polygon in remaining if polygon.category == "via"]
        filled = [polygon for polygon in remaining if polygon.category != "via" and not polygon.is_hole]
        self.assertEqual(len(vias), 1)
        self.assertEqual(len(filled), 1)
        self.assertFalse(any(polygon.is_hole for polygon in remaining))

    def test_delete_outer_removes_hole_and_nested_island(self) -> None:
        outer = _polygon(1, [(0.0, 0.0), (80.0, 0.0), (80.0, 80.0), (0.0, 80.0)])
        hole = _polygon(
            2,
            [(20.0, 20.0), (60.0, 20.0), (60.0, 60.0), (20.0, 60.0)],
            is_hole=True,
            parent_id=1,
        )
        island = _polygon(
            3,
            [(30.0, 30.0), (50.0, 30.0), (50.0, 50.0), (30.0, 50.0)],
            parent_id=2,
        )
        self._reset([outer, hole, island])

        self.assertTrue(self.scene.delete_polygon(1))
        self.assertEqual(self.scene.get_polygons(), [])

    def test_multi_delete_refreshes_and_emits_once_per_undo_operation(self) -> None:
        polygons = [
            _polygon(
                polygon_id,
                [
                    (float(polygon_id * 20), 0.0),
                    (float(polygon_id * 20 + 10), 0.0),
                    (float(polygon_id * 20 + 10), 10.0),
                    (float(polygon_id * 20), 10.0),
                ],
            )
            for polygon_id in range(1, 11)
        ]
        self._reset(polygons)
        self.scene.select_polygons([polygon.id for polygon in polygons])
        changed: list[None] = []
        self.scene.polygonsChanged.connect(lambda: changed.append(None))

        with patch.object(self.scene, "_refresh_all_items", wraps=self.scene._refresh_all_items) as refresh:
            self.assertTrue(self.scene.delete_polygon())
            self.assertEqual(refresh.call_count, 1)
            self.assertEqual(len(changed), 1)
            self.assertEqual(self.scene.get_polygons(), [])

            self.scene.undo_stack.undo()
            self.assertEqual(refresh.call_count, 2)
            self.assertEqual(len(changed), 2)
            self.assertEqual(len(self.scene.get_polygons()), len(polygons))

    def test_select_parent_polygon_shows_internal_contour_vertices(self) -> None:
        outer = _polygon(1, [(0.0, 0.0), (80.0, 0.0), (80.0, 80.0), (0.0, 80.0)])
        hole = _polygon(
            2,
            [(20.0, 20.0), (40.0, 20.0), (40.0, 40.0), (20.0, 40.0)],
            is_hole=True,
            parent_id=1,
        )
        self._reset([outer, hole])

        self.scene.select_polygon(1)

        self.assertEqual(len(self.scene._polygon_items[1]._handles), 4)
        self.assertEqual(len(self.scene._polygon_items[2]._handles), 4)

    def test_select_internal_contour_shows_parent_vertices(self) -> None:
        outer = _polygon(1, [(0.0, 0.0), (80.0, 0.0), (80.0, 80.0), (0.0, 80.0)])
        hole = _polygon(
            2,
            [(20.0, 20.0), (40.0, 20.0), (40.0, 40.0), (20.0, 40.0)],
            is_hole=True,
            parent_id=1,
        )
        self._reset([outer, hole])

        self.scene.select_polygon(2)

        self.assertEqual(len(self.scene._polygon_items[1]._handles), 4)
        self.assertEqual(len(self.scene._polygon_items[2]._handles), 4)

    def test_show_all_editable_vertices_draws_handles_on_unselected_polygons(self) -> None:
        first = _polygon(1, [(0.0, 0.0), (40.0, 0.0), (40.0, 40.0), (0.0, 40.0)])
        second = _polygon(2, [(50.0, 0.0), (90.0, 0.0), (90.0, 40.0), (50.0, 40.0)])
        via = _polygon(3, [(0.0, 50.0), (20.0, 50.0), (20.0, 70.0), (0.0, 70.0)])
        via.category = "via"
        via.shape_hint = "box"
        self._reset([first, second, via])

        self.assertEqual(len(self.scene._polygon_items[1]._handles), 0)
        self.scene.set_show_all_editable_vertices(True)

        self.assertEqual(len(self.scene._polygon_items[1]._handles), 4)
        self.assertEqual(len(self.scene._polygon_items[2]._handles), 4)
        self.assertEqual(len(self.scene._polygon_items[3]._handles), 0)

        self.scene.set_show_all_editable_vertices(False)
        self.assertEqual(len(self.scene._polygon_items[1]._handles), 0)

    def test_multi_selection_computes_editable_vertex_ids_once_for_full_refresh(self) -> None:
        polygons = [
            _polygon(
                polygon_id,
                [
                    (float(polygon_id * 20), 0.0),
                    (float(polygon_id * 20 + 10), 0.0),
                    (float(polygon_id * 20 + 10), 10.0),
                    (float(polygon_id * 20), 10.0),
                ],
            )
            for polygon_id in range(1, 11)
        ]
        self._reset(polygons)

        with patch.object(
            self.scene,
            "_editable_vertex_polygon_ids",
            wraps=self.scene._editable_vertex_polygon_ids,
        ) as editable_ids:
            self.scene.select_polygons([polygon.id for polygon in polygons])

        self.assertEqual(editable_ids.call_count, 1)

    def test_selected_via_uses_via_selection_color(self) -> None:
        via = _polygon(7, [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)])
        via.category = "via"
        self._reset([via])

        self.scene.select_polygon(7)

        self.assertEqual(self.scene._polygon_items[7].pen().color().name().upper(), "#FACC15")

    def test_hole_styling_survives_replacing_then_restoring_polygons(self) -> None:
        outer = _polygon(1, [(0.0, 0.0), (80.0, 0.0), (80.0, 80.0), (0.0, 80.0)])
        hole = _polygon(
            2,
            [(20.0, 20.0), (40.0, 20.0), (40.0, 40.0), (20.0, 40.0)],
            is_hole=True,
            parent_id=1,
        )
        self._reset([outer, hole])
        hole_color = self.scene._display_settings.hole_color.lower()
        self.assertTrue(self.scene.get_polygons()[1].is_hole)
        self.assertEqual(self.scene._polygon_items[2].pen().color().name().lower(), hole_color)
        self.assertTrue(self.scene._cutout_polygons_for(1))
        self.assertFalse(self.scene._polygon_items[1].contains(QPointF(30.0, 30.0)))
        self.assertEqual(self.scene._polygon_items[2].brush().color().alpha(), 0)

        self._reset([_polygon(1, [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])])
        self._reset([outer.clone(), hole.clone()])

        restored = {polygon.id: polygon for polygon in self.scene.get_polygons()}
        self.assertTrue(restored[2].is_hole)
        self.assertEqual(restored[2].parent_id, 1)
        self.assertEqual(self.scene._polygon_items[2].pen().color().name().lower(), hole_color)
        self.assertTrue(self.scene._cutout_polygons_for(1))
        self.assertFalse(self.scene._polygon_items[1].contains(QPointF(30.0, 30.0)))
        self.assertEqual(self.scene._polygon_items[2].brush().color().alpha(), 0)
        self.assertGreater(
            self.scene._polygon_items[2].zValue(),
            self.scene._polygon_items[1].zValue(),
        )

    def test_invalid_description_is_drawn_red_and_can_be_repaired(self) -> None:
        bowtie = _polygon(1, [(0.0, 0.0), (40.0, 40.0), (40.0, 0.0), (0.0, 40.0)])
        self._reset([bowtie])

        item = self.scene._polygon_items[1]
        self.assertEqual(item.pen().color().name().upper(), "#DC2626")
        self.assertTrue(item.toolTip())

        self.assertTrue(self.scene.repair_invalid_polygon_descriptions())
        self.assertFalse(any(polygon.description_is_invalid() for polygon in self.scene.get_polygons()))
        for item in self.scene._polygon_items.values():
            self.assertNotEqual(item.pen().color().name().upper(), "#DC2626")

    def test_overlapping_and_small_geometry_are_drawn_red_and_repaired(self) -> None:
        self.scene.set_vector_geometry_settings(
            VectorGeometrySettings(
                min_outer_area_px2=50.0,
                min_hole_area_to_remove_px2=100.0,
                drop_three_vertex_triangle_artifacts=False,
            )
        )
        a = _polygon(1, [(0.0, 0.0), (70.0, 0.0), (70.0, 70.0), (0.0, 70.0)])
        b = _polygon(2, [(35.0, 35.0), (95.0, 35.0), (95.0, 95.0), (35.0, 95.0)])
        tiny = _polygon(3, [(200.0, 200.0), (203.0, 200.0), (203.0, 203.0), (200.0, 203.0)])
        self._reset([a, b, tiny])

        self.assertEqual(self.scene._polygon_items[1].pen().color().name().upper(), "#DC2626")
        self.assertEqual(self.scene._polygon_items[2].pen().color().name().upper(), "#DC2626")
        self.assertEqual(self.scene._polygon_items[3].pen().color().name().upper(), "#DC2626")
        self.assertTrue(self.scene.repair_invalid_polygon_descriptions())
        remaining = self.scene.get_polygons()
        self.assertEqual(len([polygon for polygon in remaining if not polygon.is_hole]), 1)
        for item in self.scene._polygon_items.values():
            self.assertNotEqual(item.pen().color().name().upper(), "#DC2626")

    def test_invalid_polygon_not_committed_keeps_pending(self) -> None:
        self._reset([])
        self.scene.append_pending_point(QPointF(10.0, 10.0))
        self.scene.append_pending_point(QPointF(20.0, 20.0))
        self.scene.append_pending_point(QPointF(15.0, 15.0))
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
        self.scene.append_pending_point(QPointF(20.0, 20.0))
        self.scene.append_pending_point(QPointF(15.0, 15.0))

        self.assertEqual(self.scene._pending_path_item.brush().color().name().lower(), "#ef4444")

    def test_pending_polyline_drops_axis_aligned_extra_vertex(self) -> None:
        self._reset([])
        self.scene.append_pending_point(QPointF(10.0, 10.0))
        self.scene.append_pending_point(QPointF(20.0, 10.0))
        self.scene.append_pending_point(QPointF(30.0, 10.0))
        self.assertEqual(self.scene.pending_points_snapshot(), [(10.0, 10.0), (30.0, 10.0)])

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

    def test_trace_polyline_may_cross_itself(self) -> None:
        self._reset([])
        self.scene.start_pending_polygon(for_brush=True)
        self.scene.append_pending_point(QPointF(0.0, 0.0))
        self.scene.append_pending_point(QPointF(20.0, 20.0))
        self.scene.append_pending_point(QPointF(20.0, 0.0))
        self.scene.append_pending_point(QPointF(0.0, 20.0))
        self.assertEqual(
            self.scene.pending_points_snapshot(),
            [(0.0, 0.0), (20.0, 20.0), (20.0, 0.0), (0.0, 20.0)],
        )

    def test_closed_polygon_polyline_still_rejects_self_intersection(self) -> None:
        self._reset([])
        self.scene.append_pending_point(QPointF(0.0, 0.0))
        self.scene.append_pending_point(QPointF(20.0, 20.0))
        self.scene.append_pending_point(QPointF(20.0, 0.0))
        self.scene.append_pending_point(QPointF(0.0, 20.0))
        self.assertEqual(
            self.scene.pending_points_snapshot(),
            [(0.0, 0.0), (20.0, 20.0), (20.0, 0.0)],
        )

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


class PolygonOverlapPickTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def setUp(self) -> None:
        self.scene = PolygonEditorScene()

    def tearDown(self) -> None:
        self.scene.deleteLater()

    def test_keyhole_frame_does_not_block_inner_conductor_pick(self) -> None:
        """Self-intersecting frame hull must not steal picks from nested conductors."""
        from pathlib import Path

        from contour.serializers import clear_cif_parse_cache, load_polygons_cif

        cif_path = Path(r"D:\OZI\Нейронка\cif_metal\0516.cif")
        if not cif_path.exists():
            self.skipTest("0516.cif sample is not available")
        clear_cif_parse_cache()
        _, _, polygons = load_polygons_cif(cif_path)
        frame = max(
            (polygon for polygon in polygons if not polygon.is_hole),
            key=lambda polygon: polygon.area,
        )
        frame_holes = [polygon for polygon in polygons if polygon.parent_id == frame.id]
        center = QPointF(1146.0, 1471.0)
        inner = next(
            polygon
            for polygon in polygons
            if not polygon.is_hole
            and polygon.id != frame.id
            and polygon.bbox[0] <= center.x() <= polygon.bbox[0] + polygon.bbox[2]
            and polygon.bbox[1] <= center.y() <= polygon.bbox[1] + polygon.bbox[3]
        )
        self.scene.set_polygons([frame, *frame_holes, inner])

        hits = self.scene.polygons_at(center)
        self.assertIn(inner.id, hits)
        self.assertNotIn(frame.id, hits)

    def test_repeated_click_cycles_overlapping_conductors(self) -> None:
        small = _polygon(1, [(40.0, 40.0), (90.0, 40.0), (90.0, 90.0), (40.0, 90.0)])
        large = _polygon(2, [(0.0, 0.0), (120.0, 0.0), (120.0, 120.0), (0.0, 120.0)])
        self.scene.set_polygons([large, small])
        overlap = QPointF(65.0, 65.0)
        self.assertEqual(self.scene.polygons_at(overlap), [1, 2])
        first = self.scene.polygon_at(overlap, cycle=True)
        second = self.scene.polygon_at(overlap, cycle=True)
        self.assertNotEqual(first, second)
        self.assertEqual({first, second}, {1, 2})

    def test_editing_hole_discards_parent_authored_cif_paint_ring(self) -> None:
        outer = _polygon(1, [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)])
        outer.cif_paint_ring = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 40.0),
            (70.0, 40.0),
            (70.0, 70.0),
            (40.0, 70.0),
            (40.0, 40.0),
            (100.0, 40.0),
            (100.0, 100.0),
            (0.0, 100.0),
        ]
        hole = _polygon(2, [(40.0, 40.0), (40.0, 70.0), (70.0, 70.0), (70.0, 40.0)])
        hole.is_hole = True
        hole.parent_id = outer.id
        self.scene.set_polygons([outer, hole])

        self.scene._replace_polygon_points_internal(
            hole.id,
            [(42.0, 40.0), (40.0, 70.0), (70.0, 70.0), (70.0, 40.0)],
        )

        by_id = {polygon.id: polygon for polygon in self.scene.get_polygons()}
        self.assertEqual(by_id[outer.id].cif_paint_ring, [])
        self.assertEqual(by_id[hole.id].cif_paint_ring, [])


if __name__ == "__main__":
    unittest.main()
