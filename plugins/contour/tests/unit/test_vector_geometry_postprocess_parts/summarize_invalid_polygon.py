"""test_vector_geometry_postprocess: test summarize invalid polygon."""
from __future__ import annotations
from contour.application.vector_geometry_postprocess import (
    VectorGeometrySettings,
    apply_vertex_position_to_clone,
    polygon_description_is_invalid,
    postprocess_changed_polygon_edit,
    postprocess_changed_polygon_only,
    repair_invalid_polygon_descriptions,
    summarize_invalid_polygon_description_reasons,
)
from contour.domain import PolygonData

from test_vector_geometry_postprocess_parts.rect import _rect


class SummarizeInvalidPolygonMixin:
    def test_vertex_move_preserves_small_inner_hole_target(self) -> None:
        outer = _rect(0.0, 0.0, 100.0, 100.0, 1)
        hole = _rect(40.0, 40.0, 42.0, 42.0, 2)
        hole.is_hole = True
        hole.parent_id = outer.id
        moved = apply_vertex_position_to_clone([outer, hole], 2, 1, (43.0, 40.0))

        processed, changed = postprocess_changed_polygon_only(
            moved,
            VectorGeometrySettings(min_hole_area_to_remove_px2=11.0, min_spike_interior_angle_deg=0.0),
            polygon_id=2,
        )

        self.assertFalse(changed)
        moved_hole = next((polygon for polygon in processed if polygon.id == 2), None)
        self.assertIsNotNone(moved_hole)
        self.assertTrue(moved_hole.is_hole)
        self.assertEqual(moved_hole.points[1], (43, 40))

    def test_postprocess_changed_polygon_edit_rejects_too_small_outer(self) -> None:
        tiny = _rect(0.0, 0.0, 2.0, 2.0, 1)
        processed, accepted, changed = postprocess_changed_polygon_edit(
            [tiny],
            VectorGeometrySettings(min_outer_area_px2=50.0, min_spike_interior_angle_deg=0.0),
            polygon_id=1,
        )
        self.assertFalse(accepted)
        self.assertFalse(changed)
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0].points, tiny.points)

    def test_postprocess_changed_polygon_only_keeps_polygon_when_filters_fail(self) -> None:
        tiny = _rect(0.0, 0.0, 2.0, 2.0, 1)
        processed, changed = postprocess_changed_polygon_only(
            [tiny],
            VectorGeometrySettings(min_outer_area_px2=50.0, min_spike_interior_angle_deg=0.0),
            polygon_id=1,
        )
        self.assertFalse(changed)
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0].points, tiny.points)

    def test_via_and_box_rings_are_not_flagged_as_invalid_descriptions(self) -> None:
        via = _rect(0.0, 0.0, 8.0, 8.0, 1)
        via.category = "via"
        via.shape_hint = "box"
        via.points = [(0.0, 0.0), (8.0, 8.0), (8.0, 0.0), (0.0, 8.0)]
        self.assertFalse(polygon_description_is_invalid(via))

    def test_repair_invalid_polygon_descriptions_splits_keyhole(self) -> None:
        keyhole = PolygonData(
            id=1,
            points=[
                (0.0, 0.0),
                (80.0, 0.0),
                (80.0, 80.0),
                (0.0, 80.0),
                (0.0, 40.0),
                (40.0, 40.0),
                (40.0, 50.0),
                (20.0, 50.0),
                (20.0, 30.0),
                (40.0, 30.0),
                (40.0, 40.0),
                (0.0, 40.0),
            ],
        )
        self.assertFalse(polygon_description_is_invalid(keyhole))
        self.assertEqual(keyhole.description_invalid_reason(), "repeated_vertex")
        repaired = repair_invalid_polygon_descriptions([keyhole])
        self.assertTrue(repaired)
        self.assertFalse(any(polygon_description_is_invalid(polygon) for polygon in repaired))
        self.assertTrue(any(polygon.is_hole for polygon in repaired))
        self.assertTrue(any(not polygon.is_hole for polygon in repaired))

    def test_repair_invalid_polygon_descriptions_splits_bowtie(self) -> None:
        bowtie = PolygonData(
            id=1,
            points=[(0.0, 0.0), (40.0, 40.0), (40.0, 0.0), (0.0, 40.0)],
        )
        self.assertTrue(polygon_description_is_invalid(bowtie))
        repaired = repair_invalid_polygon_descriptions([bowtie])
        self.assertGreaterEqual(len(repaired), 2)
        self.assertFalse(any(polygon_description_is_invalid(polygon) for polygon in repaired))
        self.assertTrue(all(not polygon.is_hole for polygon in repaired))

    def test_repair_invalid_polygon_descriptions_is_noop_for_simple_rings(self) -> None:
        square = _rect(0.0, 0.0, 40.0, 40.0, 1)
        repaired = repair_invalid_polygon_descriptions([square])
        self.assertEqual(len(repaired), 1)
        self.assertEqual(repaired[0].points, square.points)
        self.assertFalse(polygon_description_is_invalid(repaired[0]))

    def test_summarize_invalid_polygon_description_reasons(self) -> None:
        keyhole = PolygonData(
            id=1,
            points=[
                (0.0, 0.0),
                (80.0, 0.0),
                (80.0, 80.0),
                (0.0, 80.0),
                (0.0, 40.0),
                (40.0, 40.0),
                (40.0, 50.0),
                (20.0, 50.0),
                (20.0, 30.0),
                (40.0, 30.0),
                (40.0, 40.0),
                (0.0, 40.0),
            ],
        )
        bowtie = PolygonData(
            id=2,
            points=[(0.0, 0.0), (40.0, 40.0), (40.0, 0.0), (0.0, 40.0)],
        )
        square = _rect(100.0, 100.0, 120.0, 120.0, 3)
        self.assertEqual(
            summarize_invalid_polygon_description_reasons([keyhole, bowtie, square, keyhole]),
            [("repeated_vertex", 2), ("self_intersecting", 1)],
        )
