from __future__ import annotations

import unittest

from unittest.mock import patch

from contour.domain import PolygonData, compute_polygon_metrics
from contour.domain.polygon_ring import (
    TOPOLOGY_CHECK_MAX_VERTICES,
    closed_ring_description_invalid_reason,
    closed_ring_description_is_invalid,
    collapse_redundant_polyline_vertices,
    is_valid_closed_polygon_vertex_move,
)
from contour.graphics.geometry import (
    is_valid_closed_polygon_ring,
    is_valid_open_polyline_last_edge,
    resolve_conductor_hover_target_id,
)


class GeometryTests(unittest.TestCase):
    def test_compute_polygon_metrics_for_rectangle(self) -> None:
        area, perimeter, bbox = compute_polygon_metrics(
            [
                (10.0, 5.0),
                (18.0, 5.0),
                (18.0, 11.0),
                (10.0, 11.0),
            ]
        )

        self.assertEqual(area, 48.0)
        self.assertEqual(perimeter, 28.0)
        self.assertEqual(bbox, (10, 5, 9, 7))

    def test_compute_polygon_metrics_for_segment(self) -> None:
        area, perimeter, bbox = compute_polygon_metrics([(1.2, 3.4), (4.8, 3.4)])

        self.assertEqual(area, 0.0)
        self.assertAlmostEqual(perimeter, 7.2)
        self.assertEqual(bbox, (1, 3, 4, 1))

    def test_is_valid_closed_rejects_bowtie(self) -> None:
        bow = [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]
        self.assertFalse(is_valid_closed_polygon_ring(bow))
        self.assertTrue(closed_ring_description_is_invalid(bow))
        self.assertEqual(closed_ring_description_invalid_reason(bow), "self_intersecting")

    def test_closed_ring_description_flags_keyhole_repeated_vertex(self) -> None:
        keyhole = [
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
        ]
        # CIF keyhole bridges are diagnosable but not an invalid description.
        self.assertFalse(closed_ring_description_is_invalid(keyhole))
        self.assertEqual(closed_ring_description_invalid_reason(keyhole), "repeated_vertex")
        square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        self.assertFalse(closed_ring_description_is_invalid(square))
        self.assertIsNone(closed_ring_description_invalid_reason(square))

    def test_is_valid_closed_accepts_convex_square(self) -> None:
        sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        self.assertTrue(is_valid_closed_polygon_ring(sq))

    def test_vertex_move_validation_ignores_unrelated_existing_defect(self) -> None:
        pts = [(0.0, 0.0), (45.0, 0.0), (40.0, 40.0), (10.0, 10.0), (0.0, 40.0), (10.0, 10.0)]
        self.assertFalse(is_valid_closed_polygon_ring(pts))
        self.assertTrue(is_valid_closed_polygon_vertex_move(pts, 1))

    def test_large_dense_degenerate_ring_is_not_accepted_without_check(self) -> None:
        n = TOPOLOGY_CHECK_MAX_VERTICES + 50
        ring = [(float(i), 0.0) for i in range(n)]
        self.assertFalse(is_valid_closed_polygon_ring(ring))

    def test_vertex_move_validation_rejects_moved_edge_crossing(self) -> None:
        pts = [(0.0, 0.0), (20.0, 60.0), (40.0, 0.0), (0.0, 40.0)]
        self.assertFalse(is_valid_closed_polygon_vertex_move(pts, 1))

    def test_open_polyline_rejects_segment_crossing_prior_edge(self) -> None:
        pts = [(0.0, 0.0), (2.0, 0.0), (1.0, 0.5), (1.0, -0.5)]
        self.assertFalse(is_valid_open_polyline_last_edge(pts))

    def test_collapse_drops_duplicate_neighbors(self) -> None:
        points = [(0.0, 0.0), (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        self.assertEqual(
            collapse_redundant_polyline_vertices(points),
            [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        )

    def test_collapse_drops_axis_aligned_middle_vertex(self) -> None:
        points = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        self.assertEqual(
            collapse_redundant_polyline_vertices(points),
            [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        )

    def test_collapse_drops_vertical_middle_and_wrap_around(self) -> None:
        points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 5.0)]
        collapsed = collapse_redundant_polyline_vertices(points)
        self.assertNotIn((0.0, 5.0), collapsed)
        self.assertEqual(len(collapsed), 4)

    def test_collapse_keeps_diagonal_middle_vertex(self) -> None:
        points = [(0.0, 0.0), (5.0, 4.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        self.assertEqual(collapse_redundant_polyline_vertices(points), points)

    def test_resolve_conductor_hover_outer_trace(self) -> None:
        outer = PolygonData(
            id=1,
            points=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
            category="conductor",
            shape_hint="polygon",
            area=10000.0,
            bbox=(0, 0, 100, 100),
        )
        registry = {1: outer}
        self.assertEqual(resolve_conductor_hover_target_id(registry, 1), 1)

    def test_resolve_conductor_hover_hole_mapped_to_parent(self) -> None:
        outer = PolygonData(
            id=1,
            points=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
            is_hole=False,
            category="conductor",
            shape_hint="polygon",
            area=10000.0,
            bbox=(0, 0, 100, 100),
        )
        hole = PolygonData(
            id=2,
            points=[(40.0, 40.0), (60.0, 40.0), (60.0, 60.0), (40.0, 60.0)],
            is_hole=True,
            parent_id=1,
            category="conductor",
            shape_hint="polygon",
            area=400.0,
            bbox=(40, 40, 21, 21),
        )
        registry = {1: outer, 2: hole}
        self.assertEqual(resolve_conductor_hover_target_id(registry, 2), 1)

    def test_resolve_conductor_hover_via_inside_trace(self) -> None:
        outer = PolygonData(
            id=1,
            points=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
            is_hole=False,
            category="conductor",
            shape_hint="polygon",
            area=10000.0,
            bbox=(0, 0, 100, 100),
        )
        via = PolygonData(
            id=3,
            points=[(45.0, 45.0), (55.0, 45.0), (55.0, 55.0), (45.0, 55.0)],
            is_hole=False,
            parent_id=None,
            category="via",
            shape_hint="box",
            area=100.0,
            bbox=(45, 45, 11, 11),
        )
        registry = {1: outer, 3: via}
        self.assertEqual(resolve_conductor_hover_target_id(registry, 3), 1)

    def test_description_is_invalid_cache_survives_clone_and_clears_on_point_replace(self) -> None:
        bowtie = PolygonData(
            id=1,
            points=[(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)],
        )
        square = PolygonData(
            id=2,
            points=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        )

        self.assertTrue(bowtie.description_is_invalid())
        self.assertEqual(bowtie.description_invalid_reason(), "self_intersecting")
        self.assertFalse(square.description_is_invalid())
        self.assertIsNone(square.description_invalid_reason())
        with patch(
            "contour.domain.polygon_ring.closed_ring_description_invalid_reason",
            side_effect=AssertionError("topology check should use the cached result"),
        ):
            self.assertTrue(bowtie.description_is_invalid())
            self.assertEqual(bowtie.description_invalid_reason(), "self_intersecting")
            self.assertTrue(bowtie.clone().description_is_invalid())
            self.assertEqual(bowtie.clone().description_invalid_reason(), "self_intersecting")
            self.assertFalse(square.description_is_invalid())
            self.assertIsNone(square.description_invalid_reason())
            self.assertFalse(square.clone().description_is_invalid())

        bowtie.points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        self.assertFalse(bowtie.description_is_invalid())
        self.assertIsNone(bowtie.description_invalid_reason())

    def test_warmed_description_cache_survives_worker_style_clone_chain(self) -> None:
        bowtie = PolygonData(
            id=1,
            points=[(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)],
        )
        bowtie.description_invalid_reason()
        stored = [bowtie.clone() for _ in range(2)]
        with patch(
            "contour.domain.polygon_ring.closed_ring_description_invalid_reason",
            side_effect=AssertionError("topology check should use the cached result"),
        ):
            self.assertTrue(all(polygon.description_is_invalid() for polygon in stored))
            self.assertTrue(all(polygon.description_invalid_reason() == "self_intersecting" for polygon in stored))


if __name__ == "__main__":
    unittest.main()
