from __future__ import annotations

import unittest

from contour.domain import PolygonData, compute_polygon_metrics
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

    def test_is_valid_closed_accepts_convex_square(self) -> None:
        sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        self.assertTrue(is_valid_closed_polygon_ring(sq))

    def test_open_polyline_rejects_segment_crossing_prior_edge(self) -> None:
        pts = [(0.0, 0.0), (2.0, 0.0), (1.0, 0.5), (1.0, -0.5)]
        self.assertFalse(is_valid_open_polyline_last_edge(pts))

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


if __name__ == "__main__":
    unittest.main()
