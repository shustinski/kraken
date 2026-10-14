"""test_cif_via_support: boundary vertex snap tests."""
from __future__ import annotations
import unittest

class BoundaryVertexSnapTests(unittest.TestCase):
    def test_snap_picks_nearest_boundary_vertex_by_x_window(self) -> None:
        from contour.serializers import _snap_hole_to_boundary_vertex

        outer = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        snap = _snap_hole_to_boundary_vertex(outer, (48.0, 32.0))
        self.assertIsNotNone(snap)
        outer_anchor, boundary_index = snap
        self.assertEqual(outer_anchor, (100.0, 0.0))
        self.assertEqual(boundary_index, 1)

    def test_snap_searches_x_sorted_neighbors_outside_hull(self) -> None:
        from contour.serializers import _snap_hole_to_boundary_vertex

        outer = [(0.0, 0.0), (40.0, 0.0), (40.0, 100.0), (0.0, 100.0)]
        snap = _snap_hole_to_boundary_vertex(outer, (60.0, 50.0))
        self.assertIsNotNone(snap)
        outer_anchor, _boundary_index = snap
        self.assertEqual(outer_anchor, (40.0, 0.0))

    def test_snap_returns_none_for_degenerate_boundary(self) -> None:
        from contour.serializers import _snap_hole_to_boundary_vertex

        snap = _snap_hole_to_boundary_vertex([(0.0, 0.0), (10.0, 0.0)], (5.0, 5.0))
        self.assertIsNone(snap)

    def test_snap_prefers_closest_right_side_vertex_in_local_x_window(self) -> None:
        from contour.serializers import _snap_hole_to_boundary_vertex

        outer = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        snap = _snap_hole_to_boundary_vertex(outer, (10.0, 10.0))
        self.assertIsNotNone(snap)
        outer_anchor, _boundary_index = snap
        self.assertEqual(outer_anchor, (100.0, 0.0))

    def test_vertex_slit_can_attach_to_linked_neighbor_hole(self) -> None:
        from shapely import make_valid
        from shapely.geometry import LineString
        from shapely.geometry import Polygon as ShapelyPolygon

        from contour.serializers import _ring_with_orientation, _select_vertex_slit_for_hole

        outer = _ring_with_orientation(
            [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
            positive_area=True,
        )
        right_hole = _ring_with_orientation(
            [(80.0, 50.0), (100.0, 50.0), (100.0, 70.0), (80.0, 70.0)],
            positive_area=False,
        )
        center_hole = _ring_with_orientation(
            [(50.0, 50.0), (70.0, 50.0), (70.0, 70.0), (50.0, 70.0)],
            positive_area=False,
        )
        parent_solid = make_valid(ShapelyPolygon(outer, holes=[right_hole, center_hole]))

        outer_only = _select_vertex_slit_for_hole(
            center_hole,
            parent_solid,
            outer=outer,
            linked_hole_rings=[],
        )
        self.assertIsNotNone(outer_only)
        self.assertEqual(outer_only[0], -1)

        neighbor = _select_vertex_slit_for_hole(
            center_hole,
            parent_solid,
            outer=outer,
            linked_hole_rings=[(0, right_hole)],
        )
        self.assertIsNotNone(neighbor)
        self.assertEqual(neighbor[0], 0)
        self.assertTrue(
            parent_solid.covers(LineString([neighbor[1], center_hole[neighbor[3]]])),
        )
        self.assertLess(
            (neighbor[1][0] - center_hole[neighbor[3]][0]) ** 2
            + (neighbor[1][1] - center_hole[neighbor[3]][1]) ** 2,
            2500.0,
        )
