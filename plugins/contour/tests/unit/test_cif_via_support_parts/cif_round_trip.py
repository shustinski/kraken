"""test_cif_via_support: test cif round trip."""
from __future__ import annotations
from contour.domain import PolygonData, compute_polygon_metrics
from contour.serializers import (
    load_polygons_cif,
    save_polygons_cif,
)

from test_cif_via_support_parts.lib_open_cif import _rectangle_polygon


class CifRoundTripMixin:
    def test_cif_round_trip_restores_outer_hole_topology_by_default(self) -> None:
        outer_points = [(10.0, 10.0), (70.0, 10.0), (70.0, 70.0), (10.0, 70.0)]
        inner_points = [(32.0, 32.0), (48.0, 32.0), (48.0, 48.0), (32.0, 48.0)]
        outer_area, outer_perimeter, outer_bbox = compute_polygon_metrics(outer_points)
        inner_area, inner_perimeter, inner_bbox = compute_polygon_metrics(inner_points)
        outer = PolygonData(
            id=1,
            points=outer_points,
            is_hole=False,
            parent_id=None,
            category="conductor",
            shape_hint="polygon",
            area=outer_area,
            perimeter=outer_perimeter,
            bbox=outer_bbox,
        )
        hole = PolygonData(
            id=2,
            points=inner_points,
            is_hole=True,
            parent_id=1,
            category="conductor",
            shape_hint="polygon",
            area=inner_area,
            perimeter=inner_perimeter,
            bbox=inner_bbox,
        )
        cif_path = self._artifact_path("round_trip_hole.cif")
        save_polygons_cif(cif_path, "sample.png", [outer, hole], image_size=(80, 80))
        _image_name, _image_size, loaded = load_polygons_cif(cif_path)
        loaded_outers = [polygon for polygon in loaded if not polygon.is_hole]
        loaded_holes = [polygon for polygon in loaded if polygon.is_hole]

        self.assertEqual(len(loaded_outers), 1)
        self.assertEqual(len(loaded_holes), 1)
        self.assertEqual(loaded_holes[0].parent_id, loaded_outers[0].id)
        self.assertEqual(set(loaded_outers[0].points), set(outer.points))
        self.assertEqual(set(loaded_holes[0].points), set(hole.points))
        self.assertAlmostEqual(
            loaded_outers[0].area - loaded_holes[0].area,
            outer.area - hole.area,
        )
        self.assertFalse(any(polygon.description_is_invalid() for polygon in loaded))

    def test_cif_round_trip_restores_hole_topology_from_standard_cutline(self) -> None:
        outer_points = [(10.0, 10.0), (70.0, 10.0), (70.0, 70.0), (10.0, 70.0)]
        inner_points = [(32.0, 32.0), (48.0, 32.0), (48.0, 48.0), (32.0, 48.0)]
        outer_area, outer_perimeter, outer_bbox = compute_polygon_metrics(outer_points)
        inner_area, inner_perimeter, inner_bbox = compute_polygon_metrics(inner_points)
        outer = PolygonData(
            id=1,
            points=outer_points,
            is_hole=False,
            parent_id=None,
            category="conductor",
            shape_hint="polygon",
            area=outer_area,
            perimeter=outer_perimeter,
            bbox=outer_bbox,
        )
        hole = PolygonData(
            id=2,
            points=inner_points,
            is_hole=True,
            parent_id=1,
            category="conductor",
            shape_hint="polygon",
            area=inner_area,
            perimeter=inner_perimeter,
            bbox=inner_bbox,
        )
        cif_path = self._artifact_path("round_trip_hole_recover.cif")
        save_polygons_cif(cif_path, "sample.png", [outer, hole], image_size=(80, 80))
        _image_name, _image_size, loaded = load_polygons_cif(cif_path)
        self.assertTrue(any(not polygon.is_hole for polygon in loaded))
        self.assertTrue(any(polygon.is_hole for polygon in loaded))
        hole_parent_ids = {polygon.parent_id for polygon in loaded if polygon.is_hole}
        self.assertTrue(any(parent_id is not None for parent_id in hole_parent_ids))

    def test_cif_saves_polygon_holes_as_single_linked_polygon(self) -> None:
        outer_points = [(10.0, 10.0), (70.0, 10.0), (70.0, 70.0), (10.0, 70.0)]
        inner_points = [(32.0, 32.0), (48.0, 32.0), (48.0, 48.0), (32.0, 48.0)]
        outer_area, outer_perimeter, outer_bbox = compute_polygon_metrics(outer_points)
        inner_area, inner_perimeter, inner_bbox = compute_polygon_metrics(inner_points)
        outer = PolygonData(
            id=3,
            points=outer_points,
            category="conductor",
            shape_hint="polygon",
            area=outer_area,
            perimeter=outer_perimeter,
            bbox=outer_bbox,
        )
        hole = PolygonData(
            id=4,
            points=inner_points,
            is_hole=True,
            parent_id=3,
            category="conductor",
            shape_hint="polygon",
            area=inner_area,
            perimeter=inner_perimeter,
            bbox=inner_bbox,
        )
        cif_path = self._artifact_path("linked_polygon_hole.cif")

        save_polygons_cif(cif_path, "sample.png", [outer, hole], image_size=(80, 80))
        payload = cif_path.read_text(encoding="utf-8")
        _image_name, _image_size, loaded = load_polygons_cif(cif_path)

        self.assertNotIn("CONTOUR", payload)
        self.assertEqual(payload.count("\nP "), 1)
        self.assertTrue(payload.strip().endswith("E"))
        self.assertEqual(len([polygon for polygon in loaded if not polygon.is_hole]), 1)
        self.assertEqual(len([polygon for polygon in loaded if polygon.is_hole]), 1)
        self.assertFalse(any(polygon.description_is_invalid() for polygon in loaded))

    def test_cif_keyhole_round_trip_preserves_multiple_blocking_holes(self) -> None:
        from shapely import make_valid
        from shapely.geometry import Polygon as ShapelyPolygon

        outer = _rectangle_polygon(0, 0, 120, 120)
        hole_bounds = [
            (50, 50, 70, 70),
            (50, 20, 70, 40),
            (50, 80, 70, 100),
            (20, 50, 40, 70),
            (80, 50, 100, 70),
        ]
        holes: list[PolygonData] = []
        for polygon_id, bounds in enumerate(hole_bounds, start=2):
            hole = _rectangle_polygon(*bounds)
            hole.id = polygon_id
            hole.is_hole = True
            hole.parent_id = outer.id
            holes.append(hole)

        cif_path = self._artifact_path("multi_hole_keyhole.cif")
        save_polygons_cif(cif_path, "sample.png", [outer, *holes], image_size=(120, 120))
        _image_name, _image_size, loaded = load_polygons_cif(cif_path)

        loaded_outer = next(polygon for polygon in loaded if not polygon.is_hole)
        loaded_holes = [polygon for polygon in loaded if polygon.parent_id == loaded_outer.id]
        expected = make_valid(ShapelyPolygon(outer.points, holes=[hole.points for hole in holes]))
        actual = make_valid(
            ShapelyPolygon(
                loaded_outer.points,
                holes=[hole.points for hole in loaded_holes],
            )
        )
        payload = cif_path.read_text(encoding="utf-8")

        self.assertEqual(payload.count("\nP "), 1)
        self.assertNotIn("CONTOUR", payload)
        self.assertEqual(len(loaded_holes), len(holes))
        self.assertAlmostEqual(float(expected.symmetric_difference(actual).area), 0.0)
