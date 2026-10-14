"""test_cif_via_support: test cv saves polygon."""
from __future__ import annotations
import json
from contour.domain import PolygonData, compute_polygon_metrics
from contour.serializers import (
    load_polygons_cv,
    load_polygons_vector,
    save_polygons_cv,
)

from test_cif_via_support_parts.lib_open_cif import _rectangle_polygon


class CvSavesPolygonMixin:
    def test_cv_round_trip_saves_via_as_ellipse_point(self) -> None:
        polygon = _rectangle_polygon(10, 12, 22, 28)
        polygon.category = "via"
        polygon.shape_hint = "box"
        cv_path = self._artifact_path("sample_via.cv")

        save_polygons_cv(cv_path, "sample.png", [polygon], image_size=(64, 48))
        payload = cv_path.read_text(encoding="utf-8")
        image_name, image_size, loaded = load_polygons_cv(cv_path)

        self.assertIn('"type": "Point"', payload)
        self.assertIn('"shape": "ellipse"', payload)
        self.assertIn('"diagonals"', payload)
        self.assertNotIn('"features"', payload)
        self.assertNotIn('"properties"', payload)
        self.assertNotIn('"metadata"', payload)
        self.assertEqual(image_name, "sample.png")
        self.assertEqual(image_size, (64, 48))
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].category, "via")
        self.assertEqual(loaded[0].shape_hint, "box")
        self.assertEqual(loaded[0].bbox, polygon.bbox)

    def test_cv_via_round_trip_preserves_odd_size_and_position_exactly(self) -> None:
        polygon = _rectangle_polygon(11, 13, 22, 22)
        polygon.category = "via"
        polygon.shape_hint = "box"
        cv_path = self._artifact_path("odd_via_box.cv")

        save_polygons_cv(cv_path, "sample.png", [polygon], image_size=(100, 80))
        _image_name, _image_size, loaded = load_polygons_cv(cv_path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].points, polygon.points)
        self.assertEqual(loaded[0].bbox, polygon.bbox)

    def test_cv_loader_reads_rectangle_point(self) -> None:
        cv_path = self._artifact_path("sample_rectangle.cv")
        cv_path.write_text(
            """
{
  "format": "contour-vector",
  "image": {"path": "sample.png", "size": [100, 80]},
  "objects": [
    {
      "type": "Point",
      "id": 7,
      "shape": "rectangle",
      "coordinates": [4, 5, 14, 25]
    }
  ]
}
""".strip(),
            encoding="utf-8",
        )

        image_name, image_size, polygons = load_polygons_vector(cv_path)

        self.assertEqual(image_name, "sample.png")
        self.assertEqual(image_size, (100, 80))
        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0].id, 7)
        self.assertEqual(polygons[0].category, "conductor")
        self.assertEqual(polygons[0].shape_hint, "box")
        self.assertEqual(polygons[0].bbox, (4, 5, 11, 21))

    def test_cv_saves_polygon_holes_as_coordinate_rings(self) -> None:
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
        cv_path = self._artifact_path("compact_polygon_hole.cv")

        save_polygons_cv(cv_path, "sample.png", [outer, hole], image_size=(80, 80))
        payload = json.loads(cv_path.read_text(encoding="utf-8"))
        _image_name, _image_size, loaded = load_polygons_cv(cv_path)

        self.assertEqual(payload["objects"][0]["type"], "Polygon")
        self.assertEqual(payload["objects"][0]["id"], 3)
        self.assertEqual(len(payload["objects"][0]["coordinates"]), 2)
        self.assertEqual(payload["objects"][0]["coordinates"][0][0], [10, 10])
        self.assertEqual(payload["objects"][0]["coordinates"][0][-1], [10, 10])
        self.assertEqual(payload["objects"][0]["coordinates"][1][0], [32, 32])
        raw_payload = cv_path.read_text(encoding="utf-8")
        self.assertIn("[[10, 10], [70, 10], [70, 70], [10, 70], [10, 10]]", raw_payload)
        self.assertIn('  "format"', raw_payload)
        self.assertNotIn('    "format"', raw_payload)
        self.assertEqual(len(loaded), 2)
        self.assertTrue(any(not polygon.is_hole for polygon in loaded))
        loaded_holes = [polygon for polygon in loaded if polygon.is_hole]
        self.assertEqual(len(loaded_holes), 1)
        self.assertEqual(loaded_holes[0].parent_id, 3)

    def test_cv_loader_generates_hole_ids_without_colliding_with_object_ids(self) -> None:
        cv_path = self._artifact_path("hole_id_collision.cv")
        cv_path.write_text(
            """
{
  "format": "contour-vector",
  "version": 2,
  "image": {"path": "sample.png", "size": [100, 100]},
  "objects": [
    {
      "type": "Polygon",
      "id": 1,
      "coordinates": [
        [[10, 10], [60, 10], [60, 60], [10, 60], [10, 10]],
        [[25, 25], [40, 25], [40, 40], [25, 40], [25, 25]]
      ]
    },
    {
      "type": "Polygon",
      "id": 2,
      "coordinates": [
        [[70, 10], [90, 10], [90, 30], [70, 30], [70, 10]]
      ]
    }
  ]
}
""".strip(),
            encoding="utf-8",
        )

        _image_name, _image_size, loaded = load_polygons_cv(cv_path)

        ids = [polygon.id for polygon in loaded]
        self.assertEqual(len(ids), len(set(ids)))
        hole = next(polygon for polygon in loaded if polygon.is_hole)
        self.assertEqual(hole.parent_id, 1)
        self.assertNotIn(hole.id, {1, 2})

    def test_cv_writes_up_to_eight_vertices_per_line(self) -> None:
        points = [(float(index), float(index + 10)) for index in range(9)]
        area, perimeter, bbox = compute_polygon_metrics(points)
        polygon = PolygonData(id=9, points=points, area=area, perimeter=perimeter, bbox=bbox)
        cv_path = self._artifact_path("compact_rows.cv")

        save_polygons_cv(cv_path, "sample.png", [polygon], image_size=(100, 100))
        payload = cv_path.read_text(encoding="utf-8")

        self.assertIn(
            "[0, 10], [1, 11], [2, 12], [3, 13], [4, 14], [5, 15], [6, 16], [7, 17]",
            payload,
        )
        self.assertIn("[8, 18], [0, 10]", payload)
