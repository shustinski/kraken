from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from contour.application.processing import ContourExtractionSettings, DisplaySettings, SaveOptions
from contour.contour_extractor import extract_polygons
from contour.domain import PolygonData, compute_polygon_metrics
from contour.serializers import (
    export_dataset_frame,
    load_polygons_cif,
    load_polygons_cv,
    load_polygons_vector,
    save_polygons_cif,
    save_polygons_cv,
    save_result_bundle,
)


def _rectangle_polygon(left: int, top: int, right: int, bottom: int) -> PolygonData:
    points = [
        (float(left), float(top)),
        (float(right), float(top)),
        (float(right), float(bottom)),
        (float(left), float(bottom)),
    ]
    area, perimeter, bbox = compute_polygon_metrics(points)
    return PolygonData(id=1, points=points, area=area, perimeter=perimeter, bbox=bbox)


class CifViaSupportTests(unittest.TestCase):
    def _artifact_path(self, name: str) -> Path:
        root = Path(".tmp-tests")
        root.mkdir(exist_ok=True)
        path = root / name
        if path.exists():
            path.unlink()
        return path

    def test_via_profile_extracts_box_shapes(self) -> None:
        mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.ellipse(mask, (40, 40), (8, 6), 0, 0, 360, 255, thickness=-1)

        polygons = extract_polygons(
            mask,
            ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                min_area=10.0,
            ),
        )

        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0].category, "via")
        self.assertEqual(polygons[0].shape_hint, "box")
        self.assertFalse(polygons[0].is_hole)
        self.assertEqual(len(polygons[0].points), 4)

    def test_via_profile_applies_via_size_limits(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.ellipse(mask, (30, 50), (5, 5), 0, 0, 360, 255, thickness=-1)
        cv2.ellipse(mask, (70, 50), (12, 12), 0, 0, 360, 255, thickness=-1)

        polygons = extract_polygons(
            mask,
            ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                min_via_width=8,
                max_via_width=20,
                min_via_height=8,
                max_via_height=20,
                min_area=10.0,
            ),
        )

        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0].category, "via")

    def test_via_profile_applies_fixed_single_size(self) -> None:
        mask = np.zeros((120, 120), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (19, 17), 255, thickness=-1)
        cv2.rectangle(mask, (50, 10), (61, 21), 255, thickness=-1)

        polygons = extract_polygons(
            mask,
            ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_size_mode="fixed",
                fixed_via_widths=[10],
                fixed_via_heights=[8],
                min_area=10.0,
            ),
        )

        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0].bbox[2:], (11, 9))

    def test_via_profile_fixed_size_allows_small_mask_variation(self) -> None:
        mask = np.zeros((120, 120), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (27, 28), 255, thickness=-1)

        polygons = extract_polygons(
            mask,
            ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_size_mode="fixed",
                fixed_via_widths=[7],
                fixed_via_heights=[7],
                min_area=10.0,
            ),
        )

        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0].bbox[2:], (8, 8))

    def test_via_profile_applies_fixed_size_sets(self) -> None:
        mask = np.zeros((160, 160), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (19, 17), 255, thickness=-1)
        cv2.rectangle(mask, (40, 10), (51, 19), 255, thickness=-1)
        cv2.rectangle(mask, (40, 40), (49, 49), 255, thickness=-1)
        cv2.rectangle(mask, (80, 10), (94, 24), 255, thickness=-1)

        polygons = extract_polygons(
            mask,
            ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_size_mode="fixed",
                fixed_via_widths=[10, 12],
                fixed_via_heights=[8, 10],
                min_area=10.0,
            ),
        )

        self.assertEqual(len(polygons), 2)
        self.assertEqual(sorted(polygon.bbox[2:] for polygon in polygons), [(11, 9), (13, 11)])

    def test_via_profile_suppresses_intersecting_boxes(self) -> None:
        mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.circle(mask, (20, 20), 5, 255, thickness=-1)
        cv2.circle(mask, (29, 29), 5, 255, thickness=-1)

        vias = extract_polygons(
            mask,
            ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                min_area=10.0,
            ),
        )

        self.assertEqual(len(vias), 1)

    def test_contour_settings_parse_fixed_via_values_from_dict(self) -> None:
        settings = ContourExtractionSettings.from_dict(
            {
                "extraction_profile": "vias",
                "object_type": "via",
                "output_mode": "box",
                "via_size_mode": "fixed",
                "fixed_via_widths": "8, 10; 12",
                "fixed_via_heights": [8, "10", 12.0],
            }
        )

        self.assertEqual(settings.via_size_mode, "fixed")
        self.assertEqual(settings.fixed_via_widths, [8, 10, 12])
        self.assertEqual(settings.fixed_via_heights, [8, 10, 12])

    def test_cif_loader_reads_b_commands_as_vias(self) -> None:
        cif_path = self._artifact_path("sample_via_box.cif")
        cif_path.write_text(
            "\n".join(
                [
                    "DS 1 1 1;",
                    "L NM;",
                    "( R sample.png );",
                    "( S 2000 2000 );",
                    "B 10 8 1000 1500;",
                    "DF;",
                    "E",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        image_name, image_size, polygons = load_polygons_cif(cif_path)

        self.assertEqual(image_name, "sample.png")
        self.assertEqual(image_size, (2000, 2000))
        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0].category, "via")
        self.assertEqual(polygons[0].shape_hint, "box")

    def test_cif_saves_box_shapes_using_b_records(self) -> None:
        mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.ellipse(mask, (30, 35), (5, 4), 0, 0, 360, 255, thickness=-1)
        polygons = extract_polygons(
            mask,
            ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                min_area=10.0,
            ),
        )

        cif_path = self._artifact_path("saved_via_box.cif")
        save_polygons_cif(cif_path, "sample.png", polygons, image_size=(80, 80))
        payload = cif_path.read_text(encoding="utf-8")

        self.assertIn("B ", payload)
        self.assertNotIn("P ", payload)

    def test_dataset_export_writes_image_and_cif_subdirectories(self) -> None:
        with TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image_path = root / "frame_1.png"
            cv2.imwrite(str(image_path), image)
            polygon = _rectangle_polygon(4, 4, 20, 20)

            saved_files = export_dataset_frame(root / "dataset", str(image_path), [polygon], image)

            saved_image = Path(saved_files["image"])
            saved_cif = Path(saved_files["cif"])
            self.assertEqual(saved_image.parent.name, "images")
            self.assertEqual(saved_cif.parent.name, "cif")
            self.assertTrue(saved_image.exists())
            self.assertTrue(saved_cif.exists())
            payload = saved_cif.read_text(encoding="utf-8")
            self.assertIn("( R frame_1.png );", payload)

    def test_cif_round_trip_restores_hole_topology_from_cut_polygon(self) -> None:
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
        self.assertTrue(any(not polygon.is_hole for polygon in loaded))
        self.assertTrue(any(polygon.is_hole for polygon in loaded))
        hole_parent_ids = {polygon.parent_id for polygon in loaded if polygon.is_hole}
        self.assertTrue(any(parent_id is not None for parent_id in hole_parent_ids))

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

    def test_result_bundle_can_save_cv_without_legacy_text_formats(self) -> None:
        with TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            polygon = _rectangle_polygon(4, 4, 20, 20)

            saved = save_result_bundle(
                root,
                "sample.png",
                [polygon],
                image,
                DisplaySettings(),
                SaveOptions(save_cif=False, save_cv=True, save_preview=True),
            )

            self.assertEqual(set(saved), {"cv", "preview"})
            self.assertTrue(Path(saved["cv"]).exists())
            self.assertNotIn('"metadata"', Path(saved["cv"]).read_text(encoding="utf-8"))
            self.assertTrue(Path(saved["preview"]).exists())
            self.assertFalse((root / "sample.csv").exists())
            self.assertFalse((root / "sample.txt").exists())
            self.assertFalse((root / "sample.svg").exists())


if __name__ == "__main__":
    unittest.main()
