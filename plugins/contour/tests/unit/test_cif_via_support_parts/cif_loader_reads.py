"""test_cif_via_support: test cif loader reads."""
from __future__ import annotations
import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings
from contour.contour_extractor import extract_polygons
from contour.serializers import (
    load_polygons_cif,
    save_polygons_cif,
)

from test_cif_via_support_parts.lib_open_cif import _rectangle_polygon


class CifLoaderReadsMixin:
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

    def test_cif_via_round_trip_preserves_odd_size_and_position_exactly(self) -> None:
        polygon = _rectangle_polygon(11, 13, 22, 22)
        polygon.category = "via"
        polygon.shape_hint = "box"
        cif_path = self._artifact_path("odd_via_box.cif")

        save_polygons_cif(cif_path, "sample.png", [polygon], image_size=(100, 80))
        _image_name, _image_size, loaded = load_polygons_cif(cif_path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].points, polygon.points)
        self.assertEqual(loaded[0].bbox, polygon.bbox)
