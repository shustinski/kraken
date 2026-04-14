from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from polygon_widget.application.processing import ContourExtractionSettings
from polygon_widget.contour_extractor import extract_polygons
from polygon_widget.serializers import load_polygons_cif, save_polygons_cif


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


if __name__ == "__main__":
    unittest.main()
