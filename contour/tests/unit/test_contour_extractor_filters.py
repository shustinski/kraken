from __future__ import annotations

import unittest

import cv2
import numpy as np

from polygon_widget.application.processing import ContourExtractionSettings
from polygon_widget.contour_extractor import extract_polygons


class ContourExtractorFilterTests(unittest.TestCase):
    def test_excludes_border_touching_contours(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.rectangle(mask, (0, 10), (20, 30), 255, thickness=-1)

        baseline = extract_polygons(mask)
        filtered = extract_polygons(mask, ContourExtractionSettings(exclude_border_touching=True))

        self.assertEqual(len(baseline), 1)
        self.assertEqual(filtered, [])

    def test_filters_by_bbox_and_perimeter_limits(self) -> None:
        mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (60, 16), 255, thickness=-1)

        by_height = extract_polygons(mask, ContourExtractionSettings(min_bbox_height=10))
        by_perimeter = extract_polygons(mask, ContourExtractionSettings(max_perimeter=80.0))

        self.assertEqual(by_height, [])
        self.assertEqual(by_perimeter, [])

    def test_filters_by_solidity_and_extent(self) -> None:
        mask = np.zeros((96, 96), dtype=np.uint8)
        points = np.array(
            [
                [10, 10],
                [70, 10],
                [70, 25],
                [30, 25],
                [30, 70],
                [10, 70],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], 255)

        baseline = extract_polygons(mask)
        by_solidity = extract_polygons(mask, ContourExtractionSettings(min_solidity=0.8))
        by_extent = extract_polygons(mask, ContourExtractionSettings(min_extent=0.8))

        self.assertEqual(len(baseline), 1)
        self.assertEqual(by_solidity, [])
        self.assertEqual(by_extent, [])

    def test_filters_by_hierarchy_depth_and_hole_ratio(self) -> None:
        mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (80, 80), 255, thickness=-1)
        cv2.rectangle(mask, (30, 30), (60, 60), 0, thickness=-1)

        all_polygons = extract_polygons(mask, ContourExtractionSettings(retrieval_mode="RETR_TREE"))
        only_holes = extract_polygons(
            mask,
            ContourExtractionSettings(retrieval_mode="RETR_TREE", min_hierarchy_depth=1),
        )
        only_external = extract_polygons(
            mask,
            ContourExtractionSettings(retrieval_mode="RETR_TREE", max_hierarchy_depth=0),
        )
        without_large_hole = extract_polygons(
            mask,
            ContourExtractionSettings(retrieval_mode="RETR_TREE", max_hole_area_ratio=0.15),
        )

        self.assertEqual(len(all_polygons), 2)
        self.assertTrue(any(polygon.is_hole for polygon in all_polygons))
        self.assertEqual(len(only_holes), 1)
        self.assertTrue(only_holes[0].is_hole)
        self.assertEqual(len(only_external), 1)
        self.assertFalse(only_external[0].is_hole)
        self.assertEqual(len(without_large_hole), 1)
        self.assertFalse(without_large_hole[0].is_hole)

    def test_preserve_corners_keeps_notch_while_removing_jitter_vertices(self) -> None:
        mask = np.zeros((128, 128), dtype=np.uint8)
        points = np.array(
            [
                [12, 12],
                [108, 12],
                [108, 36],
                [72, 36],
                [72, 72],
                [108, 72],
                [108, 108],
                [12, 108],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], 255)

        simplified = extract_polygons(
            mask,
            ContourExtractionSettings(epsilon=8.0, preserve_corners=False),
        )
        preserved = extract_polygons(
            mask,
            ContourExtractionSettings(epsilon=8.0, preserve_corners=True),
        )

        self.assertEqual(len(simplified), 1)
        self.assertEqual(len(preserved), 1)
        self.assertGreater(len(simplified[0].points), len(preserved[0].points))
        self.assertTrue(any(abs(x_coord - 72.0) <= 1.5 and abs(y_coord - 37.0) <= 1.5 for x_coord, y_coord in preserved[0].points))
        self.assertTrue(any(abs(x_coord - 72.0) <= 1.5 and abs(y_coord - 71.0) <= 1.5 for x_coord, y_coord in preserved[0].points))


if __name__ == "__main__":
    unittest.main()
