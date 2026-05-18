from __future__ import annotations

import unittest

import cv2
import numpy as np

from contour.application.processing import ContourExtractionSettings
from contour.contour_extractor import _finalize_closed_polygon_points, extract_polygons
from contour.domain.polygon_ring import is_valid_closed_polygon_ring


def _angle(
    prev_point: tuple[float, float], current_point: tuple[float, float], next_point: tuple[float, float]
) -> float:
    first = np.asarray(prev_point, dtype=np.float32) - np.asarray(current_point, dtype=np.float32)
    second = np.asarray(next_point, dtype=np.float32) - np.asarray(current_point, dtype=np.float32)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-6 or second_norm <= 1e-6:
        return 180.0
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


class ContourExtractorFilterTests(unittest.TestCase):
    def test_finalize_rounds_recognized_vertices_to_integer_coordinates(self) -> None:
        raw = np.array([[[0, 0]], [[20, 0]], [[20, 20]], [[0, 20]]], dtype=np.int32)

        out = _finalize_closed_polygon_points(
            [(0.2, 0.6), (20.4, 0.1), (20.5, 20.5), (0.2, 20.4)],
            raw,
            (22, 22),
            ContourExtractionSettings(
                epsilon=0.0, min_polygon_angle=0.0, object_type="conductor", output_mode="polygon"
            ),
        )

        self.assertEqual(out, [(0, 1), (20, 0), (20, 20), (0, 20)])

    def test_invalid_bow_tie_repaired_using_filled_lattice(self) -> None:
        # Order (0,0)->(W,H)->(0,H)->(W,0) self-intersects; same bounding square as a valid axis ring.
        raw = np.array([[[0, 0]], [[20, 0]], [[20, 20]], [[0, 20]]], dtype=np.int32)
        bad_ring = [(0.0, 0.0), (20.0, 20.0), (0.0, 20.0), (20.0, 0.0)]
        self.assertFalse(is_valid_closed_polygon_ring(bad_ring))
        out = _finalize_closed_polygon_points(
            list(bad_ring),
            raw,
            (22, 22),
            ContourExtractionSettings(
                epsilon=0.0, min_polygon_angle=0.0, object_type="conductor", output_mode="polygon"
            ),
        )
        self.assertIsNotNone(out)
        self.assertTrue(is_valid_closed_polygon_ring(out))

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

    def test_rejects_thin_strip_by_min_polygon_width(self) -> None:
        mask = np.zeros((120, 120), dtype=np.uint8)
        cv2.rectangle(mask, (10, 50), (100, 54), 255, thickness=-1)

        baseline = extract_polygons(
            mask, ContourExtractionSettings(min_area=1.0, min_perimeter=1.0, min_polygon_width_px=0.0)
        )
        filtered = extract_polygons(
            mask, ContourExtractionSettings(min_area=1.0, min_perimeter=1.0, min_polygon_width_px=8.0)
        )

        self.assertEqual(len(baseline), 1)
        self.assertEqual(filtered, [])

    def test_keeps_8px_strip_when_min_polygon_width_2_5(self) -> None:
        # Regression: 15% tile over the whole fill once measured ~2 px for an 8 px-wide bar (edge dilution).
        mask = np.zeros((120, 120), dtype=np.uint8)
        cv2.rectangle(mask, (10, 20), (100, 28), 255, thickness=-1)

        polygons = extract_polygons(
            mask, ContourExtractionSettings(min_area=1.0, min_perimeter=1.0, min_polygon_width_px=2.5)
        )
        self.assertEqual(len(polygons), 1)

    def test_via_profile_filters_by_roundness(self) -> None:
        mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.circle(mask, (24, 48), 8, 255, thickness=-1)
        cv2.rectangle(mask, (50, 44), (85, 48), 255, thickness=-1)

        vias = extract_polygons(
            mask,
            ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                min_area=1.0,
                via_min_roundness=50.0,
            ),
        )

        self.assertEqual(len(vias), 1)
        self.assertLess(vias[0].bbox[0], 40)

    def test_via_roundness_rejects_moderately_elongated_boxes(self) -> None:
        mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.circle(mask, (20, 40), 8, 255, thickness=-1)
        cv2.rectangle(mask, (44, 35), (63, 44), 255, thickness=-1)

        vias = extract_polygons(
            mask,
            ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                min_area=1.0,
                via_min_roundness=60.0,
            ),
        )

        self.assertEqual(len(vias), 1)
        self.assertLess(vias[0].bbox[0], 35)

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

    def test_filters_small_inner_contours_by_min_inner_hole_area(self) -> None:
        mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.rectangle(mask, (8, 8), (88, 88), 255, thickness=-1)
        cv2.rectangle(mask, (20, 20), (22, 22), 0, thickness=-1)

        with_small_hole = extract_polygons(
            mask,
            ContourExtractionSettings(
                retrieval_mode="RETR_TREE",
                min_inner_hole_area=0.0,
            ),
        )
        filtered_small_hole = extract_polygons(
            mask,
            ContourExtractionSettings(
                retrieval_mode="RETR_TREE",
                min_inner_hole_area=100.0,
            ),
        )

        self.assertEqual(len(with_small_hole), 2)
        self.assertTrue(any(polygon.is_hole for polygon in with_small_hole))
        self.assertEqual(len(filtered_small_hole), 1)
        self.assertFalse(filtered_small_hole[0].is_hole)

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
            ContourExtractionSettings(epsilon=20.0, preserve_corners=False, approximation_mode="CHAIN_APPROX_NONE"),
        )
        preserved = extract_polygons(
            mask,
            ContourExtractionSettings(epsilon=20.0, preserve_corners=True, approximation_mode="CHAIN_APPROX_NONE"),
        )

        self.assertEqual(len(simplified), 1)
        self.assertEqual(len(preserved), 1)
        self.assertLess(len(simplified[0].points), len(preserved[0].points))
        self.assertTrue(
            any(abs(x_coord - 72.0) <= 1.5 and abs(y_coord - 37.0) <= 1.5 for x_coord, y_coord in preserved[0].points)
        )
        self.assertTrue(
            any(abs(x_coord - 72.0) <= 1.5 and abs(y_coord - 71.0) <= 1.5 for x_coord, y_coord in preserved[0].points)
        )

    def test_epsilon_simplifies_binary_contours_without_corner_preservation(self) -> None:
        mask = np.zeros((128, 128), dtype=np.uint8)
        points = np.array(
            [
                [16, 16],
                [112, 16],
                [112, 32],
                [96, 32],
                [96, 48],
                [112, 48],
                [112, 64],
                [96, 64],
                [96, 80],
                [112, 80],
                [112, 112],
                [16, 112],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], 255)

        baseline = extract_polygons(
            mask,
            ContourExtractionSettings(epsilon=0.0, approximation_mode="CHAIN_APPROX_NONE"),
        )
        simplified = extract_polygons(
            mask,
            ContourExtractionSettings(epsilon=6.0, approximation_mode="CHAIN_APPROX_NONE"),
        )

        self.assertEqual(len(baseline), 1)
        self.assertEqual(len(simplified), 1)
        self.assertLess(len(simplified[0].points), len(baseline[0].points))

    def test_min_polygon_angle_removes_acute_vertices(self) -> None:
        mask = np.zeros((80, 100), dtype=np.uint8)
        points = np.array([[10, 10], [80, 10], [80, 40], [50, 40], [45, 20], [40, 40], [10, 40]], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

        polygons = extract_polygons(
            mask,
            ContourExtractionSettings(
                object_type="conductor",
                output_mode="polygon",
                epsilon=1.0,
                min_area=1.0,
                min_polygon_angle=90.0,
            ),
        )

        self.assertEqual(len(polygons), 1)
        polygon_points = polygons[0].points
        self.assertTrue(
            all(
                _angle(
                    polygon_points[index - 1], polygon_points[index], polygon_points[(index + 1) % len(polygon_points)]
                )
                >= 90.0
                for index in range(len(polygon_points))
            )
        )


if __name__ == "__main__":
    unittest.main()
