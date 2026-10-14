"""test_processing_use_cases: test via detector auto."""
from __future__ import annotations
import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings
from contour.application.use_cases.processing import process_image_path


class ViaDetectorAutoMixin:
    def test_via_detector_merges_with_iou_not_distance(self) -> None:
        source_image = np.full((80, 160), 60, dtype=np.uint8)
        centers = [(40, 40), (95, 40)]
        for x_coord, y_coord in centers:
            cv2.circle(source_image, (x_coord, y_coord), 7, 220, thickness=-1)

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_size_mode="fixed",
                fixed_via_widths=[15],
                fixed_via_heights=[15],
                min_area=3.0,
                via_white_range_enabled=True,
                via_white_range_min=180,
                via_white_range_max=255,
                via_black_range_enabled=False,
                via_min_score=0.25,
                via_min_contrast=10.0,
                via_min_edge_coverage=0.0,
                via_min_roundness=30.0,
                heuristic_min_edge_sharpness=0.0,
            ),
            source_image=source_image,
        )

        self.assertEqual(len(result.polygons), 2)

    def test_via_detector_auto_polarity_detects_both_bright_and_dark(self) -> None:
        source_image = np.full((80, 160), 128, dtype=np.uint8)
        cv2.circle(source_image, (40, 40), 6, 240, thickness=-1)
        cv2.circle(source_image, (120, 40), 6, 20, thickness=-1)

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_size_mode="fixed",
                fixed_via_widths=[13],
                fixed_via_heights=[13],
                min_area=3.0,
                via_white_range_enabled=True,
                via_white_range_min=180,
                via_white_range_max=255,
                via_black_range_enabled=True,
                via_black_range_min=0,
                via_black_range_max=60,
                via_min_score=0.25,
                via_min_contrast=10.0,
                via_min_edge_coverage=0.0,
                via_min_roundness=30.0,
                heuristic_min_edge_sharpness=0.0,
            ),
            source_image=source_image,
        )

        xs = sorted(polygon.bbox[0] + polygon.bbox[2] / 2 for polygon in result.polygons)
        self.assertEqual(len(result.polygons), 2)
        self.assertLess(xs[0], 80)
        self.assertGreater(xs[1], 80)

    def test_via_detector_rejects_low_edge_coverage(self) -> None:
        source_image = np.full((80, 80), 60, dtype=np.uint8)
        cv2.rectangle(source_image, (20, 30), (60, 50), 220, thickness=-1)

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_size_mode="fixed",
                fixed_via_widths=[15],
                fixed_via_heights=[15],
                min_area=3.0,
                via_white_range_enabled=True,
                via_white_range_min=180,
                via_white_range_max=255,
                via_black_range_enabled=False,
                via_min_score=0.30,
                via_min_contrast=10.0,
                via_min_edge_coverage=0.80,
                via_min_roundness=60.0,
                debug_enabled=True,
            ),
            source_image=source_image,
        )

        self.assertEqual(len(result.polygons), 0)
