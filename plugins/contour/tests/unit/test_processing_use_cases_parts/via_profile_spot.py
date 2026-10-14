"""test_processing_use_cases: test via profile spot."""
from __future__ import annotations
import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings
from contour.application.use_cases.processing import process_image_path


class ViaProfileSpotMixin:
    def test_via_profile_spot_detector_rejects_long_trace_background(self) -> None:
        source_image = np.full((100, 160), 50, dtype=np.uint8)
        for y_coord in (25, 50, 75):
            cv2.rectangle(source_image, (0, y_coord - 4), (159, y_coord + 4), 120, thickness=-1)
        for x_coord in (35, 80, 125):
            cv2.line(source_image, (x_coord, 0), (x_coord, 99), 100, thickness=2)
            for y_coord in (25, 50, 75):
                cv2.circle(source_image, (x_coord, y_coord), 3, 230, thickness=-1)

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_size_mode="fixed",
                fixed_via_widths=[7],
                fixed_via_heights=[7],
                min_area=3.0,
                via_white_range_enabled=True,
                via_white_range_min=0,
                via_white_range_max=255,
                via_black_range_enabled=False,
                via_min_contrast=40.0,
                via_min_score=0.50,
                via_min_edge_coverage=0.55,
                via_spot_line_suppression=0.85,
                via_min_roundness=50.0,
                heuristic_min_peak_prominence=1.0,
                heuristic_min_compactness=0.01,
                heuristic_min_edge_sharpness=0.0,
                heuristic_size_tolerance_fixed=0.36,
                heuristic_max_line_coherence=1.0,
            ),
            source_image=source_image,
        )

        self.assertEqual(len(result.polygons), 9)

    def test_via_profile_returns_debug_candidates_when_enabled(self) -> None:
        source_image = np.zeros((60, 60), dtype=np.uint8)
        cv2.circle(source_image, (20, 20), 4, 230, thickness=-1)
        cv2.rectangle(source_image, (38, 18), (52, 22), 230, thickness=-1)

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                algorithm_backend="sem",
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                min_area=4.0,
                via_white_range_enabled=True,
                via_white_range_min=200,
                via_white_range_max=255,
                via_black_range_enabled=False,
                via_min_roundness=60.0,
                debug_enabled=True,
            ),
            source_image=source_image,
        )

        self.assertTrue(result.debug_candidates)
        self.assertTrue(any(candidate.accepted for candidate in result.debug_candidates))
        self.assertTrue(all(polygon.recognition_score is not None for polygon in result.polygons))
        accepted = next(candidate for candidate in result.debug_candidates if candidate.accepted)
        self.assertEqual(accepted.source, "heuristic")
        self.assertIn("center_brightness", accepted.metrics)
        self.assertIn("compactness", accepted.metrics)
        self.assertIn("contribution_contrast", accepted.metrics)
