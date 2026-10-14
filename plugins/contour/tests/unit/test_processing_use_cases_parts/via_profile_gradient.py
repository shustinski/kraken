"""test_processing_use_cases: test via profile gradient."""
from __future__ import annotations
import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings
from contour.application.use_cases.processing import process_image_path


class ViaProfileGradientMixin:
    def test_via_profile_detects_binary_grid_contacts(self) -> None:
        source_image = np.zeros((120, 120), dtype=np.uint8)
        for x_coord in range(15, 106, 20):
            for y_coord in range(15, 106, 20):
                cv2.circle(source_image, (x_coord, y_coord), 4, 255, thickness=-1)

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_size_mode="fixed",
                fixed_via_widths=[9],
                fixed_via_heights=[9],
                min_area=5.0,
                via_white_range_enabled=True,
                via_white_range_min=200,
                via_white_range_max=255,
                via_black_range_enabled=False,
                via_min_roundness=60.0,
            ),
            source_image=source_image,
        )

        self.assertEqual(len(result.polygons), 25)

    def test_via_profile_detects_ring_like_contacts_with_hough_support(self) -> None:
        source_image = np.zeros((80, 80), dtype=np.uint8)
        cv2.circle(source_image, (40, 40), 10, 255, thickness=2)

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_size_mode="fixed",
                fixed_via_widths=[21],
                fixed_via_heights=[21],
                min_area=5.0,
                via_white_range_enabled=True,
                via_white_range_min=180,
                via_white_range_max=255,
                via_black_range_enabled=False,
                via_min_roundness=40.0,
            ),
            source_image=source_image,
        )

        self.assertEqual(len(result.polygons), 1)

    def test_via_profile_detects_round_gradient_edge_without_binary_components(self) -> None:
        source_image = np.full((96, 96), 80, dtype=np.uint8)
        cv2.circle(source_image, (48, 48), 10, 205, thickness=-1)

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_size_mode="fixed",
                fixed_via_widths=[21],
                fixed_via_heights=[21],
                min_area=5.0,
                via_white_range_enabled=True,
                via_white_range_min=0,
                via_white_range_max=255,
                via_black_range_enabled=False,
                via_min_contrast=8.0,
                via_min_score=0.25,
                via_min_roundness=40.0,
                debug_enabled=True,
                heuristic_min_peak_prominence=1.0,
                heuristic_min_compactness=0.01,
                heuristic_min_edge_sharpness=0.0,
            ),
            source_image=source_image,
        )

        self.assertEqual(len(result.polygons), 1)
        self.assertTrue(any(candidate.accepted for candidate in result.debug_candidates))

    def test_via_profile_gradient_detects_bright_spot_with_ui_coverage(self) -> None:
        source_image = np.full((64, 64), 90, dtype=np.uint8)
        cv2.circle(source_image, (32, 32), 4, 210, thickness=-1)

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_size_mode="fixed",
                fixed_via_widths=[9],
                fixed_via_heights=[9],
                min_area=3.0,
                via_white_range_enabled=True,
                via_white_range_min=0,
                via_white_range_max=255,
                via_black_range_enabled=False,
                via_min_contrast=19.0,
                via_min_edge_coverage=0.20,
                via_min_score=0.25,
                via_min_roundness=40.0,
                debug_enabled=True,
                heuristic_min_peak_prominence=1.0,
                heuristic_min_compactness=0.01,
            ),
            source_image=source_image,
        )

        self.assertEqual(len(result.polygons), 1)
        self.assertTrue(any(candidate.accepted for candidate in result.debug_candidates))

    def test_via_profile_gradient_detects_bright_spots_on_trace(self) -> None:
        rng = np.random.default_rng(2)
        source_image = np.full((120, 220), 75, dtype=np.uint8)
        source_image = np.clip(source_image + rng.normal(0, 10, source_image.shape), 0, 255).astype(np.uint8)
        cv2.rectangle(source_image, (0, 70), (219, 88), 145, thickness=-1)
        expected_x = (25, 48, 72, 98, 130, 158, 190)
        for x_coord in expected_x:
            cv2.circle(source_image, (x_coord, 79), 5, 235, thickness=-1)
        source_image = cv2.GaussianBlur(source_image, (3, 3), 0)

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_size_mode="fixed",
                fixed_via_widths=[12],
                fixed_via_heights=[12],
                min_area=3.0,
                via_white_range_enabled=True,
                via_white_range_min=180,
                via_white_range_max=255,
                via_black_range_enabled=False,
                via_min_contrast=10.0,
                via_min_edge_coverage=0.0,
                via_min_score=0.30,
                via_min_roundness=40.0,
                via_spot_line_suppression=0.9,
                debug_enabled=True,
                heuristic_min_peak_prominence=1.0,
                heuristic_min_compactness=0.01,
                heuristic_min_edge_sharpness=0.0,
            ),
            source_image=source_image,
        )

        accepted_centers = [
            (candidate.bbox[0] + candidate.bbox[2] / 2.0, candidate.bbox[1] + candidate.bbox[3] / 2.0)
            for candidate in result.debug_candidates
            if candidate.accepted
        ]
        self.assertEqual(len(result.polygons), len(expected_x))
        for x_coord in expected_x:
            self.assertTrue(
                any(
                    abs(center_x - x_coord) <= 3.0 and abs(center_y - 79) <= 3.0
                    for center_x, center_y in accepted_centers
                )
            )
