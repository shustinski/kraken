"""test_processing_use_cases: test via profile detects."""
from __future__ import annotations
import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings
from contour.application.use_cases.processing import process_image_path


class ViaProfileDetectsMixin:
    def test_via_profile_uses_white_range_parameters(self) -> None:
        source_image = np.zeros((40, 40), dtype=np.uint8)
        source_image[15:24, 16:25] = 220

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                min_area=1.0,
                via_channel_mode="grayscale",
                via_white_range_enabled=True,
                via_white_range_min=200,
                via_white_range_max=255,
                via_black_range_enabled=False,
                via_min_roundness=0.0,
            ),
            source_image=source_image,
        )

        self.assertEqual(len(result.polygons), 1)
        self.assertEqual(result.polygons[0].category, "via")

    def test_via_profile_uses_black_range_parameters(self) -> None:
        source_image = np.full((40, 40), 255, dtype=np.uint8)
        source_image[15:24, 16:25] = 20

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                min_area=1.0,
                via_channel_mode="grayscale",
                via_white_range_enabled=False,
                via_black_range_enabled=True,
                via_black_range_min=0,
                via_black_range_max=30,
                via_min_roundness=0.0,
            ),
            source_image=source_image,
        )

        self.assertEqual(len(result.polygons), 1)
        self.assertEqual(result.polygons[0].shape_hint, "box")

    def test_via_profile_uses_white_range_mid_tones(self) -> None:
        source_image = np.zeros((48, 48), dtype=np.uint8)
        source_image[10:18, 10:18] = 80
        source_image[28:36, 28:36] = 180

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                min_area=1.0,
                via_channel_mode="grayscale",
                via_white_range_enabled=True,
                via_white_range_min=70,
                via_white_range_max=100,
                via_black_range_enabled=False,
                via_min_roundness=0.0,
            ),
            source_image=source_image,
        )

        self.assertEqual(len(result.polygons), 1)
        self.assertLess(result.polygons[0].bbox[0], 20)

    def test_via_profile_detects_local_bright_spots_on_uneven_background(self) -> None:
        source_image = np.full((120, 160), 80, dtype=np.uint8)
        source_image[:, 55:120] = 170
        for x_coord in (35, 75, 105, 135):
            for y_coord in (25, 55, 85):
                cv2.circle(source_image, (x_coord, y_coord), 4, 225, thickness=-1)

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
                min_area=8.0,
                via_channel_mode="grayscale",
                via_white_range_enabled=True,
                via_white_range_min=200,
                via_white_range_max=255,
                via_black_range_enabled=False,
                via_min_roundness=70.0,
                heuristic_min_center_contrast=1.0,
                heuristic_min_peak_prominence=1.0,
                heuristic_min_compactness=0.01,
            ),
            source_image=source_image,
        )

        self.assertEqual(len(result.polygons), 12)
