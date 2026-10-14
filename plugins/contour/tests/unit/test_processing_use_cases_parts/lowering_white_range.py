"""test_processing_use_cases: test lowering white range."""
from __future__ import annotations
import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings
from contour.application.use_cases.processing import process_image_path


class LoweringWhiteRangeMixin:
    def test_lowering_white_range_minimum_does_not_merge_and_reduce_vias(self) -> None:
        source_image = np.full((120, 160), 80, dtype=np.uint8)
        source_image[:, 55:120] = 170
        for x_coord in (35, 75, 105, 135):
            for y_coord in (25, 55, 85):
                cv2.circle(source_image, (x_coord, y_coord), 4, 225, thickness=-1)

        base_settings = dict(
            extraction_profile="vias",
            object_type="via",
            output_mode="box",
            via_size_mode="fixed",
            fixed_via_widths=[9],
            fixed_via_heights=[9],
            min_area=8.0,
            via_white_range_enabled=True,
            via_white_range_max=255,
            via_black_range_enabled=False,
            via_min_roundness=60.0,
            heuristic_min_center_contrast=1.0,
            heuristic_min_peak_prominence=1.0,
            heuristic_min_compactness=0.01,
        )
        strict_result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(**base_settings, via_white_range_min=200),
            source_image=source_image,
        )
        wider_result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(**base_settings, via_white_range_min=120),
            source_image=source_image,
        )

        self.assertEqual(len(strict_result.polygons), 12)
        self.assertGreaterEqual(len(wider_result.polygons), len(strict_result.polygons))
