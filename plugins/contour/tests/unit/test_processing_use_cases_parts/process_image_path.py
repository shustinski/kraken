"""test_processing_use_cases: test process image path."""
from __future__ import annotations
import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings
from contour.application.use_cases.processing import process_image_path


class ProcessImagePathMixin:
    def test_contour_settings_round_trip_edge_method_and_debug_gradient_flag(self) -> None:
        settings = ContourExtractionSettings(
            edge_method="phase_congruency",
            via_gradient_edge_method="scharr",
            conductor_gradient_edge_method="combined",
            debug_gradient_map_enabled=True,
        )

        loaded = ContourExtractionSettings.from_dict(settings.to_dict())

        self.assertEqual(loaded.edge_method, "phase_congruency")
        self.assertEqual(loaded.via_gradient_edge_method, "scharr")
        self.assertEqual(loaded.conductor_gradient_edge_method, "combined")
        self.assertTrue(loaded.debug_gradient_map_enabled)

    def test_process_image_path_populates_debug_gradient_maps_when_enabled(self) -> None:
        source_image = np.zeros((32, 40), dtype=np.uint8)
        cv2.circle(source_image, (20, 16), 4, 230, thickness=-1)

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
                via_white_range_min=200,
                via_white_range_max=255,
                debug_enabled=True,
                debug_gradient_map_enabled=True,
            ),
            source_image=source_image,
        )

        self.assertTrue(result.debug_gradient_maps)
        self.assertIn("gradient_elevation", result.debug_gradient_maps)
        self.assertIn("mask", result.debug_gradient_maps)
