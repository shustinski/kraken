"""test_processing_use_cases: test build detection debug."""
from __future__ import annotations
import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings
from contour.application.use_cases.processing import build_detection_debug_maps


class BuildDetectionDebugMixin:
    def test_build_detection_debug_maps_populates_expected_layers_for_vias(self) -> None:
        source_image = np.zeros((40, 40), dtype=np.uint8)
        cv2.circle(source_image, (20, 20), 5, 230, thickness=-1)
        settings = ContourExtractionSettings(
            extraction_profile="vias",
            object_type="via",
            output_mode="box",
            via_size_mode="fixed",
            fixed_via_widths=[9],
            fixed_via_heights=[9],
            min_area=2.0,
            via_white_range_enabled=True,
            via_white_range_min=200,
            via_white_range_max=255,
            debug_enabled=True,
            debug_gradient_map_enabled=True,
            edge_method="scharr",
        )

        maps = build_detection_debug_maps(source_image, source_image, settings)

        expected = {
            "source_gray",
            "gradient_elevation",
            "gradient_color",
            "scharr",
            "phase_congruency",
            "structured",
            "ridge",
            "mask",
        }
        self.assertTrue(expected.issubset(maps.keys()))
        self.assertEqual(maps["source_gray"].shape, source_image.shape)
        self.assertEqual(maps["gradient_elevation"].dtype, np.uint8)
        self.assertEqual(maps["gradient_color"].shape[:2], source_image.shape)
        self.assertEqual(maps["gradient_color"].ndim, 3)
        self.assertEqual(maps["mask"].dtype, np.uint8)
        self.assertIn("spot_response", maps)
        self.assertEqual(maps["spot_response"].shape, source_image.shape)
        self.assertGreater(int(maps["gradient_elevation"].max()), 0)

    def test_build_detection_debug_maps_for_conductors_uses_resolved_edge_method(self) -> None:
        source_image = np.zeros((60, 80), dtype=np.uint8)
        cv2.rectangle(source_image, (20, 15), (60, 45), 220, thickness=-1)
        preprocessed = np.zeros_like(source_image)
        cv2.rectangle(preprocessed, (18, 12), (63, 48), 255, thickness=-1)

        settings = ContourExtractionSettings(
            object_type="conductor",
            output_mode="polygon",
            min_area=10.0,
            epsilon=1.0,
            min_polygon_angle=0.0,
            conductor_gradient_enabled=True,
            conductor_gradient_min_strength=10.0,
            conductor_gradient_band_radius=6,
            conductor_gradient_edge_method="scharr",
            debug_gradient_map_enabled=True,
        )

        maps = build_detection_debug_maps(source_image, preprocessed, settings)

        self.assertIn("gradient_elevation", maps)
        self.assertIn("conductor_gradient_elevation", maps)
        self.assertIn("mask", maps)
        self.assertNotIn("spot_response", maps)
        self.assertEqual(maps["gradient_elevation"].shape, source_image.shape)
        self.assertEqual(maps["mask"].shape, source_image.shape)

    def test_build_detection_debug_maps_handles_missing_source_gracefully(self) -> None:
        settings = ContourExtractionSettings()
        self.assertEqual(build_detection_debug_maps(None, None, settings), {})
