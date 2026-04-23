from __future__ import annotations

import unittest

import cv2
import numpy as np

from polygon_widget.application.processing import ContourExtractionSettings
from polygon_widget.application.use_cases.processing import build_detection_debug_maps, process_image_path


class ProcessingUseCasesTests(unittest.TestCase):
    def test_contour_settings_disable_corner_preservation_by_default(self) -> None:
        settings = ContourExtractionSettings()
        loaded = ContourExtractionSettings.from_dict({})

        self.assertFalse(settings.preserve_corners)
        self.assertFalse(loaded.preserve_corners)

    def test_via_channel_defaults_to_grayscale_when_hidden_from_ui(self) -> None:
        settings = ContourExtractionSettings()
        loaded = ContourExtractionSettings.from_dict({})

        self.assertEqual(settings.via_channel_mode, "grayscale")
        self.assertEqual(loaded.via_channel_mode, "grayscale")

    def test_contour_settings_round_trip_unified_via_controls(self) -> None:
        settings = ContourExtractionSettings(
            via_search_mode="template",
            via_min_score=0.42,
            via_min_contrast=23.5,
            via_min_edge_coverage=0.37,
            via_spot_line_suppression=0.73,
            via_template_min_score=0.48,
            via_min_roundness=55.0,
            via_template_images=[np.array([[1, 2], [3, 4]], dtype=np.uint8)],
        )

        loaded = ContourExtractionSettings.from_dict(settings.to_dict())

        self.assertEqual(loaded.via_search_mode, "template")
        self.assertEqual(loaded.via_min_score, 0.42)
        self.assertEqual(loaded.via_min_contrast, 23.5)
        self.assertEqual(loaded.via_min_edge_coverage, 0.37)
        self.assertEqual(loaded.via_spot_line_suppression, 0.73)
        self.assertEqual(loaded.via_template_min_score, 0.48)
        self.assertEqual(loaded.via_min_roundness, 55.0)
        self.assertEqual(loaded.via_template_images, [[[1, 2], [3, 4]]])

    def test_contour_settings_from_dict_ignores_legacy_via_detector_keys(self) -> None:
        legacy_payload = {
            "via_detector_gradient_enabled": True,
            "via_detector_spot_enabled": False,
            "via_detector_hough_enabled": True,
            "via_gradient_min_strength": 10.0,
            "via_gradient_min_coverage": 0.25,
            "via_spot_min_contrast": 14.0,
            "via_spot_min_roundness": 50.0,
            "via_hough_edge_threshold": 80.0,
            "via_hough_accumulator_threshold": 15.0,
            "via_component_min_score": 5.0,
            "via_contour_min_score": 7.0,
            "via_morphology_peak_scale": 0.2,
            "via_blob_min_circularity": 0.6,
            "via_auto_threshold_enabled": True,
            "via_min_score": 0.55,
        }

        loaded = ContourExtractionSettings.from_dict(legacy_payload)

        self.assertEqual(loaded.via_min_score, 0.55)
        self.assertFalse(hasattr(loaded, "via_detector_gradient_enabled"))

    def test_via_search_mode_defaults_to_hybrid_for_legacy_payloads(self) -> None:
        loaded = ContourExtractionSettings.from_dict({"via_min_score": 0.5})
        self.assertEqual(loaded.via_search_mode, "hybrid")

    def test_process_image_path_reuses_provided_preprocessed_image(self) -> None:
        loader_calls: list[str] = []
        source_image = np.zeros((32, 32), dtype=np.uint8)
        preprocessed_image = np.zeros((32, 32), dtype=np.uint8)
        preprocessed_image[8:24, 8:24] = 255

        def image_loader(path: str):
            loader_calls.append(path)
            raise AssertionError("image_loader should not be called when source_image is provided")

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(min_area=1.0),
            source_image=source_image,
            preprocessed_image=preprocessed_image,
            image_loader=image_loader,
        )

        self.assertEqual(loader_calls, [])
        self.assertIs(result.source_image, source_image)
        self.assertIs(result.preprocessed_image, preprocessed_image)
        self.assertEqual(result.pipeline_config, {"steps": []})
        self.assertEqual(len(result.polygons), 1)

    def test_process_image_path_can_drop_images_from_result(self) -> None:
        source_image = np.zeros((32, 32), dtype=np.uint8)
        source_image[8:24, 8:24] = 255

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(min_area=1.0),
            source_image=source_image,
            include_images_in_result=False,
        )

        self.assertIsNone(result.source_image)
        self.assertIsNone(result.preprocessed_image)
        self.assertIsNone(result.mask_image)
        self.assertEqual(len(result.polygons), 1)

    def test_conductor_gradient_refines_mask_to_source_edges(self) -> None:
        source_image = np.zeros((80, 100), dtype=np.uint8)
        cv2.rectangle(source_image, (25, 20), (74, 59), 220, thickness=-1)
        loose_mask = np.zeros_like(source_image)
        cv2.rectangle(loose_mask, (20, 15), (79, 64), 255, thickness=-1)

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                object_type="conductor",
                output_mode="polygon",
                min_area=10.0,
                epsilon=1.0,
                min_polygon_angle=0.0,
                conductor_gradient_enabled=True,
                conductor_gradient_min_strength=10.0,
                conductor_gradient_band_radius=8,
            ),
            source_image=source_image,
            preprocessed_image=loose_mask,
        )

        self.assertEqual(len(result.polygons), 1)
        self.assertEqual(result.polygons[0].bbox, (25, 20, 50, 40))

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
            ),
            source_image=source_image,
        )

        self.assertEqual(len(result.polygons), 12)

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

    def test_via_profile_gradient_rejects_linear_edges(self) -> None:
        source_image = np.full((96, 160), 80, dtype=np.uint8)
        cv2.line(source_image, (8, 40), (150, 40), 205, thickness=3)
        cv2.line(source_image, (8, 70), (150, 70), 205, thickness=3)

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
                via_min_edge_coverage=0.60,
                via_min_score=0.55,
                via_min_roundness=70.0,
                via_spot_line_suppression=0.95,
                debug_enabled=True,
            ),
            source_image=source_image,
        )

        self.assertEqual(len(result.polygons), 0)

    def test_via_profile_detects_contacts_with_saved_templates_only(self) -> None:
        source_image = np.full((80, 120), 60, dtype=np.uint8)
        for x_coord in (30, 70):
            cv2.circle(source_image, (x_coord, 40), 7, 210, thickness=-1)
        template = source_image[33:48, 23:38].copy()

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
                via_white_range_min=0,
                via_white_range_max=255,
                via_black_range_enabled=False,
                via_template_images=[template],
                via_template_min_score=0.5,
                via_min_score=0.4,
                via_min_contrast=0.0,
                via_min_edge_coverage=0.0,
                via_min_roundness=0.0,
                debug_enabled=True,
            ),
            source_image=source_image,
        )

        self.assertEqual(len(result.polygons), 2)
        self.assertTrue(
            any(candidate.source == "template" for candidate in result.debug_candidates if candidate.accepted)
        )

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

    def test_via_detector_template_boosts_score(self) -> None:
        rng = np.random.default_rng(7)
        source_image = np.full((80, 80), 90, dtype=np.uint8)
        source_image = np.clip(source_image + rng.normal(0, 4, source_image.shape), 0, 255).astype(np.uint8)
        cv2.circle(source_image, (40, 40), 5, 150, thickness=-1)
        template = source_image[34:47, 34:47].copy()

        base_settings = dict(
            extraction_profile="vias",
            object_type="via",
            output_mode="box",
            via_size_mode="fixed",
            fixed_via_widths=[11],
            fixed_via_heights=[11],
            min_area=3.0,
            via_white_range_enabled=True,
            via_white_range_min=0,
            via_white_range_max=255,
            via_black_range_enabled=False,
            via_min_score=0.92,
            via_min_contrast=4.0,
            via_min_edge_coverage=0.0,
            via_min_roundness=0.0,
            via_template_min_score=0.35,
        )
        without_template = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(**base_settings),
            source_image=source_image,
        )
        with_template = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(**base_settings, via_template_images=[template]),
            source_image=source_image,
        )

        self.assertEqual(len(without_template.polygons), 0)
        self.assertGreaterEqual(len(with_template.polygons), 1)

    def test_conductor_flow_can_merge_template_vias_in_sem_mode(self) -> None:
        source_image = np.zeros((96, 96), dtype=np.uint8)
        cv2.rectangle(source_image, (8, 40), (88, 56), 160, thickness=-1)
        cv2.circle(source_image, (48, 48), 6, 230, thickness=-1)
        template = source_image[40:57, 40:57].copy()
        settings = ContourExtractionSettings(
            extraction_profile="conductors",
            object_type="conductor",
            output_mode="polygon",
            min_area=4.0,
            via_search_mode="template",
            via_template_images=[template],
            via_template_min_score=0.25,
            via_min_score=0.35,
            via_min_contrast=10.0,
            via_min_edge_coverage=0.15,
            via_min_roundness=10.0,
            via_white_range_enabled=True,
            via_white_range_min=120,
            via_white_range_max=255,
            via_black_range_enabled=False,
        )
        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=settings,
            source_image=source_image,
        )
        categories = [polygon.category for polygon in result.polygons]
        self.assertIn("conductor", categories)
        self.assertIn("via", categories)


if __name__ == "__main__":
    unittest.main()
