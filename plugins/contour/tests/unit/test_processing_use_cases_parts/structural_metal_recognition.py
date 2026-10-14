"""test_processing_use_cases: test structural metal recognition."""
from __future__ import annotations
from unittest.mock import patch
import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings
from contour.application.use_cases.processing import process_image_path


class StructuralMetalRecognitionMixin:
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
        self.assertTrue(settings.debug_enabled)
        self.assertTrue(loaded.debug_enabled)

    def test_contour_settings_round_trip_unified_via_controls(self) -> None:
        settings = ContourExtractionSettings(
            via_search_mode="template",
            via_min_score=0.42,
            via_min_contrast=23.5,
            via_min_edge_coverage=0.37,
            via_spot_line_suppression=0.73,
            via_template_min_score=0.48,
            via_min_roundness=55.0,
            via_output_diameter=17,
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
        self.assertEqual(loaded.via_output_diameter, 17)
        self.assertEqual(loaded.via_template_images, [[[1, 2], [3, 4]]])

    def test_contour_settings_round_trip_metal_preprocessing(self) -> None:
        settings = ContourExtractionSettings(
            metal_preprocess_subtract_background=False,
            metal_preprocess_background_sigma_fraction=0.075,
            metal_preprocess_clahe_clip=3.5,
            metal_preprocess_clahe_grid=12,
            metal_preprocess_denoise="high",
        )

        loaded = ContourExtractionSettings.from_dict(settings.to_dict())

        self.assertFalse(loaded.metal_preprocess_subtract_background)
        self.assertEqual(loaded.metal_preprocess_background_sigma_fraction, 0.075)
        self.assertEqual(loaded.metal_preprocess_clahe_clip, 3.5)
        self.assertEqual(loaded.metal_preprocess_clahe_grid, 12)
        self.assertEqual(loaded.metal_preprocess_denoise, "high")

    def test_structural_metal_recognition_skips_sem_preprocessing(self) -> None:
        source_image = np.full((32, 40), 90, dtype=np.uint8)
        settings = ContourExtractionSettings(
            recognition_mode="conductors",
            extraction_profile="conductors",
            object_type="conductor",
            output_mode="polygon",
            algorithm_backend="legacy",
            metal_structural_pipeline=True,
            metal_preprocess_subtract_background=False,
            metal_preprocess_background_sigma_fraction=0.075,
            metal_preprocess_clahe_clip=3.5,
            metal_preprocess_clahe_grid=12,
            metal_preprocess_denoise="high",
        )

        from contour.vision import preprocessing

        with patch.object(preprocessing, "preprocess_for_sem", wraps=preprocessing.preprocess_for_sem) as apply:
            result = process_image_path(
                image_path="sample.png",
                pipeline_config={"steps": []},
                contour_settings=settings,
                source_image=source_image,
            )

        apply.assert_not_called()
        self.assertIn("metal_preprocessed", result.debug_gradient_maps)
        np.testing.assert_array_equal(result.debug_gradient_maps["metal_preprocessed"], source_image)

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

    def test_via_search_mode_defaults_to_heuristic_when_absent_from_payload(self) -> None:
        loaded = ContourExtractionSettings.from_dict({"via_min_score": 0.5})
        self.assertEqual(loaded.via_search_mode, "heuristic")

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

    def test_conductor_recognition_uses_preprocessed_image_not_source(self) -> None:
        source_image = np.zeros((80, 100), dtype=np.uint8)
        cv2.rectangle(source_image, (25, 20), (74, 59), 220, thickness=-1)
        preprocessed = np.zeros_like(source_image)
        cv2.rectangle(preprocessed, (25, 20), (74, 59), 255, thickness=-1)

        empty_preprocessed = np.zeros_like(source_image)
        result_from_source = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(min_area=10.0, epsilon=1.0, min_polygon_angle=0.0),
            source_image=source_image,
            preprocessed_image=empty_preprocessed,
        )
        result_from_pipeline = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(min_area=10.0, epsilon=1.0, min_polygon_angle=0.0),
            source_image=np.zeros_like(source_image),
            preprocessed_image=preprocessed,
        )

        self.assertEqual(len(result_from_source.polygons), 0)
        self.assertEqual(len(result_from_pipeline.polygons), 1)
        self.assertEqual(result_from_pipeline.polygons[0].bbox, (25, 20, 50, 40))
