from __future__ import annotations

import unittest

import cv2
import numpy as np

from polygon_widget.application.processing import ContourExtractionSettings
from polygon_widget.application.use_cases.processing import process_image_path


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
        self.assertTrue(any(candidate.reason == "roundness" for candidate in result.debug_candidates))


if __name__ == "__main__":
    unittest.main()
