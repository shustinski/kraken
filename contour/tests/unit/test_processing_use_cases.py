from __future__ import annotations

import unittest

import numpy as np

from polygon_widget.application.processing import ContourExtractionSettings
from polygon_widget.application.use_cases.processing import process_image_path


class ProcessingUseCasesTests(unittest.TestCase):
    def test_contour_settings_disable_corner_preservation_by_default(self) -> None:
        settings = ContourExtractionSettings()
        loaded = ContourExtractionSettings.from_dict({})

        self.assertFalse(settings.preserve_corners)
        self.assertFalse(loaded.preserve_corners)

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


if __name__ == "__main__":
    unittest.main()
