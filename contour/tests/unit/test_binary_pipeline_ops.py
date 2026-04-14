from __future__ import annotations

import unittest

import cv2
import numpy as np

from polygon_widget.application.processing import PipelineStepConfig
from polygon_widget.pipeline import PreprocessingPipeline


class BinaryPipelineOperationTests(unittest.TestCase):
    def test_color_binarize_selects_pixels_within_delta(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[1, 1] = (16, 16, 16)
        image[1, 2] = (22, 22, 22)
        image[2, 2] = (40, 40, 40)

        pipeline = PreprocessingPipeline(
            [
                PipelineStepConfig(
                    operation="color_binarize",
                    name="Color Binarize",
                    parameters={
                        "delta": 10,
                        "selected_colors": [{"rgb": [16, 16, 16], "enabled": True}],
                    },
                )
            ]
        )

        result = pipeline.apply(image)

        self.assertEqual(int(result[1, 1]), 255)
        self.assertEqual(int(result[1, 2]), 255)
        self.assertEqual(int(result[2, 2]), 0)

    def test_binary_fill_holes_fills_inner_voids(self) -> None:
        mask = np.zeros((32, 32), dtype=np.uint8)
        cv2.rectangle(mask, (4, 4), (27, 27), 255, thickness=-1)
        cv2.rectangle(mask, (11, 11), (20, 20), 0, thickness=-1)

        pipeline = PreprocessingPipeline([PipelineStepConfig(operation="binary_fill_holes", name="Fill Holes")])
        result = pipeline.apply(mask)

        self.assertEqual(int(result[15, 15]), 255)

    def test_binary_filter_area_removes_small_components(self) -> None:
        mask = np.zeros((40, 40), dtype=np.uint8)
        cv2.rectangle(mask, (2, 2), (8, 8), 255, thickness=-1)
        cv2.rectangle(mask, (15, 15), (34, 34), 255, thickness=-1)

        pipeline = PreprocessingPipeline(
            [
                PipelineStepConfig(
                    operation="binary_filter_area",
                    name="Filter By Area",
                    parameters={"min_component_area": 100.0, "max_component_area": 0.0},
                )
            ]
        )
        result = pipeline.apply(mask)

        self.assertEqual(int(result[5, 5]), 0)
        self.assertEqual(int(result[20, 20]), 255)

    def test_binary_filter_perimeter_removes_long_component(self) -> None:
        mask = np.zeros((60, 60), dtype=np.uint8)
        cv2.rectangle(mask, (5, 5), (15, 15), 255, thickness=-1)
        cv2.rectangle(mask, (20, 20), (55, 25), 255, thickness=-1)

        pipeline = PreprocessingPipeline(
            [
                PipelineStepConfig(
                    operation="binary_filter_perimeter",
                    name="Filter By Perimeter",
                    parameters={"min_component_perimeter": 0.0, "max_component_perimeter": 50.0},
                )
            ]
        )
        result = pipeline.apply(mask)

        self.assertEqual(int(result[10, 10]), 255)
        self.assertEqual(int(result[22, 22]), 0)

    def test_watershed_split_separates_touching_components(self) -> None:
        mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.circle(mask, (38, 48), 18, 255, thickness=-1)
        cv2.circle(mask, (58, 48), 18, 255, thickness=-1)

        pipeline = PreprocessingPipeline(
            [
                PipelineStepConfig(
                    operation="watershed_split",
                    name="Watershed Split",
                    parameters={
                        "distance_ratio": 0.35,
                        "min_peak_area": 1,
                        "kernel_size": 3,
                        "shape": "ellipse",
                        "background_iterations": 1,
                    },
                )
            ]
        )
        result = pipeline.apply(mask)
        count, _labels = cv2.connectedComponents((result > 0).astype(np.uint8), connectivity=8)

        self.assertEqual(count - 1, 2)


if __name__ == "__main__":
    unittest.main()
