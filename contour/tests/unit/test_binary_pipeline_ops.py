from __future__ import annotations

import unittest

import cv2
import numpy as np

from polygon_widget.application.processing import PipelineStepConfig
from polygon_widget.pipeline import PreprocessingPipeline


def _mask_iou(first_mask: np.ndarray, second_mask: np.ndarray) -> float:
    first = first_mask > 0
    second = second_mask > 0
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


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

    def test_edge_guided_threshold_refines_filled_mask_to_intensity_edges(self) -> None:
        image = np.full((80, 80), 25, dtype=np.uint8)
        cv2.rectangle(image, (22, 18), (57, 61), 220, thickness=-1)
        image = cv2.GaussianBlur(image, (11, 11), 0)

        expected = np.zeros_like(image)
        cv2.rectangle(expected, (22, 18), (57, 61), 255, thickness=-1)
        plain_pipeline = PreprocessingPipeline(
            [
                PipelineStepConfig(
                    operation="threshold",
                    name="Threshold",
                    parameters={"threshold": 180.0, "max_value": 255.0, "threshold_type": "binary"},
                )
            ]
        )
        refined_pipeline = PreprocessingPipeline(
            [
                PipelineStepConfig(
                    operation="edge_guided_threshold",
                    name="Edge-guided Threshold",
                    parameters={
                        "threshold_mode": "manual",
                        "threshold": 180.0,
                        "max_value": 255.0,
                        "threshold_type": "binary",
                        "edge_detector": "canny",
                        "correction_radius": 5,
                        "aperture_size": 3,
                        "fill_holes": True,
                    },
                )
            ]
        )

        plain = plain_pipeline.apply(image)
        refined = refined_pipeline.apply(image)

        self.assertGreater(cv2.countNonZero(refined), cv2.countNonZero(plain))
        self.assertGreater(_mask_iou(expected, refined), _mask_iou(expected, plain))
        contours, _hierarchy = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.assertEqual(len(contours), 1)


if __name__ == "__main__":
    unittest.main()
