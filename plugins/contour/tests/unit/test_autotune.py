from __future__ import annotations

import unittest

import cv2
import numpy as np

from contour.application.use_cases.autotune import auto_tune_pipeline
from contour.contour_extractor import extract_polygons
from contour.domain import PolygonData, compute_polygon_metrics
from contour.pipeline import PreprocessingPipeline


def _rectangle_polygon(left: int, top: int, right: int, bottom: int) -> PolygonData:
    points = [
        (float(left), float(top)),
        (float(right), float(top)),
        (float(right), float(bottom)),
        (float(left), float(bottom)),
    ]
    area, perimeter, bbox = compute_polygon_metrics(points)
    return PolygonData(id=1, points=points, area=area, perimeter=perimeter, bbox=bbox)


def _render_mask(image_shape: tuple[int, int], polygons: list[PolygonData]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    for polygon in sorted(polygons, key=lambda item: item.is_hole):
        points = np.asarray(
            [[round(x_coord), round(y_coord)] for x_coord, y_coord in polygon.points],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points.reshape((-1, 1, 2))], 0 if polygon.is_hole else 255)
    return mask


def _mask_iou(first_mask: np.ndarray, second_mask: np.ndarray) -> float:
    first = first_mask > 0
    second = second_mask > 0
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


def _operations(result_pipeline_config: dict[str, object]) -> list[str]:
    return [str(step.get("operation", "")) for step in result_pipeline_config.get("steps", [])]


def _polygons_from_mask(mask: np.ndarray) -> list[PolygonData]:
    return extract_polygons(mask, None)


class AutoTuneTests(unittest.TestCase):
    def test_auto_tune_finds_configuration_for_bright_object(self) -> None:
        image = np.full((96, 96), 25, dtype=np.uint8)
        cv2.rectangle(image, (22, 18), (71, 64), 220, thickness=-1)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        reference = [_rectangle_polygon(22, 18, 71, 64)]

        result = auto_tune_pipeline(image, reference)

        processed = PreprocessingPipeline.from_dict(result.pipeline_config).apply(image)
        polygons = extract_polygons(processed, result.contour_settings)
        predicted_mask = _render_mask(image.shape, polygons)
        expected_mask = _render_mask(image.shape, reference)

        self.assertGreater(result.score, 0.80)
        self.assertGreater(_mask_iou(expected_mask, predicted_mask), 0.88)

    def test_auto_tune_handles_dark_object_on_bright_background(self) -> None:
        image = np.full((96, 96), 215, dtype=np.uint8)
        cv2.rectangle(image, (20, 24), (68, 70), 35, thickness=-1)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        reference = [_rectangle_polygon(20, 24, 68, 70)]

        result = auto_tune_pipeline(image, reference)

        processed = PreprocessingPipeline.from_dict(result.pipeline_config).apply(image)
        polygons = extract_polygons(processed, result.contour_settings)
        predicted_mask = _render_mask(image.shape, polygons)
        expected_mask = _render_mask(image.shape, reference)

        self.assertGreater(result.score, 0.80)
        self.assertGreater(_mask_iou(expected_mask, predicted_mask), 0.88)

    def test_auto_tune_uses_color_segmentation_when_grayscale_is_ambiguous(self) -> None:
        image = np.zeros((96, 96, 3), dtype=np.uint8)
        image[:] = (0, 130, 0)
        cv2.rectangle(image, (18, 20), (74, 72), (0, 0, 255), thickness=-1)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        reference = [_rectangle_polygon(18, 20, 74, 72)]

        result = auto_tune_pipeline(image, reference)

        processed = PreprocessingPipeline.from_dict(result.pipeline_config).apply(image)
        polygons = extract_polygons(processed, result.contour_settings)
        predicted_mask = _render_mask(image.shape[:2], polygons)
        expected_mask = _render_mask(image.shape[:2], reference)

        self.assertIn("color_binarize", _operations(result.pipeline_config))
        self.assertGreater(result.score, 0.80)
        self.assertGreater(_mask_iou(expected_mask, predicted_mask), 0.90)

    def test_auto_tune_can_split_touching_blobs(self) -> None:
        image = np.full((128, 128), 20, dtype=np.uint8)
        cv2.circle(image, (44, 64), 14, 220, thickness=-1)
        cv2.circle(image, (76, 64), 14, 220, thickness=-1)
        cv2.line(image, (57, 64), (63, 64), 180, thickness=7)
        image = cv2.GaussianBlur(image, (9, 9), 0)

        reference_mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.circle(reference_mask, (44, 64), 12, 255, thickness=-1)
        cv2.circle(reference_mask, (76, 64), 12, 255, thickness=-1)
        reference = _polygons_from_mask(reference_mask)

        result = auto_tune_pipeline(image, reference)

        processed = PreprocessingPipeline.from_dict(result.pipeline_config).apply(image)
        polygons = extract_polygons(processed, result.contour_settings)
        predicted_mask = _render_mask(image.shape, polygons)
        expected_mask = _render_mask(image.shape, reference)

        self.assertEqual(sum(1 for polygon in polygons if not polygon.is_hole), 2)
        self.assertGreater(_mask_iou(expected_mask, predicted_mask), 0.75)


if __name__ == "__main__":
    unittest.main()
