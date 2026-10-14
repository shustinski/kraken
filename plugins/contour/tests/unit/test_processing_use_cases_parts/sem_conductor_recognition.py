"""test_processing_use_cases: test sem conductor recognition."""
from __future__ import annotations
import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings
from contour.application.use_cases.processing import process_image_path


class SemConductorRecognitionMixin:
    def test_sem_conductor_recognition_uses_preprocessed_image(self) -> None:
        source_image = np.zeros((80, 100), dtype=np.uint8)
        cv2.rectangle(source_image, (5, 5), (30, 30), 220, thickness=-1)
        preprocessed = np.zeros_like(source_image)
        cv2.rectangle(preprocessed, (50, 40), (75, 65), 220, thickness=-1)

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                object_type="conductor",
                output_mode="polygon",
                min_area=10.0,
                epsilon=1.0,
                min_polygon_angle=0.0,
                algorithm_backend="sem",
            ),
            source_image=source_image,
            preprocessed_image=preprocessed,
        )

        self.assertGreaterEqual(len(result.polygons), 1)
        self.assertGreaterEqual(result.polygons[0].bbox[0], 45)
