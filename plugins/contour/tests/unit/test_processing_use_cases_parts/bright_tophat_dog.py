"""test_processing_use_cases: test bright tophat dog."""
from __future__ import annotations
import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings
from contour.application.use_cases.processing import process_image_path


class BrightTophatDogMixin:
    def test_bright_tophat_dog_mode_works_without_templates(self) -> None:
        source_image = np.full((96, 96), 100, dtype=np.uint8)
        cv2.circle(source_image, (48, 48), 4, 240, thickness=-1)
        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_search_mode="bright_tophat_dog",
                via_template_images=[],
                bright_via_min_final_score=10.0,
                bright_via_min_isolation_score=0.0,
                bright_via_use_metal_mask=False,
            ),
            source_image=source_image,
        )
        self.assertGreaterEqual(len(result.polygons), 1)

    def test_template_mode_without_templates_returns_controlled_warning(self) -> None:
        source_image = np.full((64, 64), 120, dtype=np.uint8)
        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_search_mode="template",
                via_template_images=[],
                debug_enabled=True,
            ),
            source_image=source_image,
        )
        self.assertEqual(result.polygons, [])
        self.assertTrue(result.debug_candidates)
        self.assertIn("добавьте хотя бы один шаблон", result.debug_candidates[0].reason.lower())
