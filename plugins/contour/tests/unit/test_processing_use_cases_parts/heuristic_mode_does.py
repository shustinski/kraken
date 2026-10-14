"""test_processing_use_cases: test heuristic mode does."""
from __future__ import annotations
import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings
from contour.application.use_cases.processing import process_image_path


class HeuristicModeDoesMixin:
    def test_heuristic_mode_does_not_implicitly_use_templates(self) -> None:
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

        self.assertEqual(len(with_template.polygons), len(without_template.polygons))
        self.assertEqual(
            [(candidate.bbox, candidate.accepted) for candidate in with_template.debug_candidates],
            [(candidate.bbox, candidate.accepted) for candidate in without_template.debug_candidates],
        )

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
