"""test_processing_use_cases: test via profile detects 2."""
from __future__ import annotations
import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings
from contour.application.use_cases.processing import process_image_path


class ViaProfileDetects2Mixin:
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
                via_search_mode="template",
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
        accepted = [
            candidate
            for candidate in result.debug_candidates
            if candidate.accepted and candidate.source == "template"
        ]
        self.assertTrue(accepted)
        self.assertTrue(all(candidate.template_index == 0 for candidate in accepted))
        self.assertTrue(all(0.0 <= candidate.score <= 100.0 for candidate in accepted))
