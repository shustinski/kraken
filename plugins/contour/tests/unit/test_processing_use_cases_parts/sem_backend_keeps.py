"""test_processing_use_cases: test sem backend keeps."""
from __future__ import annotations
import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings
from contour.application.use_cases.processing import process_image_path


class SemBackendKeepsMixin:
    def test_sem_backend_via_detector_runs_independently_from_conductors(self) -> None:
        source_image = np.full((80, 80), 60, dtype=np.uint8)
        cv2.circle(source_image, (40, 40), 5, 220, thickness=-1)

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                algorithm_backend="sem",
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_size_mode="fixed",
                fixed_via_widths=[11],
                fixed_via_heights=[11],
                via_white_range_enabled=True,
                via_white_range_min=180,
                via_white_range_max=255,
                via_black_range_enabled=False,
                via_min_score=0.25,
                via_min_contrast=8.0,
                via_min_edge_coverage=0.0,
                debug_enabled=True,
            ),
            source_image=source_image,
        )

        self.assertEqual(len(result.polygons), 1)
        self.assertEqual(result.polygons[0].category, "via")
        self.assertEqual(result.polygons[0].shape_hint, "box")
        self.assertTrue(any(candidate.source == "heuristic" for candidate in result.debug_candidates))

    def test_sem_backend_keeps_template_strategy_selectable(self) -> None:
        source_image = np.full((80, 120), 60, dtype=np.uint8)
        for x_coord in (30, 70):
            cv2.circle(source_image, (x_coord, 40), 7, 210, thickness=-1)
        template = source_image[33:48, 23:38].copy()

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                algorithm_backend="sem",
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_search_mode="template",
                via_size_mode="fixed",
                fixed_via_widths=[15],
                fixed_via_heights=[15],
                via_white_range_enabled=True,
                via_white_range_min=0,
                via_white_range_max=255,
                via_black_range_enabled=False,
                via_template_images=[template],
                via_template_min_score=0.5,
                via_template_min_scores=[0.5],
                via_template_diameters=[23],
                via_output_diameter=23,
                via_min_score=0.95,
                via_min_contrast=0.0,
                via_min_edge_coverage=0.0,
                via_min_roundness=0.0,
                debug_enabled=True,
            ),
            source_image=source_image,
        )

        self.assertEqual(len(result.polygons), 2)
        self.assertTrue(
            all(
                max(point[0] for point in polygon.points) - min(point[0] for point in polygon.points) == 23
                for polygon in result.polygons
            )
        )
        self.assertTrue(any(candidate.source == "template" for candidate in result.debug_candidates))

    def test_sem_backend_conductor_can_return_boxes(self) -> None:
        source_image = np.full((128, 128), 180, dtype=np.uint8)
        cv2.rectangle(source_image, (32, 36), (94, 82), 40, thickness=-1)

        result = process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(
                algorithm_backend="sem",
                extraction_profile="conductors",
                object_type="conductor",
                output_mode="box",
                min_area=20.0,
            ),
            source_image=source_image,
        )

        self.assertTrue(result.polygons)
        self.assertTrue(all(polygon.shape_hint == "box" for polygon in result.polygons))
