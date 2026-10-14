"""test_processing_use_cases: test changing bright final."""
from __future__ import annotations
import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings
from contour.application.use_cases.processing import _core as processing_core
from contour.application.use_cases.processing import process_image_path


class ChangingBrightFinalMixin:
    def test_changing_bright_final_score_reuses_heavy_detection_cache(self) -> None:
        processing_core._VIA_DETECTION_CACHE.clear()
        source_image = np.full((120, 120), 90, dtype=np.uint8)
        cv2.circle(source_image, (36, 36), 4, 235, thickness=-1)
        cv2.circle(source_image, (82, 70), 4, 235, thickness=-1)
        settings = dict(
            extraction_profile="vias",
            object_type="via",
            output_mode="box",
            via_search_mode="bright_tophat_dog",
            bright_via_use_metal_mask=False,
            bright_via_min_final_score=20.0,
        )
        process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(**settings),
            source_image=source_image,
        )
        process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(**{**settings, "bright_via_min_final_score": 60.0}),
            source_image=source_image,
        )
        bright_cache_entries = [k for k in processing_core._VIA_DETECTION_CACHE if k[0] == "bright_tophat_dog"]
        self.assertEqual(len(bright_cache_entries), 1)

    def test_changing_bright_tophat_kernel_invalidates_heavy_detection_cache(self) -> None:
        processing_core._VIA_DETECTION_CACHE.clear()
        source_image = np.full((120, 120), 90, dtype=np.uint8)
        cv2.circle(source_image, (36, 36), 4, 235, thickness=-1)
        settings = dict(
            extraction_profile="vias",
            object_type="via",
            output_mode="box",
            via_search_mode="bright_tophat_dog",
            bright_via_use_metal_mask=False,
            bright_via_min_final_score=20.0,
        )
        process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(**{**settings, "bright_via_tophat_kernel_size": 9}),
            source_image=source_image,
        )
        process_image_path(
            image_path="sample.png",
            pipeline_config={"steps": []},
            contour_settings=ContourExtractionSettings(**{**settings, "bright_via_tophat_kernel_size": 15}),
            source_image=source_image,
        )
        bright_cache_entries = [k for k in processing_core._VIA_DETECTION_CACHE if k[0] == "bright_tophat_dog"]
        self.assertEqual(len(bright_cache_entries), 2)
