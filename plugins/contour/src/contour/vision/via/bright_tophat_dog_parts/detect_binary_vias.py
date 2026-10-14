"""bright_tophat_dog: detect binary vias."""
from __future__ import annotations
import numpy as np
from ...via_detection.binary_mask import detect_binary_components

from contour.vision.via.bright_tophat_dog_parts.draw_overlay import _draw_overlay

from contour.vision.via.bright_tophat_dog_parts.via_matched_filter_kernel import _center_mean_score_f32

def _detect_binary_vias(gray: np.ndarray, cfg: BrightViaDetectorConfig) -> BrightViaDetectionResult:
    """Deterministic connected-component detector for binary mask inputs.

    Binary masks already contain the segmentation result.  Running CLAHE,
    top-hat and radial gates on them can only lose information, especially at
    frame borders.  Geometry is therefore the only filter on this path.
    """

    from contour.vision.via.bright_tophat_dog_parts.bright_via_detection import BrightViaDetection, BrightViaDetectionResult
    foreground, components = detect_binary_components(
        gray,
        diameter_min=cfg.diameter_min,
        diameter_max=cfg.diameter_max,
        min_area_factor=cfg.min_area_factor,
        max_area_factor=cfg.max_area_factor,
        min_aspect=cfg.min_aspect,
        max_aspect=cfg.max_aspect,
        nms_distance=cfg.nms_distance,
    )
    gray_f32 = gray.astype(np.float32)
    detections: list[BrightViaDetection] = []
    for component in components:
        cx, cy = component.center
        x, y, width, height = component.bbox
        detections.append(
            BrightViaDetection(
                center=(cx, cy),
                bbox=(x, y, width, height),
                area=component.area,
                circularity=component.circularity,
                aspect=component.aspect,
                brightness_score=float(_center_mean_score_f32(gray_f32, (cx, cy), min(width, height))),
                local_peak_score=255.0,
                tophat_response=255.0,
                dog_response=255.0,
                metal_fraction=1.0,
                final_score=100.0,
                annular_contrast=255.0,
                isolation_score=1.0,
                status="accepted",
            )
        )
    overlay = _draw_overlay(gray, detections, cfg, show_rejected=False)
    empty = np.zeros_like(gray, dtype=np.uint8)
    return BrightViaDetectionResult(
        detections=detections,
        candidates=list(detections),
        debug_images={
            "raw_gray": gray,
            "processed": gray,
            "via_mask": foreground,
            "candidate_mask": foreground,
            "tophat": empty,
            "dog": empty,
            "tophat_mask": empty,
            "dog_mask": empty,
            "metal_mask": empty,
            "radial_symmetry": empty,
            "edge_likeness": empty,
            "line_likeness": empty,
            "distance_to_edge": empty,
            "final_overlay": overlay,
        },
    )
