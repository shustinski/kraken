"""heuristic_detector: segment candidate."""
from __future__ import annotations
import math
import cv2
import numpy as np
from contour.vision.via_detection.config import HeuristicViaDetectorConfig, ViaPolarity
from contour.vision.via_detection.result import ViaDetection

from contour.vision.via_detection.heuristic_detector_parts.candidate_features import (
    SegmentedCandidate,
    ViaRejectCode,
    _hard_reason,
)

from contour.vision.via_detection.heuristic_detector_parts.local_structure_metrics import (
    _annulus_masks,
    _candidate_bbox,
    _center_in_brightness_range,
    _contrast_for_polarity,
    _local_center_background_contrast,
)

def _segment_candidate(
    patch: np.ndarray,
    pcx: int,
    pcy: int,
    diameter: int,
    offset: tuple[int, int],
    hypothesis: str,
    config: HeuristicViaDetectorConfig,
    *,
    enforce_rejections: bool = True,
) -> tuple[SegmentedCandidate | None, ViaDetection | None]:
    """Build the local component and return either its data or a typed rejection."""

    from contour.vision.via_detection.heuristic_detector_parts.score_seed import _fast_percentile_u8
    height, width = patch.shape[:2]
    center_mask, inner_mask, outer_mask = _annulus_masks(
        (height, width), pcx, pcy, float(diameter)
    )
    center_grey = float(patch[pcy, pcx])

    def rejected(code: ViaRejectCode, *, contrast: float = 0.0, prominence: float = 0.0) -> ViaDetection:
        return ViaDetection(
            float(offset[0] + pcx),
            float(offset[1] + pcy),
            _candidate_bbox(offset[0] + pcx, offset[1] + pcy, diameter),
            0.0,
            float(diameter),
            float(contrast),
            float(prominence),
            0.0,
            0.0,
            hypothesis,
            _hard_reason(code),
        )

    if enforce_rejections and not _center_in_brightness_range(center_grey, hypothesis, config):
        return None, rejected(ViaRejectCode.BRIGHTNESS_RANGE)
    if enforce_rejections and center_grey < float(config.min_center_brightness):
        return None, rejected(ViaRejectCode.LOW_CENTER_BRIGHTNESS)
    annular_contrast = _contrast_for_polarity(
        patch, center_mask, inner_mask, outer_mask, hypothesis
    )
    background_contrast = _local_center_background_contrast(
        patch, pcx, pcy, float(diameter), hypothesis
    )
    contrast = max(float(annular_contrast), float(background_contrast))
    if enforce_rejections and contrast < float(config.min_center_contrast):
        return None, rejected(ViaRejectCode.LOW_CONTRAST, contrast=contrast)

    median = _fast_percentile_u8(patch, 50.0)
    peak_radius = min(4, max(1, max(height, width) // 6))
    y0, y1 = max(0, pcy - peak_radius), min(height, pcy + peak_radius + 1)
    x0, x1 = max(0, pcx - peak_radius), min(width, pcx + peak_radius + 1)
    neighbourhood = patch[y0:y1, x0:x1].ravel()
    prominence = float(np.max(np.abs(neighbourhood - median))) if neighbourhood.size else 0.0
    if enforce_rejections and prominence < float(config.min_peak_prominence):
        return None, rejected(
            ViaRejectCode.LOW_PROMINENCE,
            contrast=contrast,
            prominence=prominence,
        )

    percentile = float(config.local_binarize_percentile)
    segmentation_contrast_floor = (
        float(config.min_center_contrast)
        if enforce_rejections or contrast >= float(config.min_center_contrast)
        else 0.0
    )
    delta = max(segmentation_contrast_floor, abs(contrast) * 0.30, 2.0)
    dark_component = hypothesis in (
        str(ViaPolarity.DARK),
        "dark",
        str(ViaPolarity.RING_DARK_RING),
        "ring_dark_ring",
    )
    ring_component = hypothesis in (
        str(ViaPolarity.RING_LIGHT_RING),
        "ring_light_ring",
        str(ViaPolarity.RING_DARK_RING),
        "ring_dark_ring",
    )
    clamp_dark_seed = enforce_rejections and (
        config.use_intensity_range_seeds or float(config.dark_range_max) <= 127.0
    )
    clamp_bright_seed = enforce_rejections and (
        config.use_intensity_range_seeds or float(config.bright_range_min) >= 128.0
    )
    if dark_component:
        threshold = min(
            _fast_percentile_u8(patch, max(1.0, 100.0 - percentile)),
            median - delta,
        )
        if config.dark_range_enabled and clamp_dark_seed:
            threshold = min(threshold, float(config.dark_range_max))
            if not ring_component and clamp_dark_seed:
                threshold = max(center_grey, threshold)
                if center_grey < threshold:
                    threshold = max(center_grey, math.ceil(threshold) - 1.0)
            binary = (
                (patch >= float(config.dark_range_min)) & (patch <= threshold)
            ).astype(np.uint8) * 255
        else:
            if not ring_component and clamp_dark_seed:
                threshold = max(center_grey, threshold)
                if center_grey < threshold:
                    threshold = max(center_grey, math.ceil(threshold) - 1.0)
            binary = (patch <= threshold).astype(np.uint8) * 255
    else:
        threshold = max(_fast_percentile_u8(patch, percentile), median + delta)
        if (
            hypothesis in (str(ViaPolarity.BRIGHT), "bright")
            and config.bright_range_enabled
            and clamp_bright_seed
        ):
            threshold = max(threshold, float(config.bright_range_min))
            if not ring_component and clamp_bright_seed:
                threshold = min(center_grey, threshold)
                if center_grey > threshold:
                    threshold = min(center_grey, math.floor(threshold) + 1.0)
            binary = (
                (patch >= threshold) & (patch <= float(config.bright_range_max))
            ).astype(np.uint8) * 255
        else:
            if not ring_component and clamp_bright_seed:
                threshold = min(center_grey, threshold)
                if center_grey > threshold:
                    threshold = min(center_grey, math.floor(threshold) + 1.0)
            binary = (patch >= threshold).astype(np.uint8) * 255

    _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if hypothesis in (
        str(ViaPolarity.RING_LIGHT_RING),
        "ring_light_ring",
        str(ViaPolarity.RING_DARK_RING),
        "ring_dark_ring",
    ):
        ring_labels = labels[inner_mask]
        ring_labels = ring_labels[ring_labels > 0]
        if ring_labels.size:
            label = int(np.bincount(ring_labels).argmax())
        else:
            label = 0
    else:
        label = int(labels[pcy, pcx])
    if label <= 0:
        return None, rejected(
            ViaRejectCode.NO_COMPONENT,
            contrast=contrast,
            prominence=prominence,
        )
    area = float(stats[label, cv2.CC_STAT_AREA])
    if area < 1.0:
        return None, None
    component_bool = labels == label
    component_mask = component_bool.astype(np.uint8) * 255
    contours, _hierarchy = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None
    return (
        SegmentedCandidate(
            gray=patch,
            median=median,
            contrast=contrast,
            prominence=prominence,
            area=area,
            component_mask=component_bool,
            contour=max(contours, key=cv2.contourArea),
            component_stats=stats[label],
            binarization_threshold=float(threshold),
        ),
        None,
    )
