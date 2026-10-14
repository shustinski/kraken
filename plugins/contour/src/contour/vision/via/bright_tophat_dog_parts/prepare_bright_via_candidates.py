"""bright_tophat_dog: prepare bright via candidates."""
from __future__ import annotations
from math import pi
import cv2
import numpy as np
from ....application.preview_cancellation import raise_if_preview_cancelled
from contour.vision.io_normalize import to_gray_u8

from contour.vision.via.bright_tophat_dog_parts.module_state import _MATCHED_PCT_OFFSET

from contour.vision.via.bright_tophat_dog_parts.build_isolation_map import (
    _bright_local_peak_mask,
    _build_isolation_map,
    _edge_distance_map,
    _metal_mask,
)

from contour.vision.via.bright_tophat_dog_parts.draw_overlay import (
    _normalize_metal_constraint_mode,
    _normalize_response,
    _normalize_u8,
)

from contour.vision.via.bright_tophat_dog_parts.local_maxima_points import _raw_nms

from contour.vision.via.bright_tophat_dog_parts.refine_center_bright_core import _stage1_raw_candidates

from contour.vision.via.bright_tophat_dog_parts.via_matched_filter_kernel import (
    _dog_response,
    _percentile_mask,
    _preprocess,
    _via_matched_filter_response,
    _white_tophat,
)

def prepare_bright_via_candidates(image: np.ndarray, config: BrightViaDetectorConfig) -> BrightViaPreparedCandidates:
    from contour.vision.via.bright_tophat_dog_parts.bright_via_detection import BrightViaPreparedCandidates
    cfg = config.validated()
    gray = _normalize_u8(to_gray_u8(image))
    if gray.size == 0:
        return BrightViaPreparedCandidates(
            gray=gray,
            processed=gray,
            tophat=gray,
            dog=gray,
            tophat_mask=gray,
            dog_mask=gray,
            via_mask=gray,
            candidate_mask=gray,
            metal_mask=gray,
            edge_distance=gray.astype(np.float32),
            raw_candidates=[],
            min_area=0.0,
            max_area=0.0,
            nominal_diameter=0.0,
            nominal_radius=0.0,
            metal_mode="disabled",
            isolation=None,
        )
    raise_if_preview_cancelled()
    processed = _preprocess(gray, cfg)
    raise_if_preview_cancelled()
    tophat = _white_tophat(processed, cfg.tophat_kernel_size)
    raise_if_preview_cancelled()
    dog = _dog_response(processed, cfg.dog_sigma_small, cfg.dog_sigma_large)
    raise_if_preview_cancelled()
    nominal_diameter = (float(cfg.diameter_min) + float(cfg.diameter_max)) * 0.5
    matched = _via_matched_filter_response(processed, nominal_diameter)
    matched_norm = _normalize_response(matched)
    tophat_norm = _normalize_response(tophat)
    dog_norm = _normalize_response(dog)
    tile_size = int(cfg.tiled_threshold_size)
    tophat_mask = _percentile_mask(tophat_norm, cfg.threshold_percentile)
    dog_mask = _percentile_mask(dog_norm, cfg.threshold_percentile)
    # The matched filter is the primary, brightness-robust candidate source.
    # A slightly lower tiled percentile keeps recall high; scoring prunes noise.
    # Global (not tiled) threshold: the matched response is zero-DC, so bright
    # metal stripes yield low response and do not inflate a local floor that would
    # hide vias sharing their tile.
    matched_mask = _percentile_mask(matched_norm, max(95.0, float(cfg.threshold_percentile) - _MATCHED_PCT_OFFSET))
    bright_peak_mask = _bright_local_peak_mask(gray, tile_size, cfg.threshold_percentile)
    if cfg.mask_combine_mode.upper() == "AND":
        via_mask = cv2.bitwise_and(tophat_mask, dog_mask)
    else:
        via_mask = cv2.bitwise_or(tophat_mask, dog_mask)
    if cv2.countNonZero(matched_mask) > 0:
        via_mask = cv2.bitwise_or(via_mask, matched_mask)
    if cv2.countNonZero(bright_peak_mask) > 0:
        via_mask = cv2.bitwise_or(via_mask, bright_peak_mask)
    raise_if_preview_cancelled()
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    via_mask = cv2.morphologyEx(via_mask, cv2.MORPH_OPEN, open_kernel)
    candidate_soup = cv2.bitwise_or(via_mask, cv2.bitwise_or(tophat_mask, dog_mask))
    metal_mode = "disabled" if not cfg.use_metal_mask else _normalize_metal_constraint_mode(cfg.metal_constraint_mode)
    metal_mask = _metal_mask(processed) if metal_mode != "disabled" else np.zeros_like(gray, dtype=np.uint8)
    edge_distance = _edge_distance_map(processed)
    min_area = pi * (float(cfg.diameter_min) * 0.5) ** 2 * float(cfg.min_area_factor)
    max_area = pi * (float(cfg.diameter_max) * 0.5) ** 2 * float(cfg.max_area_factor)
    nominal_radius = max(1.0, nominal_diameter * 0.5)
    isolation = _build_isolation_map(processed, round(nominal_diameter))
    raise_if_preview_cancelled()
    raws = _stage1_raw_candidates(
        gray_u8=gray,
        tophat_u8=tophat_norm,
        dog_u8=dog_norm,
        matched_u8=matched_norm,
        tophat_mask=tophat_mask,
        dog_mask=dog_mask,
        matched_mask=matched_mask,
        via_mask=via_mask,
        cfg=cfg,
        min_area=min_area,
        max_area=max_area,
    )
    raise_if_preview_cancelled()
    raws = _raw_nms(raws, float(cfg.nms_distance))
    return BrightViaPreparedCandidates(
        gray=gray,
        processed=processed,
        tophat=tophat_norm,
        dog=dog_norm,
        tophat_mask=tophat_mask,
        dog_mask=dog_mask,
        via_mask=via_mask,
        candidate_mask=candidate_soup,
        metal_mask=metal_mask,
        edge_distance=edge_distance,
        raw_candidates=raws,
        min_area=min_area,
        max_area=max_area,
        nominal_diameter=nominal_diameter,
        nominal_radius=nominal_radius,
        metal_mode=metal_mode,
        isolation=isolation,
    )
