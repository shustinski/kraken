"""heuristic_detector: detect vias heuristic."""
from __future__ import annotations
import time
from collections import Counter
import cv2
import numpy as np
from contour.vision.via_detection.binary_mask import detect_binary_components, looks_like_binary_mask
from contour.vision.via_detection.config import HeuristicViaDetectorConfig, ViaPolarity
from contour.vision.via_detection.result import DetectionResult, ViaDetection

from contour.vision.via_detection.heuristic_detector_parts.candidate_features import PreparedHeuristicFrame

from contour.vision.via_detection.heuristic_detector_parts.debug_viz import _debug_viz

from contour.vision.via_detection.heuristic_detector_parts.intensity_range_seeds import (
    _dedupe_by_score,
    _intensity_range_seeds,
    _local_extrema_seeds,
    _merge_seeds,
    _nms_simple,
    _spread_points,
)

from contour.vision.via_detection.heuristic_detector_parts.score_seed import (
    _normalize01_to_u8,
    _preprocess_denoise,
    _score_seed,
    _to_gray_u8,
)

def detect_vias_heuristic(image: np.ndarray, config: HeuristicViaDetectorConfig) -> DetectionResult:
    started_at = time.perf_counter()
    config.validate()
    g = _to_gray_u8(image)
    h, w = g.shape[:2]
    snap0 = dict(config.snapshot())
    if h < 3 or w < 3:
        return DetectionResult(method="heuristic", accepted=[], parameters_snapshot=snap0)

    allowed = config.allowed_diameters()
    if not allowed:
        return DetectionResult(method="heuristic", accepted=[], parameters_snapshot=snap0)
    d_min, d_max = min(allowed), max(allowed)
    if looks_like_binary_mask(g):
        _mask, components = detect_binary_components(
            g,
            diameter_min=d_min,
            diameter_max=d_max,
            min_area_factor=0.35,
            max_area_factor=2.0,
            min_aspect=1.0 / max(1.0, float(config.max_elongation)),
            max_aspect=max(1.0, float(config.max_elongation)),
            nms_distance=config.nms_distance,
        )
        accepted = [
            ViaDetection(
                x=item.center[0],
                y=item.center[1],
                bbox=item.bbox,
                score=100.0,
                diameter_estimate=0.5 * (item.bbox[2] + item.bbox[3]),
                contrast=255.0,
                prominence=255.0,
                compactness=item.circularity,
                aspect=item.aspect,
                polarity_hypothesis="binary",
            )
            for item in components
        ]
        return DetectionResult(
            method="heuristic_binary",
            accepted=accepted,
            debug_images={"binary_mask": _mask},
            parameters_snapshot={**snap0, "binary_mask": True, "accepted_count": len(accepted)},
        )

    prepare_started = time.perf_counter()
    seed_threshold = {"percentile": float(config.seed_percentile)}
    g_pre = _preprocess_denoise(g, config)
    bg = cv2.GaussianBlur(g_pre, (0, 0), float(config.background_sigma))
    gray_f32 = g_pre.astype(np.float32)
    corr = gray_f32 - bg.astype(np.float32)
    corr_u8 = _normalize01_to_u8(corr)
    gradient_x = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
    frame = PreparedHeuristicFrame(
        gray=g_pre,
        gray_f32=gray_f32,
        corrected_u8=corr_u8,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
        gradient_magnitude=cv2.magnitude(gradient_x, gradient_y),
    )
    prepare_seconds = time.perf_counter() - prepare_started

    bright_map = corr_u8
    dark_map = 255 - corr_u8

    ksize = int(max(3, 2 * round(0.5 * d_max) + 1))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    seeds_b, mask_b = _local_extrema_seeds(
        bright_map,
        ker,
        seed_threshold["percentile"],
        float(config.min_peak_grey),
    )
    seeds_d, mask_d = _local_extrema_seeds(
        dark_map,
        ker,
        seed_threshold["percentile"],
        float(config.min_peak_grey),
    )

    # An explicitly enabled intensity range is a candidate-generation rule, not
    # merely a late validation gate. Otherwise dense frames keep only the top
    # response percentile and valid, slightly weaker vias never get scored.
    if config.bright_range_enabled and (
        float(config.bright_range_min) >= 128.0 or config.use_intensity_range_seeds
    ):
        range_seeds, range_mask = _intensity_range_seeds(
            g_pre,
            ker,
            float(config.bright_range_min),
            float(config.bright_range_max),
            d_min,
            d_max,
            bright=True,
        )
        seeds_b.extend(range_seeds)
        mask_b = cv2.bitwise_or(mask_b, range_mask)
    if config.dark_range_enabled and (
        float(config.dark_range_max) <= 127.0 or config.use_intensity_range_seeds
    ):
        range_seeds, range_mask = _intensity_range_seeds(
            g_pre,
            ker,
            float(config.dark_range_min),
            float(config.dark_range_max),
            d_min,
            d_max,
            bright=False,
        )
        seeds_d.extend(range_seeds)
        mask_d = cv2.bitwise_or(mask_d, range_mask)

    polar = str(config.polarity or ViaPolarity.AUTO).lower()
    min_sep = int(config.min_distance_between_peaks) if config.min_distance_between_peaks else max(2, d_min // 2)
    if polar in (str(ViaPolarity.BRIGHT), "bright"):
        raw_seeds = _spread_points(seeds_b, min_sep, h, w)
    elif polar in (str(ViaPolarity.DARK), "dark"):
        raw_seeds = _spread_points(seeds_d, min_sep, h, w)
    else:
        raw_seeds = _merge_seeds(list(seeds_b) + list(seeds_d), min_sep, h, w)
    hyps: list[str]
    if polar in ("auto", str(ViaPolarity.AUTO)):
        hyps = []
        if config.bright_range_enabled:
            hyps.append(str(ViaPolarity.BRIGHT))
        if config.dark_range_enabled:
            hyps.append(str(ViaPolarity.DARK))
        if not hyps:
            hyps = [str(ViaPolarity.BRIGHT)]
    elif polar in (
        str(ViaPolarity.RING_LIGHT_RING),
        str(ViaPolarity.RING_DARK_RING),
        ViaPolarity.RING_LIGHT_RING,
        ViaPolarity.RING_DARK_RING,
    ):
        hyps = [polar] if not isinstance(polar, ViaPolarity) else [str(polar)]
    else:
        hyps = [polar]

    score_started = time.perf_counter()
    dets = [
        item
        for seed in raw_seeds
        if (item := _score_seed(frame, seed, allowed, hyps, config)) is not None
    ]

    scored_count = len(dets)
    dets = [d for d in dets if d.reject_reason is None or (d.reject_reason and "hard" in d.reject_reason)]
    dets = _dedupe_by_score(dets, min_dist=1.0)
    dets.sort(key=lambda d: d.score, reverse=True)
    effective_nms = max(0, int(config.nms_distance), round(0.6 * d_max))
    after = _nms_simple(dets, effective_nms)

    accepted = [
        d for d in after if d.reject_reason is None and d.score >= float(config.min_final_score)
    ]
    below: list[ViaDetection] = [
        d for d in after if d.reject_reason is None and 0 < d.score < float(config.min_final_score)
    ]
    hard = [d for d in after if d.reject_reason and "hard" in d.reject_reason]

    scoring_seconds = time.perf_counter() - score_started
    dbg = _debug_viz(
        g_pre,
        corr_u8,
        mask_b,
        mask_d,
        accepted,
        below,
        hard,
        d_max,
    )

    reject_counts = Counter(
        str(item.reject_reason).split("(", 1)[0].removeprefix("hard:") for item in hard if item.reject_reason
    )
    total_seconds = time.perf_counter() - started_at
    return DetectionResult(
        method="heuristic",
        accepted=accepted,
        rejected=hard,
        below_threshold=below,
        debug_images=dbg,
        parameters_snapshot={
            **snap0,
            "raw_seed_count": len(raw_seeds),
            "scored_candidate_count": scored_count,
            "candidate_count_after_dedupe": len(dets),
            "candidate_count_after_nms": len(after),
            "effective_nms_distance": effective_nms,
            "accepted_count": len(accepted),
            "below_threshold_count": len(below),
            "hard_rejected_count": len(hard),
            "reject_counts": dict(sorted(reject_counts.items())),
            "seed_threshold": seed_threshold,
            "timings_seconds": {
                "prepare": prepare_seconds,
                "score": scoring_seconds,
                "total": total_seconds,
            },
            "scoring_workers": 1,
        },
    )
