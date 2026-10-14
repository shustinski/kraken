"""heuristic_detector: score one."""
from __future__ import annotations
import math
import cv2
import numpy as np
from contour.vision.via_detection.config import HeuristicViaDetectorConfig, ViaPolarity
from contour.vision.via_detection.result import ViaDetection

from contour.vision.via_detection.heuristic_detector_parts.candidate_features import (
    CandidateFeatures,
    ViaRejectCode,
    _hard_reason,
)

from contour.vision.via_detection.heuristic_detector_parts.candidate_score_metrics import _candidate_score_metrics

from contour.vision.via_detection.heuristic_detector_parts.final_candidate_score import _scale01

from contour.vision.via_detection.heuristic_detector_parts.local_structure_metrics import (
    _candidate_bbox,
    _component_geometry_center,
    _local_structure_metrics,
    _refine_center_xy,
)

from contour.vision.via_detection.heuristic_detector_parts.segment_candidate import _segment_candidate

def _score_one(
    patch: np.ndarray,
    pcx: int,
    pcy: int,
    d_est: int,
    offset: tuple[int, int],
    hyp: str,
    config: HeuristicViaDetectorConfig,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    gradient_magnitude: np.ndarray,
    allowed_diameters: list[int],
    *,
    enforce_rejections: bool = True,
) -> ViaDetection | None:
    h, w = patch.shape[:2]
    if pcx < 0 or pcy < 0 or pcx >= w or pcy >= h:
        return None
    ph, pw = h, w
    segmented, rejection = _segment_candidate(
        patch,
        pcx,
        pcy,
        d_est,
        offset,
        hyp,
        config,
        enforce_rejections=enforce_rejections,
    )
    if rejection is not None:
        return rejection
    if segmented is None:
        return None
    gpatch = segmented.gray
    med = segmented.median
    contrast = segmented.contrast
    prom = segmented.prominence
    area = segmented.area
    cnt0 = segmented.contour
    if hyp in (
        str(ViaPolarity.RING_LIGHT_RING),
        "ring_light_ring",
        str(ViaPolarity.RING_DARK_RING),
        "ring_dark_ring",
    ):
        _ring_center, ring_radius = cv2.minEnclosingCircle(cnt0)
        eq_diam = 2.0 * float(ring_radius)
    else:
        eq_diam = 2.0 * math.sqrt(max(area, 1.0) / math.pi)
    tol = config.effective_size_tolerance()
    re = max(float(d_est), 1.0)
    if enforce_rejections and abs(eq_diam - float(d_est)) / re > tol:
        return ViaDetection(
            float(offset[0] + pcx),
            float(offset[1] + pcy),
            _candidate_bbox(offset[0] + pcx, offset[1] + pcy, d_est),
            0.0,
            float(d_est),
            float(contrast),
            float(prom),
            0.0,
            0.0,
            hyp,
            _hard_reason(ViaRejectCode.SIZE_MISMATCH, f"(eq={eq_diam:.1f},d={d_est})"),
        )
    weighted_center = _refine_center_xy(
        gpatch,
        segmented.component_mask,
        seed_x=float(pcx),
        seed_y=float(pcy),
        med=med,
        hyp=hyp,
    )
    fcx, fcy = _component_geometry_center(cnt0, segmented.component_stats, weighted_center)
    drift = math.hypot(fcx - float(pcx), fcy - float(pcy))
    max_drift = float(config.max_center_drift_ratio) * re
    if enforce_rejections and drift > max_drift:
        return ViaDetection(
            float(offset[0] + fcx),
            float(offset[1] + fcy),
            _candidate_bbox(offset[0] + fcx, offset[1] + fcy, d_est),
            0.0,
            float(d_est),
            float(contrast),
            float(prom),
            0.0,
            0.0,
            hyp,
            _hard_reason(ViaRejectCode.CENTER_DRIFT, f"({drift:.1f}>{max_drift:.1f})"),
        )
    r_rect = cv2.minAreaRect(cnt0)
    w_r, h_r = float(r_rect[1][0]), float(r_rect[1][1])
    if w_r < 1e-3 or h_r < 1e-3:
        aspect = 1.0
    else:
        aspect = max(w_r, h_r) / (min(w_r, h_r) + 1e-6)
    r_expect = d_est * 0.5
    perim = float(cv2.arcLength(cnt0, True)) + 1e-3
    circ2 = (
        min(1.0, max(0.0, 4.0 * math.pi * max(area, 1.0) / (perim * perim)))
        if perim > 0
        else 0.0
    )
    fill = 4.0 * area / (w_r * h_r + 1e-6) if w_r * h_r > 1e-6 else 0.0
    compact2 = 0.5 * min(1.0, min(fill, 1.0)) + 0.5 * min(1.0, area / (math.pi * r_expect**2 + 1e-3))
    if enforce_rejections and compact2 < float(config.min_compactness):
        return ViaDetection(
            float(offset[0] + fcx),
            float(offset[1] + fcy),
            _candidate_bbox(offset[0] + fcx, offset[1] + fcy, d_est),
            0.0,
            float(d_est),
            float(contrast),
            float(prom),
            float(compact2),
            float(aspect),
            hyp,
            _hard_reason(ViaRejectCode.LOW_COMPACTNESS),
        )
    if enforce_rejections and circ2 < float(config.min_circularity):
        return ViaDetection(
            float(offset[0] + fcx),
            float(offset[1] + fcy),
            _candidate_bbox(offset[0] + fcx, offset[1] + fcy, d_est),
            0.0,
            float(d_est),
            float(contrast),
            float(prom),
            float(compact2),
            float(aspect),
            hyp,
            _hard_reason(ViaRejectCode.LOW_CIRCULARITY),
        )
    if enforce_rejections and aspect > float(config.max_elongation):
        return ViaDetection(
            float(offset[0] + fcx),
            float(offset[1] + fcy),
            _candidate_bbox(offset[0] + fcx, offset[1] + fcy, d_est),
            0.0,
            float(d_est),
            float(contrast),
            float(prom),
            float(compact2),
            float(aspect),
            hyp,
            _hard_reason(ViaRejectCode.ELONGATION),
        )

    structure_coherence, edge_snr, edge_sharpness = _local_structure_metrics(
        gradient_x,
        gradient_y,
        gradient_magnitude,
        fcx,
        fcy,
        float(d_est),
        float(contrast),
    )
    if enforce_rejections and structure_coherence > float(config.max_line_coherence):
        return ViaDetection(
            float(offset[0] + fcx),
            float(offset[1] + fcy),
            _candidate_bbox(offset[0] + fcx, offset[1] + fcy, d_est),
            0.0,
            float(d_est),
            float(contrast),
            float(prom),
            float(compact2),
            float(aspect),
            hyp,
            _hard_reason(ViaRejectCode.LINE_COHERENCE, f"({structure_coherence:.2f})"),
        )
    if enforce_rejections and edge_sharpness < float(config.min_edge_sharpness):
        return ViaDetection(
            float(offset[0] + fcx),
            float(offset[1] + fcy),
            _candidate_bbox(offset[0] + fcx, offset[1] + fcy, d_est),
            0.0,
            float(d_est),
            float(contrast),
            float(prom),
            float(compact2),
            float(aspect),
            hyp,
            _hard_reason(ViaRejectCode.DIFFUSE_SPOT, f"({edge_sharpness:.2f})"),
        )

    icx, icy = round(fcx), round(fcy)
    icx = max(0, min(pw - 1, icx))
    icy = max(0, min(ph - 1, icy))
    r_edge = int(max(1, d_est // 3))
    ys = int(max(0, icy - r_edge))
    ye = int(min(ph - 1, icy + r_edge))
    xs = int(max(0, icx - r_edge))
    xe = int(min(pw - 1, icx + r_edge))
    left = float(np.mean(gpatch[icy, xs:icx])) if icx > xs else gpatch[icy, icx]
    right = float(np.mean(gpatch[icy, icx : xe + 1])) if icx < xe else gpatch[icy, icx]
    up = float(np.mean(gpatch[ys:icy, icx])) if icy > ys else gpatch[icy, icx]
    down = float(np.mean(gpatch[icy:ye, icx])) if icy < ye else gpatch[icy, icx]
    border_n = (abs(left - right) + abs(up - down)) / 255.0
    el = max(0.0, aspect - 1.0)
    aspect_line_n = min(1.0, el / (float(config.max_elongation) + 0.1))
    coherence_line_n = _scale01(structure_coherence, 0.30, 0.80)
    line_n = max(aspect_line_n, coherence_line_n)

    features = CandidateFeatures(
        center_x=fcx,
        center_y=fcy,
        diameter=float(d_est),
        contrast=float(contrast),
        prominence=float(prom),
        compactness=float(compact2),
        circularity=float(circ2),
        aspect=float(aspect),
        line_coherence=float(structure_coherence),
        edge_snr=float(edge_snr),
        edge_sharpness=float(edge_sharpness),
        border_imbalance=float(border_n),
        center_brightness=float(gpatch[icy, icx]),
        equivalent_diameter=float(eq_diam),
        center_drift=float(drift),
        binarization_threshold=float(segmented.binarization_threshold),
    )
    score_metrics = _candidate_score_metrics(features, config, allowed_diameters, line_n)
    final = score_metrics["final_score"]

    gx = float(offset[0]) + fcx
    gy = float(offset[1]) + fcy
    half = float(d_est) * 0.5
    ox = round(gx - half)
    oy = round(gy - half)
    bbox = (ox, oy, int(d_est), int(d_est))
    return ViaDetection(
        x=gx,
        y=gy,
        bbox=bbox,
        score=float(final),
        diameter_estimate=float(d_est),
        contrast=float(contrast),
        prominence=float(prom),
        compactness=float(compact2),
        aspect=float(aspect),
        polarity_hypothesis=hyp,
        reject_reason=None,
        features={
            "center_brightness": features.center_brightness,
            "contrast": features.contrast,
            "prominence": features.prominence,
            "diameter": features.diameter,
            "equivalent_diameter": features.equivalent_diameter,
            "center_drift": features.center_drift,
            "compactness": features.compactness,
            "circularity": features.circularity,
            "aspect": features.aspect,
            "line_coherence": features.line_coherence,
            "edge_snr": features.edge_snr,
            "edge_sharpness": features.edge_sharpness,
            "border_imbalance": features.border_imbalance,
            "line_likeness": float(line_n),
            "binarization_threshold": features.binarization_threshold,
            **score_metrics,
        },
    )
