"""bright_tophat_dog: score bright via candidates."""
from __future__ import annotations
from typing import TYPE_CHECKING
import cv2
import numpy as np
from ....application.preview_cancellation import raise_if_preview_cancelled

from contour.vision.via.bright_tophat_dog_parts.module_state import _RADIAL_SMOOTH_SIGMA

from contour.vision.via.bright_tophat_dog_parts.draw_overlay import (
    _draw_overlay,
    _normalize_response,
)

from contour.vision.via.bright_tophat_dog_parts.score_one_candidate import _score_one_candidate

from contour.vision.via.bright_tophat_dog_parts.via_matched_filter_kernel import suppress_close_points

def score_bright_via_candidates(
    prepared: BrightViaPreparedCandidates, config: BrightViaDetectorConfig
) -> BrightViaDetectionResult:
    from contour.vision.via.bright_tophat_dog_parts.bright_via_detection import BrightViaDetectionResult, _CandidateDebugMaps
    cfg = config.validated()
    if prepared.gray.size == 0:
        return BrightViaDetectionResult()
    debug = _CandidateDebugMaps.empty(prepared.gray.shape[:2])
    gray_f32 = prepared.gray.astype(np.float32, copy=False)
    processed_f32 = prepared.processed.astype(np.float32, copy=False)
    # Lightly smoothed copy for the radial-symmetry gate only: CLAHE amplifies
    # pixel noise on flat regions, which would otherwise make a clean isolated via
    # look asymmetric. A small blur keeps the via structure (>=8 px) but removes
    # that noise so the symmetry score reflects real geometry.
    radial_f32 = cv2.GaussianBlur(processed_f32, (0, 0), _RADIAL_SMOOTH_SIGMA)
    tophat_f32 = prepared.tophat.astype(np.float32, copy=False)
    dog_f32 = prepared.dog.astype(np.float32, copy=False)
    raise_if_preview_cancelled()

    candidates_scored: list[BrightViaDetection] = []
    for i, raw in enumerate(prepared.raw_candidates):
        if i & 31 == 0:
            raise_if_preview_cancelled()
        det = _score_one_candidate(
            raw=raw,
            gray=prepared.gray,
            gray_f32=gray_f32,
            processed=prepared.processed,
            processed_f32=processed_f32,
            radial_f32=radial_f32,
            tophat=prepared.tophat,
            tophat_f32=tophat_f32,
            dog=prepared.dog,
            dog_f32=dog_f32,
            metal_mask=prepared.metal_mask,
            edge_distance=prepared.edge_distance,
            cfg=cfg,
            metal_mode=prepared.metal_mode,
            nominal_diameter=prepared.nominal_diameter,
            nominal_radius=prepared.nominal_radius,
            min_area=prepared.min_area,
            max_area=prepared.max_area,
            isolation=prepared.isolation,
            debug=debug,
        )
        if det is not None:
            candidates_scored.append(det)
    raise_if_preview_cancelled()
    candidates_scored = suppress_close_points(candidates_scored, cfg.nms_distance)
    accepted = [c for c in candidates_scored if c.status == "accepted"]
    rejected = [c for c in candidates_scored if c.status != "accepted"]
    compact_candidates = accepted + rejected
    show_rejected = bool(cfg.show_rejected_candidates)
    if prepared.gray.shape[0] * prepared.gray.shape[1] >= 1_800_000:
        show_rejected = False
    overlay = _draw_overlay(prepared.gray, compact_candidates, cfg, show_rejected=show_rejected)

    return BrightViaDetectionResult(
        detections=accepted,
        candidates=compact_candidates,
        debug_images={
            "raw_gray": prepared.gray,
            "processed": prepared.processed,
            "tophat": prepared.tophat,
            "dog": prepared.dog,
            "tophat_mask": prepared.tophat_mask,
            "dog_mask": prepared.dog_mask,
            "via_mask": prepared.via_mask,
            "candidate_mask": prepared.candidate_mask,
            "metal_mask": prepared.metal_mask,
            "radial_symmetry": debug.radial_symmetry,
            "edge_likeness": debug.edge_likeness,
            "line_likeness": debug.line_likeness,
            "distance_to_edge": _normalize_response(prepared.edge_distance),
            "final_overlay": overlay,
        },
    )

if TYPE_CHECKING:
    from contour.vision.via.bright_tophat_dog_parts.bright_via_detection import BrightViaDetection, BrightViaDetectionResult, BrightViaPreparedCandidates
    from contour.vision.via.bright_tophat_dog_parts.bright_via_detector_config import BrightViaDetectorConfig
