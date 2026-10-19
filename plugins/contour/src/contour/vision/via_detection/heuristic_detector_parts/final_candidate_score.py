"""heuristic_detector: final candidate score."""
from __future__ import annotations
from typing import TYPE_CHECKING
from contour.vision.via_detection.config import HeuristicViaDetectorConfig

def _final_candidate_score(
    features: CandidateFeatures,
    config: HeuristicViaDetectorConfig,
    allowed_diameters: list[int],
    line_likeness: float,
) -> float:
    """Combine normalized features; all calibration constants live in config."""

    from contour.vision.via_detection.heuristic_detector_parts.candidate_score_metrics import _candidate_score_metrics
    return _candidate_score_metrics(features, config, allowed_diameters, line_likeness)["final_score"]

def _scale01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

if TYPE_CHECKING:
    from contour.vision.via_detection.heuristic_detector_parts.candidate_features import CandidateFeatures
