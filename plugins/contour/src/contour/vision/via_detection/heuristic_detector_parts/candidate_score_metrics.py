"""heuristic_detector: candidate score metrics."""
from __future__ import annotations
from contour.vision.via_detection.config import HeuristicViaDetectorConfig

from contour.vision.via_detection.heuristic_detector_parts.final_candidate_score import _scale01

def _candidate_score_metrics(
    features: CandidateFeatures,
    config: HeuristicViaDetectorConfig,
    allowed_diameters: list[int],
    line_likeness: float,
) -> dict[str, float]:
    """Return the normalized features and weighted contributions used by the score."""

    sc_contrast = _scale01(features.contrast, config.contrast_score_min, config.contrast_score_max)
    edge_span = 1.0 - float(config.edge_quality_floor)
    edge_quality = float(config.edge_quality_floor) + edge_span * _scale01(
        features.edge_snr,
        config.edge_snr_score_min,
        config.edge_snr_score_max,
    )
    sc_prominence = _scale01(
        features.prominence,
        config.prominence_score_min,
        config.prominence_score_max,
    ) * edge_quality
    diameter_min = float(min(allowed_diameters))
    diameter_max = float(max(allowed_diameters))
    sc_size = 1.0 - min(
        1.0,
        abs(features.diameter - 0.5 * (diameter_min + diameter_max))
        / (diameter_max - diameter_min + 1.0)
        * 0.4,
    )
    sc_compact = min(1.0, max(0.0, features.compactness))
    sc_round = min(1.0, max(0.0, min(features.circularity, 1.0)))
    sc_balance = 1.0 - min(1.0, features.border_imbalance * float(config.border_balance_scale))
    contribution_contrast = float(config.w_contrast) * sc_contrast
    contribution_prominence = float(config.w_prominence) * sc_prominence
    contribution_size = float(config.w_size) * sc_size
    contribution_compact = float(config.w_compact) * sc_compact
    contribution_round = float(config.w_round) * sc_round
    contribution_balance = float(config.w_balance) * sc_balance
    penalty_line = float(config.w_line) * line_likeness * float(config.line_penalty_scale)
    penalty_border = (
        float(config.w_border) * features.border_imbalance * float(config.border_penalty_scale)
    )
    raw = (
        contribution_contrast
        + contribution_prominence
        + contribution_size
        + contribution_compact
        + contribution_round
        + contribution_balance
        - penalty_line
        - penalty_border
    )
    return {
        "normalized_contrast": float(sc_contrast),
        "normalized_prominence": float(sc_prominence),
        "normalized_size": float(sc_size),
        "normalized_compactness": float(sc_compact),
        "normalized_roundness": float(sc_round),
        "normalized_balance": float(sc_balance),
        "edge_quality": float(edge_quality),
        "contribution_contrast": float(contribution_contrast),
        "contribution_prominence": float(contribution_prominence),
        "contribution_size": float(contribution_size),
        "contribution_compactness": float(contribution_compact),
        "contribution_roundness": float(contribution_round),
        "contribution_balance": float(contribution_balance),
        "penalty_line": float(penalty_line),
        "penalty_border": float(penalty_border),
        "raw_score": float(raw),
        "final_score": max(0.0, min(100.0, float(raw))),
    }
