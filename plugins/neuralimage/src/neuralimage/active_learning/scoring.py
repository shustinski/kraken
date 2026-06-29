from __future__ import annotations

import numpy as np


def _binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    return -(values * np.log(values) + (1.0 - values) * np.log(1.0 - values))


def score_prediction_uncertainty(
    probabilities: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    ensemble_variance: np.ndarray | None = None,
    low_confidence_threshold: float = 0.35,
    high_entropy_threshold: float = 0.65,
    instability_threshold: float = 0.15,
) -> dict[str, np.ndarray | float]:
    probs = np.asarray(probabilities, dtype=np.float32)
    entropy = _binary_entropy(probs)
    mean_entropy = float(entropy.mean())
    if confidence is None:
        confidence_map = 1.0 - entropy / np.log(2.0)
    else:
        confidence_map = np.asarray(confidence, dtype=np.float32)
    low_confidence = confidence_map < float(low_confidence_threshold)
    high_entropy = entropy > float(high_entropy_threshold)
    unstable = np.zeros_like(probs, dtype=bool)
    if ensemble_variance is not None:
        unstable = np.asarray(ensemble_variance, dtype=np.float32) > float(instability_threshold)
    score = (
        low_confidence.astype(np.float32) * 0.4
        + high_entropy.astype(np.float32) * 0.35
        + unstable.astype(np.float32) * 0.25
    )
    return {
        'score': score,
        'entropy': entropy,
        'confidence': confidence_map,
        'low_confidence': low_confidence,
        'high_entropy': high_entropy,
        'unstable': unstable,
        'mean_entropy': mean_entropy,
        'mean_confidence': float(confidence_map.mean()),
    }
