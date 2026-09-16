from __future__ import annotations

import numpy as np


def render_critical_regions_overlay(
    pred_probs: np.ndarray,
    critical_mask: np.ndarray,
    *,
    alpha: float = 0.55,
) -> np.ndarray:
    """Overlay critical Topograph regions in red on a grayscale prediction preview."""
    pred = np.asarray(pred_probs, dtype=np.float32)
    mask = np.asarray(critical_mask, dtype=bool)
    if pred.ndim == 3:
        pred = pred[0]
    if mask.ndim == 3:
        mask = mask[0]

    pred = np.clip(pred, 0.0, 1.0)
    base = np.stack([pred, pred, pred], axis=-1)
    if not mask.any():
        return np.clip(base, 0.0, 1.0)

    overlay = base.copy()
    overlay[mask, 0] = np.clip(overlay[mask, 0] * (1.0 - alpha) + alpha, 0.0, 1.0)
    overlay[mask, 1] = overlay[mask, 1] * (1.0 - alpha)
    overlay[mask, 2] = overlay[mask, 2] * (1.0 - alpha)
    return np.clip(overlay, 0.0, 1.0)
