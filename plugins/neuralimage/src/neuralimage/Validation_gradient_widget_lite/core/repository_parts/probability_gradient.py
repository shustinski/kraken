"""repository: probability gradient."""
from __future__ import annotations
import numpy as np

def _probability_gradient(probability: np.ndarray) -> np.ndarray:
    """Compute simple central-difference gradient magnitude for a probability map."""

    prob = np.asarray(probability, dtype=np.float32)
    if prob.ndim != 2 or prob.size == 0:
        return np.zeros_like(prob, dtype=np.float32)
    padded = np.pad(prob, 1, mode='edge')
    grad_x = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
    grad_y = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
    return np.hypot(grad_x, grad_y).astype(np.float32)
