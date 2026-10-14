"""_core: refine center."""
from __future__ import annotations
import numpy as np

def _refine_center(gray: np.ndarray, cx: int, cy: int, radius: int, *, bright: bool) -> tuple[float, float]:
    """Return an intensity-weighted centroid in the neighbourhood of ``(cx, cy)``.

    The blob/ring response peaks do not always coincide with the geometric
    centre of a via - they can be pulled towards a neighbouring trace or the
    brighter half of an asymmetric contact. We therefore re-centre each
    candidate on the centroid of the brightest (or darkest for dark vias)
    pixels inside a local window. Movement is clamped so that noisy patches
    cannot teleport the candidate across the image.
    """

    r = max(2, int(radius))
    half = max(r, round(r * 1.1))
    height, width = gray.shape[:2]
    x0 = max(0, int(cx) - half)
    x1 = min(width, int(cx) + half + 1)
    y0 = max(0, int(cy) - half)
    y1 = min(height, int(cy) + half + 1)
    if x1 <= x0 or y1 <= y0:
        return float(cx), float(cy)
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    lo = float(patch.min())
    hi = float(patch.max())
    span = hi - lo
    if span < 5.0:
        return float(cx), float(cy)
    threshold = lo + span * 0.5
    weights = patch - threshold if bright else threshold - patch
    np.maximum(weights, 0.0, out=weights)
    total = float(weights.sum())
    if total <= 1e-6:
        return float(cx), float(cy)
    ys = np.arange(patch.shape[0], dtype=np.float32).reshape(-1, 1)
    xs = np.arange(patch.shape[1], dtype=np.float32).reshape(1, -1)
    mean_x = float((weights * xs).sum() / total)
    mean_y = float((weights * ys).sum() / total)
    new_cx = float(x0) + mean_x
    new_cy = float(y0) + mean_y
    dx = new_cx - float(cx)
    dy = new_cy - float(cy)
    dist = float(np.hypot(dx, dy))
    max_move = float(r) * 0.9
    if dist > max_move and dist > 1e-6:
        scale = max_move / dist
        new_cx = float(cx) + dx * scale
        new_cy = float(cy) + dy * scale
    return new_cx, new_cy
