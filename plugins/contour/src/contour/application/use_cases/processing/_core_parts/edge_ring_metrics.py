"""_core: edge ring metrics."""
from __future__ import annotations
import numpy as np
from contour.utils import ensure_uint8
from contour.application.processing import (
    ContourExtractionSettings,
)

from contour.application.use_cases.processing._core_parts.refine_conductor_mask_by import _via_grayscale

def _isolation_score(extent: tuple[int, int, int] | None, radius: int) -> float:
    """Map the connected-component extent around the candidate to ``[0, 1]``.

    A via sits on an isolated bright (or dark) island. If the island that
    contains the peak is much larger than the expected via, the peak is most
    likely a trace extension or the corner of a trace and should be penalised.
    """

    if extent is None:
        return 1.0
    _area, comp_width, comp_height = extent
    comp_max = float(max(comp_width, comp_height))
    if comp_max <= 0.0:
        return 1.0
    # Give a full score while the island fits within ~2.5 via diameters; drop
    # to zero once the island grows past ~5 diameters (clearly a trace).
    diameter = float(radius * 2 + 1)
    lo = diameter * 2.5
    hi = diameter * 5.0
    if comp_max <= lo:
        return 1.0
    if comp_max >= hi:
        return 0.0
    return float(1.0 - (comp_max - lo) / max(1e-6, hi - lo))

def _radial_contrast(gray: np.ndarray, cx: int, cy: int, radius: int, *, bright: bool) -> float:
    radius = max(2, int(radius))
    padding = max(2, round(radius * 0.7))
    left = max(0, int(cx) - radius - padding)
    top = max(0, int(cy) - radius - padding)
    right = min(gray.shape[1], int(cx) + radius + padding + 1)
    bottom = min(gray.shape[0], int(cy) + radius + padding + 1)
    if right <= left or bottom <= top:
        return 0.0
    patch = ensure_uint8(gray)[top:bottom, left:right].astype(np.float32)
    yy, xx = np.ogrid[top - cy : bottom - cy, left - cx : right - cx]
    distance_sq = xx * xx + yy * yy
    inner_mask = distance_sq <= max(1.0, radius * 0.55) ** 2
    outer_lo = (radius * 1.15) ** 2
    outer_hi = (radius * 1.75) ** 2
    outer_mask = (distance_sq >= outer_lo) & (distance_sq <= outer_hi)
    if not np.any(inner_mask) or not np.any(outer_mask):
        return 0.0
    inner_mean = float(np.mean(patch[inner_mask]))
    outer_mean = float(np.mean(patch[outer_mask]))
    return inner_mean - outer_mean if bright else outer_mean - inner_mean

def _edge_ring_metrics(gradient: np.ndarray, cx: int, cy: int, radius: int) -> tuple[float, float]:
    """Return (mean ring gradient, angular coverage in [0..1])."""

    radius = max(2, int(radius))
    padding = max(2, round(radius * 0.35))
    left = max(0, int(cx) - radius - padding)
    top = max(0, int(cy) - radius - padding)
    right = min(gradient.shape[1], int(cx) + radius + padding + 1)
    bottom = min(gradient.shape[0], int(cy) + radius + padding + 1)
    if right <= left or bottom <= top or gradient.size == 0:
        return 0.0, 0.0
    patch = gradient[top:bottom, left:right]
    if patch.size == 0:
        return 0.0, 0.0
    yy, xx = np.ogrid[top - cy : bottom - cy, left - cx : right - cx]
    grid_x = np.broadcast_to(xx.astype(np.float32), patch.shape)
    grid_y = np.broadcast_to(yy.astype(np.float32), patch.shape)
    distance = np.sqrt(grid_x * grid_x + grid_y * grid_y)
    inner_radius = max(1.0, float(radius) - max(1.0, radius * 0.30))
    outer_radius = float(radius) + max(1.0, radius * 0.30)
    ring_mask = (distance >= inner_radius) & (distance <= outer_radius)
    if not np.any(ring_mask):
        return 0.0, 0.0
    ring_values = patch[ring_mask].astype(np.float32)
    mean_strength = float(ring_values.mean()) if ring_values.size else 0.0
    threshold = max(20.0, mean_strength * 1.10)
    strong_mask = ring_mask & (patch.astype(np.float32) >= threshold)
    coverage = _angular_coverage(strong_mask, grid_x, grid_y, sectors=16)
    return mean_strength, coverage

def _angular_coverage(mask: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, *, sectors: int) -> float:
    if sectors <= 0 or not np.any(mask):
        return 0.0
    angles = np.arctan2(grid_y[mask], grid_x[mask])
    bins = np.floor(((angles + np.pi) / (2.0 * np.pi)) * sectors).astype(np.int32)
    bins = np.clip(bins, 0, sectors - 1)
    return float(np.unique(bins).size) / float(sectors)

def _via_saved_template_arrays(settings: ContourExtractionSettings) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for payload in settings.via_template_images:
        try:
            template = ensure_uint8(np.asarray(payload))
        except Exception:
            continue
        if template.ndim == 3:
            template = _via_grayscale(template)
        if template.ndim != 2 or template.shape[0] < 2 or template.shape[1] < 2:
            continue
        arrays.append(template)
    return arrays

def _normalize_template_for_matching(image: np.ndarray) -> np.ndarray:
    data = ensure_uint8(image)
    if data.size == 0:
        return data
    minimum = int(data.min())
    maximum = int(data.max())
    if maximum <= minimum:
        return data
    return np.clip((data.astype(np.float32) - minimum) * (255.0 / float(maximum - minimum)), 0, 255).astype(np.uint8)
