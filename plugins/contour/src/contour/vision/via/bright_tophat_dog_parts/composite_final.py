"""bright_tophat_dog: composite final."""
from __future__ import annotations

from contour.vision.via.bright_tophat_dog_parts.center_distance_index import _clip01

def _composite_final_0_100(
    *,
    center_mean: float,
    local_contrast: float,
    local_peak: float,
    area: float,
    min_area: float,
    max_area: float,
    circularity: float,
    min_circularity: float,
    radial_asymmetry: float,
    edge_likeness: float,
    line_likeness: float,
    max_asym: float,
    max_edge: float,
    max_line: float,
    metal_fraction: float,
    metal_mode: str,
    distance_to_edge: float,
    annular_contrast: float = 0.0,
    isolation_score: float = 1.0,
) -> float:
    bright_abs_norm = _clip01(center_mean / 255.0)
    bright_contrast_norm = _clip01(local_contrast / 40.0)
    lp_norm = _clip01(local_peak / 255.0)
    mid = 0.5 * (min_area + max_area)
    span = max(max_area - min_area, 1.0)
    area_norm = 1.0 - float(min(1.0, abs(area - mid) / (span * 0.5 + 1e-6)))
    area_norm = _clip01(area_norm)
    circ_ref = max(min_circularity, 0.01)
    circ_norm = _clip01(circularity / circ_ref)
    m_as = max(float(max_asym), 1e-3)
    sym_norm = _clip01(1.0 - radial_asymmetry / m_as)
    annular_norm = _clip01(float(annular_contrast) / 40.0)
    isolation_norm = _clip01(float(isolation_score))
    mbonus = 0.0
    if metal_mode == "soft":
        mbonus = 5.0 * _clip01(metal_fraction)
    elif metal_mode == "strict":
        mbonus = 3.0 * _clip01(metal_fraction)
    distb = min(5.0, max(0.0, float(distance_to_edge) * 0.25))
    edge_penalty = 30.0 * max(0.0, float(edge_likeness) / max(float(max_edge), 1e-3) - 1.0)
    line_penalty = 5.0 * max(0.0, float(line_likeness) / max(float(max_line), 1e-3) - 1.0)
    proximity_penalty = 5.0 * max(0.0, 1.0 - float(distance_to_edge))
    core = (
        10.0 * bright_abs_norm
        + 8.0 * bright_contrast_norm
        + 15.0 * lp_norm
        + 5.0 * area_norm
        + 12.0 * circ_norm
        + 15.0 * sym_norm
        + 20.0 * annular_norm
        + 25.0 * isolation_norm
    )
    total = core + mbonus + distb - edge_penalty - line_penalty - proximity_penalty
    if total <= 0.0:
        return 0.0
    if total >= 100.0:
        return 100.0
    return float(total)
