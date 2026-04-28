"""Configuration dataclasses for template and heuristic via detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


def parse_diameter_list(text: str) -> list[int]:
    """Parse e.g. '6,8,10', '6-8', '6-8, 12' into a sorted unique list of positive ints."""

    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            try:
                lo, hi = int(a.strip()), int(b.strip())
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            try:
                out.append(int(p))
            except ValueError:
                continue
    return sorted(set(n for n in out if n > 0))


class ViaPolarity(StrEnum):
    BRIGHT = "bright"
    DARK = "dark"
    RING_LIGHT_RING = "ring_light_ring"  # bright ring, dark center
    RING_DARK_RING = "ring_dark_ring"  # dark ring, bright center
    AUTO = "auto"


@dataclass
class HeuristicViaDetectorConfig:
    diameter_mode: str = "range"  # "range" | "fixed"
    diameter_min: int = 6
    diameter_max: int = 12
    fixed_diameters: list[int] = field(default_factory=lambda: [6, 8, 10])
    polarity: str = ViaPolarity.AUTO
    """Sensitivity: higher recall vs precision via threshold mapping."""
    sensitivity: str = "medium"
    nms_distance: int = 5
    min_final_score: float = 40.0
    min_distance_between_peaks: int = 0  # 0 = derive from min diameter
    min_peak_grey: float = 0.0  # absolute floor on response map for a seed
    background_sigma: float = 25.0
    analysis_window_scale: float = 3.0
    min_analyze_size: int = 24
    use_bilateral: bool = False
    bilateral_d: int = 5
    bilateral_sigma_color: float = 32.0
    bilateral_sigma_space: float = 32.0
    # Hard reject / score gates
    min_center_contrast: float = 6.0
    min_peak_prominence: float = 4.0
    min_compactness: float = 0.12
    max_elongation: float = 3.2
    line_penalty_scale: float = 1.0
    border_penalty_scale: float = 1.0
    local_binarize_percentile: float = 88.0
    # Weights for final 0..100
    w_contrast: float = 25.0
    w_prominence: float = 20.0
    w_size: float = 20.0
    w_compact: float = 15.0
    w_round: float = 10.0
    w_balance: float = 10.0
    w_line: float = 20.0
    w_border: float = 20.0
    # |D_eq - d_est| / d_est; D_eq = 2*sqrt(area/pi). Stricter when diameter_mode == "fixed".
    size_tolerance_ratio: float = 0.30
    size_tolerance_ratio_fixed: float = 0.18
    # Reject if |centroid - seed| exceeds this fraction of d_est (wrong CC)
    max_center_drift_ratio: float = 0.55

    def effective_size_tolerance(self) -> float:
        return float(
            self.size_tolerance_ratio_fixed if self.diameter_mode == "fixed" else self.size_tolerance_ratio
        )

    def allowed_diameters(self) -> list[int]:
        if self.diameter_mode == "fixed" and self.fixed_diameters:
            return sorted({int(d) for d in self.fixed_diameters if d > 0})
        d0 = max(1, int(self.diameter_min))
        d1 = max(d0, int(self.diameter_max))
        return list(range(d0, d1 + 1))

    def snapshot(self) -> dict[str, Any]:
        return {
            "diameter_mode": self.diameter_mode,
            "diameter_min": self.diameter_min,
            "diameter_max": self.diameter_max,
            "fixed_diameters": list(self.fixed_diameters),
            "polarity": self.polarity,
            "sensitivity": self.sensitivity,
            "nms_distance": self.nms_distance,
            "min_final_score": self.min_final_score,
            "background_sigma": self.background_sigma,
            "min_center_contrast": self.min_center_contrast,
            "min_peak_prominence": self.min_peak_prominence,
            "min_compactness": self.min_compactness,
            "max_elongation": self.max_elongation,
            "local_binarize_percentile": self.local_binarize_percentile,
        }


@dataclass
class TemplateViaDetectorConfig:
    templates: list[Any]  # list of HxW uint8 grayscale
    min_correlation: float = 0.35
    nms_distance: int = 4
    scale_min: float = 1.0
    scale_max: float = 1.0
    scale_step: float = 0.1
    use_ccoeff_normed: bool = True

    def snapshot(self) -> dict[str, Any]:
        n = 0
        t0 = self.templates[0] if self.templates else None
        if t0 is not None and hasattr(t0, "shape"):
            n = len(self.templates)
        return {
            "num_templates": n,
            "min_correlation": self.min_correlation,
            "nms_distance": self.nms_distance,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "scale_step": self.scale_step,
        }
