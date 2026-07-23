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
    normalized = text.replace(";", ",").replace("–", "-")
    parts: list[str] = []
    for chunk in normalized.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            parts.extend(token.strip() for token in chunk.split() if token.strip())
        else:
            parts.append(chunk)
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
    profile_version: int = 2
    diameter_mode: str = "range"  # "range" | "fixed"
    diameter_min: int = 6
    diameter_max: int = 12
    fixed_diameters: list[int] = field(default_factory=lambda: [6, 8, 10])
    polarity: str = ViaPolarity.AUTO
    nms_distance: int = 5
    min_final_score: float = 38.0
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
    min_center_brightness: float = 0.0
    min_center_contrast: float = 50.0
    min_peak_prominence: float = 50.0
    min_compactness: float = 0.9
    min_circularity: float = 0.40
    max_elongation: float = 2.5
    line_penalty_scale: float = 3.0
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
    # Hard structure gates.
    max_line_coherence: float = 0.82
    min_edge_sharpness: float = 0.20
    # Feature normalization used by the final score.
    contrast_score_min: float = 3.0
    contrast_score_max: float = 20.0
    prominence_score_min: float = 2.0
    prominence_score_max: float = 25.0
    edge_snr_score_min: float = 0.70
    edge_snr_score_max: float = 2.80
    edge_quality_floor: float = 0.55
    border_balance_scale: float = 2.0
    # Candidate-generation response percentile.
    seed_percentile: float = 90.0
    use_intensity_range_seeds: bool = False
    # |D_eq - d_est| / d_est; D_eq = 2*sqrt(area/pi). Stricter when diameter_mode == "fixed".
    size_tolerance_ratio: float = 0.36
    size_tolerance_ratio_fixed: float = 0.26
    # Reject if |centroid - seed| exceeds this fraction of d_est (wrong CC)
    max_center_drift_ratio: float = 0.72

    # Brightness range gates (0-255 on preprocessed gray).
    bright_range_enabled: bool = True
    bright_range_min: float = 140.0
    bright_range_max: float = 255.0
    dark_range_enabled: bool = False
    dark_range_min: float = 0.0
    dark_range_max: float = 60.0

    def validate(self) -> None:
        """Reject inconsistent expert settings before expensive image work starts."""

        pairs = (
            ("contrast score", self.contrast_score_min, self.contrast_score_max),
            ("prominence score", self.prominence_score_min, self.prominence_score_max),
            ("edge SNR score", self.edge_snr_score_min, self.edge_snr_score_max),
            ("bright range", self.bright_range_min, self.bright_range_max),
            ("dark range", self.dark_range_min, self.dark_range_max),
        )
        for name, lower, upper in pairs:
            if float(lower) > float(upper):
                raise ValueError(f"{name}: minimum must not exceed maximum")
        if not 0.0 <= float(self.max_line_coherence) <= 1.0:
            raise ValueError("max_line_coherence must be in range 0..1")
        if not 0.0 <= float(self.edge_quality_floor) <= 1.0:
            raise ValueError("edge_quality_floor must be in range 0..1")
        if not 0.0 <= float(self.min_circularity) <= 1.0:
            raise ValueError("min_circularity must be in range 0..1")
        if not 0.0 <= float(self.seed_percentile) <= 100.0:
            raise ValueError("seed_percentile must be in range 0..100")
        if not 0.0 <= float(self.min_center_brightness) <= 255.0:
            raise ValueError("min_center_brightness must be in range 0..255")

    def effective_size_tolerance(self) -> float:
        return float(self.size_tolerance_ratio_fixed if self.diameter_mode == "fixed" else self.size_tolerance_ratio)

    def allowed_diameters(self) -> list[int]:
        if self.diameter_mode == "fixed" and self.fixed_diameters:
            return sorted({int(d) for d in self.fixed_diameters if d > 0})
        d0 = max(1, int(self.diameter_min))
        d1 = max(d0, int(self.diameter_max))
        return list(range(d0, d1 + 1))

    def snapshot(self) -> dict[str, Any]:
        values = dict(vars(self))
        values["fixed_diameters"] = list(self.fixed_diameters)
        values["polarity"] = str(self.polarity)
        return values


@dataclass
class TemplateViaDetectorConfig:
    templates: list[Any]  # list of HxW uint8 grayscale
    min_correlation: float = 0.35
    min_correlations: list[float] = field(default_factory=list)
    output_diameters: list[int] = field(default_factory=list)
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
            "min_correlations": list(self.min_correlations),
            "output_diameters": list(self.output_diameters),
            "nms_distance": self.nms_distance,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "scale_step": self.scale_step,
        }
