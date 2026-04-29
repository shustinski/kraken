"""Structured detection result types (no ML, JSON-friendly snapshots)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ViaDetection:
    x: float
    y: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    score: float
    diameter_estimate: float
    contrast: float
    prominence: float
    compactness: float
    aspect: float
    polarity_hypothesis: str = ""
    reject_reason: str | None = None


@dataclass(slots=True)
class DetectionResult:
    method: str
    accepted: list[ViaDetection]
    rejected: list[ViaDetection] = field(default_factory=list)
    below_threshold: list[ViaDetection] = field(default_factory=list)
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)
    parameters_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_json_safe_snapshot(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "accepted": [self._det_dict(d) for d in self.accepted],
            "rejected": [self._det_dict(d) for d in self.rejected],
            "below_threshold": [self._det_dict(d) for d in self.below_threshold],
            "parameters": dict(self.parameters_snapshot),
        }

    @staticmethod
    def _det_dict(d: ViaDetection) -> dict[str, Any]:
        return {
            "x": d.x,
            "y": d.y,
            "bbox": list(d.bbox),
            "score": d.score,
            "diameter_estimate": d.diameter_estimate,
            "contrast": d.contrast,
            "prominence": d.prominence,
            "compactness": d.compactness,
            "aspect": d.aspect,
            "polarity_hypothesis": d.polarity_hypothesis,
            "reject_reason": d.reject_reason,
        }
