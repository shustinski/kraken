"""heuristic_detector: candidate features."""
from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
import numpy as np
from contour.vision.via_detection.result import ViaDetection

class ViaRejectCode(StrEnum):
    BRIGHTNESS_RANGE = "brightness_range"
    LOW_CENTER_BRIGHTNESS = "low_center_brightness"
    LOW_CONTRAST = "low_contrast"
    LOW_PROMINENCE = "low_prominence"
    NO_COMPONENT = "no_component"
    SIZE_MISMATCH = "size_mismatch"
    CENTER_DRIFT = "center_drift"
    LOW_COMPACTNESS = "compact"
    LOW_CIRCULARITY = "circularity"
    ELONGATION = "elongation"
    LINE_COHERENCE = "line_coherence"
    DIFFUSE_SPOT = "diffuse_spot"

@dataclass(frozen=True)
class PreparedHeuristicFrame:
    gray: np.ndarray
    gray_f32: np.ndarray
    corrected_u8: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    gradient_magnitude: np.ndarray

@dataclass(frozen=True)
class CandidateFeatures:
    center_x: float
    center_y: float
    diameter: float
    contrast: float
    prominence: float
    compactness: float
    circularity: float
    aspect: float
    line_coherence: float
    edge_snr: float
    edge_sharpness: float
    border_imbalance: float
    center_brightness: float = 0.0
    equivalent_diameter: float = 0.0
    center_drift: float = 0.0
    binarization_threshold: float = 0.0

@dataclass(frozen=True)
class SegmentedCandidate:
    gray: np.ndarray
    median: float
    contrast: float
    prominence: float
    area: float
    component_mask: np.ndarray
    contour: np.ndarray
    component_stats: np.ndarray
    binarization_threshold: float

def _hard_reason(code: ViaRejectCode, detail: str = "") -> str:
    return f"hard:{code.value}{detail}"

def _det_better(candidate: ViaDetection, current: ViaDetection | None) -> bool:
    if current is None:
        return True
    c_hard = bool(current.reject_reason and str(current.reject_reason).startswith("hard:"))
    n_hard = bool(candidate.reject_reason and str(candidate.reject_reason).startswith("hard:"))
    if c_hard and not n_hard:
        return True
    if n_hard and not c_hard:
        return False
    return float(candidate.score) > float(current.score)
