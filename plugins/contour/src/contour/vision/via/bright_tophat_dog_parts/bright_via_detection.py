"""bright_tophat_dog: bright via detection."""
from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np
from contour.vision.io_normalize import to_gray_u8
from ...via_detection.binary_mask import looks_like_binary_mask

from contour.vision.via.bright_tophat_dog_parts.detect_binary_vias import _detect_binary_vias

from contour.vision.via.bright_tophat_dog_parts.prepare_bright_via_candidates import prepare_bright_via_candidates

from contour.vision.via.bright_tophat_dog_parts.score_bright_via_candidates import score_bright_via_candidates

@dataclass(frozen=True, slots=True)
class BrightViaDetection:
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: float
    circularity: float
    aspect: float
    brightness_score: float
    local_peak_score: float
    tophat_response: float
    dog_response: float
    metal_fraction: float
    final_score: float
    radial_asymmetry: float = 0.0
    edge_likeness: float = 0.0
    line_likeness: float = 0.0
    distance_to_edge: float = 0.0
    annular_contrast: float = 0.0
    isolation_score: float = 1.0
    status: str = "accepted"
    hard_reason: str = ""

@dataclass(slots=True)
class BrightViaDetectionResult:
    """detections: accepted only (exported / mask). candidates: all after NMS + scoring."""

    detections: list[BrightViaDetection] = field(default_factory=list)
    candidates: list[BrightViaDetection] = field(default_factory=list)
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)

@dataclass(slots=True)
class BrightViaPreparedCandidates:
    gray: np.ndarray
    processed: np.ndarray
    tophat: np.ndarray
    dog: np.ndarray
    tophat_mask: np.ndarray
    dog_mask: np.ndarray
    via_mask: np.ndarray
    candidate_mask: np.ndarray
    metal_mask: np.ndarray
    edge_distance: np.ndarray
    raw_candidates: list[_RawStage1]
    min_area: float
    max_area: float
    nominal_diameter: float
    nominal_radius: float
    metal_mode: str
    isolation: tuple[np.ndarray, np.ndarray] | None = None

@dataclass(slots=True)
class _CandidateDebugMaps:
    radial_symmetry: np.ndarray
    edge_likeness: np.ndarray
    line_likeness: np.ndarray

    @classmethod
    def empty(cls, shape: tuple[int, ...]) -> _CandidateDebugMaps:
        return cls(
            radial_symmetry=np.zeros(shape, dtype=np.uint8),
            edge_likeness=np.zeros(shape, dtype=np.uint8),
            line_likeness=np.zeros(shape, dtype=np.uint8),
        )

@dataclass(slots=True)
class _RawStage1:
    center: tuple[float, float]
    contour: np.ndarray | None
    bbox: tuple[int, int, int, int]
    area: float
    circularity: float
    aspect: float
    prelim: float
    source: str

def detect_bright_vias(image: np.ndarray, config: BrightViaDetectorConfig) -> BrightViaDetectionResult:
    cfg = config.validated().with_diameter_scaled_kernels()
    gray = to_gray_u8(image)
    if looks_like_binary_mask(gray):
        return _detect_binary_vias(gray, cfg)
    prepared = prepare_bright_via_candidates(image, cfg)
    return score_bright_via_candidates(prepared, cfg)

if TYPE_CHECKING:
    from contour.vision.via.bright_tophat_dog_parts.bright_via_detector_config import BrightViaDetectorConfig
