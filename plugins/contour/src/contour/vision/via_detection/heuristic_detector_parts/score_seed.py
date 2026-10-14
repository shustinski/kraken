"""heuristic_detector: score seed."""
from __future__ import annotations
import math
import cv2
import numpy as np
from contour.vision.via_detection.config import HeuristicViaDetectorConfig, ViaPolarity
from contour.vision.via_detection.result import ViaDetection

from contour.vision.via_detection.heuristic_detector_parts.candidate_features import (
    PreparedHeuristicFrame,
    _det_better,
)

from contour.vision.via_detection.heuristic_detector_parts.score_one import _score_one

def analyze_via_at(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    config: HeuristicViaDetectorConfig,
) -> ViaDetection | None:
    """Measure one caller-supplied contact center with the heuristic scorer."""

    return analyze_vias_at(image, [(center_x, center_y)], config)[0]

def analyze_vias_at(
    image: np.ndarray,
    centers: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    config: HeuristicViaDetectorConfig,
) -> list[ViaDetection | None]:
    """Measure caller-supplied contact centers while preparing the frame once."""

    config.validate()
    gray = _to_gray_u8(np.asarray(image))
    height, width = gray.shape[:2]
    if height < 3 or width < 3:
        return [None] * len(centers)
    allowed = config.allowed_diameters()
    if not allowed:
        return [None] * len(centers)
    frame = _prepare_heuristic_frame(gray, config)
    hypotheses = _candidate_hypotheses(config)
    results: list[ViaDetection | None] = []
    for center_x, center_y in centers:
        x_coord = max(0, min(width - 1, round(float(center_x))))
        y_coord = max(0, min(height - 1, round(float(center_y))))
        results.append(
            _score_seed(
                frame,
                (y_coord, x_coord),
                allowed,
                hypotheses,
                config,
                enforce_rejections=False,
            )
        )
    return results

def _prepare_heuristic_frame(
    gray: np.ndarray,
    config: HeuristicViaDetectorConfig,
) -> PreparedHeuristicFrame:
    denoised = _preprocess_denoise(_to_gray_u8(gray), config)
    background = cv2.GaussianBlur(denoised, (0, 0), float(config.background_sigma))
    gray_f32 = denoised.astype(np.float32)
    corrected = gray_f32 - background.astype(np.float32)
    corrected_u8 = _normalize01_to_u8(corrected)
    gradient_x = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
    return PreparedHeuristicFrame(
        gray=denoised,
        gray_f32=gray_f32,
        corrected_u8=corrected_u8,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
        gradient_magnitude=cv2.magnitude(gradient_x, gradient_y),
    )

def _candidate_hypotheses(config: HeuristicViaDetectorConfig) -> list[str]:
    polarity = str(config.polarity or ViaPolarity.AUTO).lower()
    if polarity in ("auto", str(ViaPolarity.AUTO)):
        hypotheses: list[str] = []
        if config.bright_range_enabled:
            hypotheses.append(str(ViaPolarity.BRIGHT))
        if config.dark_range_enabled:
            hypotheses.append(str(ViaPolarity.DARK))
        return hypotheses or [str(ViaPolarity.BRIGHT)]
    if polarity in {
        str(ViaPolarity.RING_LIGHT_RING),
        str(ViaPolarity.RING_DARK_RING),
    }:
        return [polarity]
    return [polarity]

def _score_seed(
    frame: PreparedHeuristicFrame,
    seed: tuple[int, int],
    allowed_diameters: list[int],
    hypotheses: list[str],
    config: HeuristicViaDetectorConfig,
    *,
    enforce_rejections: bool = True,
) -> ViaDetection | None:
    """Score every allowed diameter for one seed and return its best candidate."""

    cy, cx = seed
    height, width = frame.gray.shape[:2]
    patch_scale = max(1.0, float(config.analysis_window_scale))
    min_patch_size = int(config.min_analyze_size)
    best: ViaDetection | None = None
    for diameter in allowed_diameters:
        patch_size = int(max(min_patch_size, round(patch_scale * float(diameter))))
        half = patch_size // 2
        y0, y1 = max(0, cy - half), min(height, cy + half + 1)
        x0, x1 = max(0, cx - half), min(width, cx + half + 1)
        patch = frame.gray[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        patch_x, patch_y = cx - x0, cy - y0
        for hypothesis in hypotheses:
            if hypothesis in {str(ViaPolarity.AUTO), "auto"}:
                continue
            detection = _score_one(
                patch,
                patch_x,
                patch_y,
                diameter,
                (x0, y0),
                hypothesis,
                config,
                frame.gradient_x[y0:y1, x0:x1],
                frame.gradient_y[y0:y1, x0:x1],
                frame.gradient_magnitude[y0:y1, x0:x1],
                allowed_diameters,
                enforce_rejections=enforce_rejections,
            )
            if detection is not None and _det_better(detection, best):
                best = detection
    return best

def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.shape[2] >= 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image[:, :, 0].astype(np.uint8, copy=False)

def _preprocess_denoise(g: np.ndarray, config: HeuristicViaDetectorConfig) -> np.ndarray:
    if bool(config.use_bilateral):
        return cv2.bilateralFilter(
            g, int(config.bilateral_d), float(config.bilateral_sigma_color), float(config.bilateral_sigma_space)
        )
    return cv2.medianBlur(g, 3)

def _normalize01_to_u8(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi <= lo + 1e-6:
        return (np.zeros_like(a) + 128.0).astype(np.uint8)
    u = ((a - lo) / (hi - lo) * 255.0).clip(0, 255)
    return u.astype(np.uint8)

def _fast_percentile(values: np.ndarray, percentile: float) -> float:
    """Linear percentile without NumPy's comparatively costly quantile setup."""

    flat = values.reshape(-1)
    if flat.size == 0:
        return 0.0
    rank = (flat.size - 1) * max(0.0, min(100.0, float(percentile))) / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    selected = np.partition(flat, (lower, upper))
    if lower == upper:
        return float(selected[lower])
    fraction = rank - lower
    lower_value = float(selected[lower])
    result = lower_value + (float(selected[upper]) - lower_value) * fraction
    if np.issubdtype(flat.dtype, np.floating):
        # np.percentile preserves a floating input dtype, so keep the same
        # precision for float32 gradient rings.
        result = float(flat.dtype.type(result))
    return result

def _fast_percentile_u8(values: np.ndarray, percentile: float) -> float:
    return _fast_percentile(values, percentile)
