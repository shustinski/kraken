"""Confidence-map normalization and uncertainty overlay utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - exercised indirectly when OpenCV is installed.
    import cv2
except Exception:  # pragma: no cover - fallback keeps unit logic importable.
    cv2 = None


DEFAULT_CONFIDENCE_BAD_AREA_THRESHOLD = 0.80
DEFAULT_ALGORITHMIC_BOUNDARY_RADIUS = 3.0


@dataclass(frozen=True, slots=True)
class CombinedUncertaintyMaps:
    """Store aligned bad-area intensities for two uncertainty sources."""

    algorithmic_only: np.ndarray
    model_only: np.ndarray
    agreement: np.ndarray
    algorithmic_badness: np.ndarray
    model_badness: np.ndarray


def normalize_unit_map(values: np.ndarray | object) -> np.ndarray:
    """Return a finite float32 map in [0, 1] without assuming mask semantics."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        return np.zeros((1, 1), dtype=np.float32)
    if array.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    finite = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    if finite.size and float(np.nanmax(finite)) > 1.0:
        finite = finite / 255.0
    return np.clip(finite, 0.0, 1.0).astype(np.float32, copy=False)


def normalize_algorithmic_confidence(prediction_map: np.ndarray | object) -> np.ndarray:
    """Normalize a prediction-like map where black is background and bright is object."""

    probability = normalize_unit_map(prediction_map)
    return np.clip(2.0 * np.abs(probability - np.float32(0.5)), 0.0, 1.0).astype(np.float32, copy=False)


def normalize_model_confidence(confidence_map: np.ndarray | object) -> np.ndarray:
    """Normalize a model confidence output map.

    Model confidence outputs are treated as confidence values, not as prediction
    masks. A white background therefore remains high confidence and does not
    become foreground support.
    """

    return normalize_unit_map(confidence_map)


def build_model_uncertainty(confidence_map: np.ndarray | object) -> np.ndarray:
    """Build model-output uncertainty with maximum uncertainty near 0.5."""

    confidence = normalize_model_confidence(confidence_map)
    return np.clip(1.0 - 2.0 * np.abs(confidence - np.float32(0.5)), 0.0, 1.0).astype(np.float32, copy=False)


def _is_binary_like(values: np.ndarray, *, tolerance: float = 1e-4) -> bool:
    array = normalize_unit_map(values)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return False
    distance = np.minimum(np.abs(finite), np.abs(finite - 1.0))
    return bool(np.max(distance, initial=0.0) <= float(max(1e-8, tolerance)))


def _resize_to_shape(values: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_height = max(1, int(target_shape[0]))
    target_width = max(1, int(target_shape[1]))
    array = normalize_unit_map(values)
    if array.shape == (target_height, target_width):
        return array
    if array.size == 0:
        return np.zeros((target_height, target_width), dtype=np.float32)
    if cv2 is not None:
        resized = cv2.resize(array, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        return np.clip(resized, 0.0, 1.0).astype(np.float32, copy=False)
    y_index = np.linspace(0, array.shape[0] - 1, target_height).round().astype(np.int32)
    x_index = np.linspace(0, array.shape[1] - 1, target_width).round().astype(np.int32)
    return array[y_index][:, x_index].astype(np.float32, copy=False)


def _binary_boundary_uncertainty(mask: np.ndarray, *, radius: float) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2:
        return np.zeros((1, 1), dtype=np.float32)
    if mask_bool.size == 0:
        return np.zeros_like(mask_bool, dtype=np.float32)
    if not np.any(mask_bool) or np.all(mask_bool):
        return np.zeros(mask_bool.shape, dtype=np.float32)
    radius_value = max(1e-6, float(radius))
    if cv2 is not None:
        inside = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 3)
        outside = cv2.distanceTransform((~mask_bool).astype(np.uint8), cv2.DIST_L2, 3)
        distance = np.where(mask_bool, inside, outside).astype(np.float32)
        return np.clip(1.0 - distance / radius_value, 0.0, 1.0).astype(np.float32, copy=False)
    padded = np.pad(mask_bool, 1, mode="edge")
    boundary = (
        (padded[1:-1, 1:-1] != padded[:-2, 1:-1])
        | (padded[1:-1, 1:-1] != padded[2:, 1:-1])
        | (padded[1:-1, 1:-1] != padded[1:-1, :-2])
        | (padded[1:-1, 1:-1] != padded[1:-1, 2:])
    )
    return boundary.astype(np.float32)


def build_algorithmic_uncertainty(
    prediction_map: np.ndarray | object,
    *,
    boundary_radius: float = DEFAULT_ALGORITHMIC_BOUNDARY_RADIUS,
) -> np.ndarray:
    """Build uncertainty derived from a prediction result, not from model confidence output."""

    probability = normalize_unit_map(prediction_map)
    if _is_binary_like(probability):
        return _binary_boundary_uncertainty(probability >= 0.5, radius=float(boundary_radius))
    confidence = normalize_algorithmic_confidence(probability)
    return np.clip(1.0 - confidence, 0.0, 1.0).astype(np.float32, copy=False)


def confidence_bad_area_intensity(
    uncertainty: np.ndarray | object,
    *,
    threshold: float = DEFAULT_CONFIDENCE_BAD_AREA_THRESHOLD,
    gamma: float = 0.65,
) -> np.ndarray:
    """Convert uncertainty into a thresholded visual bad-area intensity map."""

    values = normalize_unit_map(uncertainty)
    cutoff = float(np.clip(threshold, 0.0, 0.999))
    scaled = np.zeros_like(values, dtype=np.float32)
    active = values > cutoff
    if not np.any(active):
        return scaled
    scaled[active] = (values[active] - cutoff) / max(1e-8, 1.0 - cutoff)
    gamma_value = max(0.05, float(gamma))
    if abs(gamma_value - 1.0) > 1e-8:
        scaled[active] = np.power(scaled[active], gamma_value)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32, copy=False)


def combine_uncertainty_maps(
    algorithmic_uncertainty: np.ndarray | object,
    model_uncertainty: np.ndarray | object,
    *,
    algorithmic_threshold: float = DEFAULT_CONFIDENCE_BAD_AREA_THRESHOLD,
    model_threshold: float = DEFAULT_CONFIDENCE_BAD_AREA_THRESHOLD,
) -> CombinedUncertaintyMaps:
    """Align and combine algorithmic and model-output uncertainty maps."""

    algorithmic = normalize_unit_map(algorithmic_uncertainty)
    model = _resize_to_shape(model_uncertainty, algorithmic.shape)
    algorithmic_badness = confidence_bad_area_intensity(algorithmic, threshold=float(algorithmic_threshold))
    model_badness = confidence_bad_area_intensity(model, threshold=float(model_threshold))
    agreement = np.minimum(algorithmic_badness, model_badness).astype(np.float32, copy=False)
    algorithmic_only = np.clip(algorithmic_badness - model_badness, 0.0, 1.0).astype(np.float32, copy=False)
    model_only = np.clip(model_badness - algorithmic_badness, 0.0, 1.0).astype(np.float32, copy=False)
    return CombinedUncertaintyMaps(
        algorithmic_only=algorithmic_only,
        model_only=model_only,
        agreement=agreement,
        algorithmic_badness=algorithmic_badness,
        model_badness=model_badness,
    )
