"""structural_watershed: extract structural features."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

from contour.vision.metal_recovery.structural_watershed_parts.module_state import (
    _DENSE_RIDGE_SCALE_STEP,
    _FRANGI_BETA,
    _RIDGE_SCALE_STEP,
)

@dataclass(frozen=True, slots=True)
class _StructuralFeatures:
    denoised: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    magnitude: np.ndarray
    structure_orientation: np.ndarray
    coherence: np.ndarray
    ridge_response: np.ndarray
    ridge_orientation: np.ndarray
    ridge_confidence: np.ndarray
    oriented_gradient: np.ndarray
    rim_response: np.ndarray
    persistent_edge: np.ndarray

def _extract_structural_features(
    gray: np.ndarray,
    config: StructuralWatershedConfig,
) -> _StructuralFeatures:
    denoised = cv2.GaussianBlur(gray, (0, 0), max(0.1, float(config.smoothing_sigma)))
    gradient_x = cv2.Scharr(denoised, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Scharr(denoised, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    orientation, coherence = _structure_tensor_field(
        gradient_x,
        gradient_y,
        smoothing_sigma=float(config.orientation_smoothing_scale),
    )
    ridge_response, ridge_orientation, ridge_confidence = _multiscale_ridge_evidence(
        denoised,
        config,
    )
    across_x = np.cos(orientation)
    across_y = np.sin(orientation)
    oriented_gradient = np.abs(gradient_x * across_x + gradient_y * across_y)
    rim_response = _bright_rim_response(denoised, magnitude)
    persistent_edge = cv2.GaussianBlur(
        oriented_gradient,
        (0, 0),
        max(1.0, float(config.orientation_smoothing_scale) * 0.5),
    )
    return _StructuralFeatures(
        denoised=denoised,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
        magnitude=magnitude,
        structure_orientation=orientation,
        coherence=coherence,
        ridge_response=ridge_response,
        ridge_orientation=ridge_orientation,
        ridge_confidence=ridge_confidence,
        oriented_gradient=oriented_gradient,
        rim_response=rim_response,
        persistent_edge=persistent_edge,
    )

def _structure_tensor_field(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    *,
    smoothing_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (edge orientation in radians, coherence) from the averaged structure tensor."""

    jxx = cv2.GaussianBlur(gradient_x * gradient_x, (0, 0), smoothing_sigma)
    jyy = cv2.GaussianBlur(gradient_y * gradient_y, (0, 0), smoothing_sigma)
    jxy = cv2.GaussianBlur(gradient_x * gradient_y, (0, 0), smoothing_sigma)
    trace = jxx + jyy
    delta = cv2.magnitude(jxx - jyy, 2.0 * jxy)
    lambda_max = 0.5 * (trace + delta)
    coherence = np.divide(delta, np.maximum(trace, 1e-6)).astype(np.float32)
    coherence = np.clip(coherence, 0.0, 1.0)
    orientation = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy).astype(np.float32)
    orientation = np.where(lambda_max <= 1e-6, 0.0, orientation).astype(np.float32)
    return orientation, coherence

def _ridge_scales(config: StructuralWatershedConfig) -> tuple[float, ...]:
    step = _DENSE_RIDGE_SCALE_STEP if config.variant == "s4" else _RIDGE_SCALE_STEP
    scale = float(config.ridge_scale_min)
    limit = float(config.ridge_scale_max)
    values: list[float] = []
    while scale <= limit + 1e-6:
        values.append(float(scale))
        scale *= step
    if not values:
        values.append(float(config.ridge_scale_min))
    if values[-1] < limit - 1e-3:
        values.append(limit)
    return tuple(values)

def _multiscale_ridge_evidence(
    denoised: np.ndarray,
    config: StructuralWatershedConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale-selected Hessian ridge: response, along-ridge orientation, confidence.

    Each scale is scored independently. The pixel keeps the strongest scale;
    binary detections are never OR-combined.
    """

    source = denoised.astype(np.float32)
    best_response = np.zeros(source.shape, dtype=np.float32)
    best_orientation = np.zeros(source.shape, dtype=np.float32)
    for sigma in _ridge_scales(config):
        response, orientation = _hessian_ridge_at_scale(source, sigma)
        stronger = response > best_response
        best_response = np.where(stronger, response, best_response).astype(np.float32)
        best_orientation = np.where(stronger, orientation, best_orientation).astype(np.float32)
    peak = float(np.percentile(best_response, 99.5)) if best_response.size else 0.0
    confidence = np.clip(best_response / max(peak, 1e-6), 0.0, 1.0).astype(np.float32)
    return best_response, best_orientation, confidence

def _hessian_ridge_at_scale(source: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    blurred = cv2.GaussianBlur(source, (0, 0), max(0.4, float(sigma)))
    lx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    ly = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    lxx: np.ndarray = cv2.Sobel(lx, cv2.CV_32F, 1, 0, ksize=3)
    lxy: np.ndarray = cv2.Sobel(lx, cv2.CV_32F, 0, 1, ksize=3)
    lyy: np.ndarray = cv2.Sobel(ly, cv2.CV_32F, 0, 1, ksize=3)
    scale = float(sigma) * float(sigma)
    lxx *= scale
    lxy *= scale
    lyy *= scale
    tmp = cv2.magnitude(lxx - lyy, 2.0 * lxy)
    lambda_pos = 0.5 * (lxx + lyy + tmp)
    lambda_neg = 0.5 * (lxx + lyy - tmp)
    mag_pos = np.abs(lambda_pos)
    mag_neg = np.abs(lambda_neg)
    lambda_across = np.where(mag_pos >= mag_neg, lambda_pos, lambda_neg)
    lambda_along = np.where(mag_pos >= mag_neg, lambda_neg, lambda_pos)
    across_strength = np.abs(lambda_across)
    along_strength = np.abs(lambda_along)
    rb = along_strength / np.maximum(across_strength, 1e-6)
    hessian_norm = cv2.magnitude(lambda_across, lambda_along)
    frangi_c = max(float(np.percentile(hessian_norm, 90.0)), 1e-3)
    vesselness = np.exp(-(rb * rb) / (_FRANGI_BETA * _FRANGI_BETA)) * (
        1.0 - np.exp(-(hessian_norm * hessian_norm) / (frangi_c * frangi_c))
    )
    # Bright ridges only: dark grooves are separators, not conductor markers.
    vesselness = np.where(
        (lambda_across < 0.0) & (across_strength > along_strength),
        vesselness,
        0.0,
    ).astype(np.float32)
    across_x = lxy
    across_y = lambda_across - lxx
    fallback = (np.abs(across_x) + np.abs(across_y)) < 1e-6
    across_x = np.where(fallback, 1.0, across_x)
    across_y = np.where(fallback, 0.0, across_y)
    along_orientation = np.arctan2(across_x, -across_y).astype(np.float32)
    return vesselness, along_orientation

def _bright_rim_response(denoised: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    from contour.vision.metal_recovery.structural_watershed_parts.geodesic_label_competition import _unit_range
    background = cv2.GaussianBlur(denoised.astype(np.float32), (0, 0), 6.0)
    relief = np.maximum(denoised.astype(np.float32) - background, 0.0)
    return (relief * _unit_range(magnitude)).astype(np.float32)

def _angles_aligned(first: np.ndarray, second: np.ndarray, *, max_delta: float) -> np.ndarray:
    delta = np.abs(np.mod(first - second + np.pi * 0.5, np.pi) - np.pi * 0.5)
    return delta <= max_delta
