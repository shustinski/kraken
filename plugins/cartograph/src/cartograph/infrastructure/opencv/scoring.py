"""ZNCC and structural (gradient-magnitude) scoring. OpenCV stays in this module."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from cartograph.domain.coordinates import Translation2D


Gray = NDArray[np.float32]


def gaussian_smooth(image: Gray, sigma: float) -> Gray:
    if sigma <= 0.0:
        return np.ascontiguousarray(image, dtype=np.float32)
    blurred = cv2.GaussianBlur(np.asarray(image, dtype=np.float32), ksize=(0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    return np.ascontiguousarray(blurred, dtype=np.float32)


def gradient_magnitude(image: Gray, sigma: float) -> Gray:
    smoothed = gaussian_smooth(image, sigma)
    grad_x = cv2.Scharr(smoothed, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(smoothed, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return np.ascontiguousarray(magnitude, dtype=np.float32)


def mean_gradient(image: Gray, sigma: float) -> float:
    return float(np.mean(gradient_magnitude(image, sigma)))


def zncc(fixed: Gray, moving: Gray) -> float:
    if fixed.size == 0 or moving.size == 0 or fixed.shape != moving.shape:
        return 0.0
    left = np.asarray(fixed, dtype=np.float64).ravel()
    right = np.asarray(moving, dtype=np.float64).ravel()
    left = left - left.mean()
    right = right - right.mean()
    denom = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def zncc_at_shift(fixed: Gray, moving: Gray, shift: Translation2D) -> float:
    warped = warp_translation(moving, shift.dx, shift.dy, output_shape=fixed.shape)
    mask = warp_translation(np.ones_like(moving, dtype=np.float32), shift.dx, shift.dy, output_shape=fixed.shape)
    valid = mask > 0.5
    if int(valid.sum()) < 16:
        return 0.0
    return zncc(fixed[valid], warped[valid])


def zncc_translation_candidates(
    fixed: Gray,
    moving: Gray,
    *,
    search_radius_px: float,
    top_k: int,
    coarse_step: float = 0.5,
    refine_step: float = 0.05,
) -> tuple[Translation2D, ...]:
    """Deterministic ZNCC search for subpixel residuals inside a pre-aligned overlap."""

    radius = float(search_radius_px)
    coarse_step = max(float(coarse_step), 0.25)
    refine_step = max(float(refine_step), 0.01)
    coarse: list[tuple[float, float, float]] = []
    xs = np.arange(-radius, radius + 1e-9, coarse_step, dtype=np.float64)
    ys = np.arange(-radius, radius + 1e-9, coarse_step, dtype=np.float64)
    for dx in xs:
        for dy in ys:
            if abs(dx) > radius or abs(dy) > radius:
                continue
            shift = Translation2D(float(dx), float(dy))
            coarse.append((zncc_at_shift(fixed, moving, shift), float(dx), float(dy)))
    coarse.sort(key=lambda item: item[0], reverse=True)

    seen: set[tuple[int, int]] = set()
    refined: list[tuple[float, float, float]] = []
    for _, seed_dx, seed_dy in coarse[: max(top_k, 3)]:
        for dx in np.arange(seed_dx - coarse_step, seed_dx + coarse_step + 1e-9, refine_step):
            for dy in np.arange(seed_dy - coarse_step, seed_dy + coarse_step + 1e-9, refine_step):
                if abs(dx) > radius or abs(dy) > radius:
                    continue
                key = (int(round(dx * 100.0)), int(round(dy * 100.0)))
                if key in seen:
                    continue
                seen.add(key)
                shift = Translation2D(float(dx), float(dy))
                refined.append((zncc_at_shift(fixed, moving, shift), float(dx), float(dy)))
    refined.sort(key=lambda item: item[0], reverse=True)
    ordered = refined if refined else coarse
    return tuple(Translation2D(dx, dy) for _, dx, dy in ordered[:top_k])


def warp_translation(image: Gray, dx: float, dy: float, output_shape: tuple[int, ...]) -> Gray:
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    warped = cv2.warpAffine(
        np.asarray(image, dtype=np.float32),
        matrix,
        (int(output_shape[1]), int(output_shape[0])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return np.ascontiguousarray(warped, dtype=np.float32)
