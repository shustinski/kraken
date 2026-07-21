"""Lossless via extraction for binary and JPEG-damaged binary masks."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import cv2
import numpy as np

from ..io_normalize import to_gray_u8


@dataclass(frozen=True, slots=True)
class BinaryViaComponent:
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: float
    circularity: float
    aspect: float


def looks_like_binary_mask(image: np.ndarray) -> bool:
    """Recognize masks without mistaking narrow-range SEM images for masks."""

    gray = to_gray_u8(image)
    if gray.size == 0:
        return False
    near_extremes = (gray <= 12) | (gray >= 243)
    if float(np.mean(near_extremes)) < 0.97:
        return False
    return float(np.mean(gray <= 32)) > 1e-6 and float(np.mean(gray >= 223)) > 1e-6


def detect_binary_components(
    image: np.ndarray,
    *,
    diameter_min: float,
    diameter_max: float,
    min_area_factor: float,
    max_area_factor: float,
    min_aspect: float,
    max_aspect: float,
    nms_distance: float,
) -> tuple[np.ndarray, list[BinaryViaComponent]]:
    gray = to_gray_u8(image)
    _threshold, foreground = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(cv2.countNonZero(foreground)) / float(foreground.size) > 0.5:
        foreground = cv2.bitwise_not(foreground)

    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    min_span = max(1.0, float(diameter_min) * 0.65)
    max_span = max(min_span, float(diameter_max) * 1.5)
    min_area = pi * (float(diameter_min) * 0.5) ** 2 * float(min_area_factor)
    max_area = pi * (float(diameter_max) * 0.5) ** 2 * float(max_area_factor)
    ranked: list[BinaryViaComponent] = []
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = float(stats[label, cv2.CC_STAT_AREA])
        if not (min_span <= width <= max_span and min_span <= height <= max_span):
            continue
        if not (min_area * 0.65 <= area <= max_area * 1.35):
            continue
        aspect = float(width) / float(max(1, height))
        if not (float(min_aspect) <= aspect <= float(max_aspect)):
            continue
        cx, cy = (float(value) for value in centroids[label])
        perimeter = float(2 * (width + height))
        ranked.append(
            BinaryViaComponent(
                center=(cx, cy),
                bbox=(x, y, width, height),
                area=area,
                circularity=float(4.0 * pi * area / max(perimeter * perimeter, 1e-6)),
                aspect=aspect,
            )
        )

    kept: list[BinaryViaComponent] = []
    distance_sq = float(max(0.0, nms_distance)) ** 2
    for component in sorted(ranked, key=lambda item: item.area, reverse=True):
        if distance_sq > 0.0 and any(
            (component.center[0] - other.center[0]) ** 2 + (component.center[1] - other.center[1]) ** 2 < distance_sq
            for other in kept
        ):
            continue
        kept.append(component)
    kept.sort(key=lambda item: (item.center[1], item.center[0]))
    return foreground, kept
