"""Segmentation for conductors whose fill is as dark as the substrate.

On such frames the metal carries no intensity of its own: the pour interior and
the bare substrate sit at the same grey level, so every seeded algorithm that
starts from bright cores fails.  What remains is the topographic edge, which the
detector renders as a bright ridge on the metal side and a dark trench on the
substrate side.  Those edges close into walls that partition the frame, and the
polarity of the wall tells which of the two neighbouring regions is metal.
"""

from __future__ import annotations

import cv2
import numpy as np

from ...utils import ensure_binary_mask, ensure_uint8
from .gradient_watershed import GradientWatershedConfig

# Structural constants: they describe how an edge is shaped, not how strong it is.
_WALL_SEAL_PX = 2
_RIM_BAND_PX = 4
_MIN_REGION_AREA_PX = 60


def boundary_relief_field(smoothed: np.ndarray, *, background_sigma: float) -> np.ndarray:
    """Height of every pixel above the local background: ridges positive, trenches negative."""
    background = cv2.GaussianBlur(smoothed, (0, 0), max(0.5, float(background_sigma)))
    return smoothed.astype(np.float32) - background.astype(np.float32)


def seal_edge_walls(ridge: np.ndarray, trench: np.ndarray, *, seal_px: int) -> np.ndarray:
    """Join ridge and trench into continuous walls so that regions stay enclosed."""
    wall = cv2.bitwise_or(ridge, trench)
    if seal_px <= 0:
        return wall
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * seal_px + 1, 2 * seal_px + 1))
    return cv2.morphologyEx(wall, cv2.MORPH_CLOSE, kernel)


def regions_lined_by_ridges(
    wall: np.ndarray,
    ridge: np.ndarray,
    trench: np.ndarray,
    *,
    rim_band_px: int,
    min_region_area: int,
) -> np.ndarray:
    """Keep the regions that a bright ridge lines from the inside.

    An edge is a dipole: its ridge lies on the metal side and its trench on the
    substrate side.  So a region is metal when more of its border touches ridges
    than trenches, which is a discrete count rather than a brightness
    comparison and therefore survives frames where metal and substrate share the
    same grey level.
    """
    interior = cv2.bitwise_not(wall)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(interior, connectivity=4)
    if count <= 1:
        return np.zeros(wall.shape, dtype=np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * rim_band_px + 1, 2 * rim_band_px + 1)
    )
    inside = interior > 0
    ridge_contact = np.bincount(labels[(cv2.dilate(ridge, kernel) > 0) & inside], minlength=count)
    trench_contact = np.bincount(labels[(cv2.dilate(trench, kernel) > 0) & inside], minlength=count)
    keep = ridge_contact > trench_contact
    keep &= stats[:, cv2.CC_STAT_AREA] >= max(1, int(min_region_area))
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def closed_boundary_mask(gray: np.ndarray, config: GradientWatershedConfig) -> np.ndarray:
    """Return the metal mask built from closed edge walls rather than from brightness."""
    source = ensure_uint8(gray)
    if source.ndim != 2 or source.size == 0:
        return np.zeros(source.shape[:2], dtype=np.uint8)

    smoothed = cv2.GaussianBlur(source, (0, 0), max(0.1, float(config.smoothing_sigma)))
    relief = boundary_relief_field(smoothed, background_sigma=config.boundary_background_sigma)
    threshold = max(1.0, float(config.boundary_relief))
    ridge = (relief >= threshold).astype(np.uint8) * 255
    trench = (relief <= -threshold).astype(np.uint8) * 255
    if not np.any(ridge):
        return np.zeros(source.shape[:2], dtype=np.uint8)

    wall = seal_edge_walls(ridge, trench, seal_px=_WALL_SEAL_PX)
    filled = regions_lined_by_ridges(
        wall,
        ridge,
        trench,
        rim_band_px=_RIM_BAND_PX,
        min_region_area=_MIN_REGION_AREA_PX,
    )
    return ensure_binary_mask(cv2.bitwise_or(filled, ridge))
