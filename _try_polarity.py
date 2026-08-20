"""Two ways to decide the side of a wall: rim brightness versus ridge/trench contact."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(r"d:\code\kraken\plugins\contour\src")))
sys.path.insert(0, r"d:\code\kraken")

from _proto_enclosure import load, score  # noqa: E402

from contour.vision.metal_recovery.closed_boundary import (  # noqa: E402
    _RIM_BAND_PX,
    _WALL_SEAL_PX,
    boundary_relief_field,
    seal_edge_walls,
)
from contour.vision.metal_recovery.segmentation import (  # noqa: E402
    MetalSegmentationConfig,
    apply_topology_repair,
    filter_mask_components,
)

POST = MetalSegmentationConfig(
    gap_bridge_px=4, speckle_removal_px=0, min_component_area=60,
    segmentation_strategy="closed_boundary",
)


def contact_mask(gray, *, relief=16.0, background_sigma=12.0, sigma=1.0, band=_RIM_BAND_PX):
    smoothed = cv2.GaussianBlur(gray, (0, 0), sigma)
    field = boundary_relief_field(smoothed, background_sigma=background_sigma)
    ridge = (field >= relief).astype(np.uint8) * 255
    trench = (field <= -relief).astype(np.uint8) * 255
    wall = seal_edge_walls(ridge, trench, seal_px=_WALL_SEAL_PX)
    interior = cv2.bitwise_not(wall)
    count, labels, stats, _c = cv2.connectedComponentsWithStats(interior, connectivity=4)
    if count <= 1:
        return np.zeros(gray.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    near_ridge = (cv2.dilate(ridge, kernel) > 0) & (interior > 0)
    near_trench = (cv2.dilate(trench, kernel) > 0) & (interior > 0)
    ridge_touch = np.bincount(labels[near_ridge], minlength=count)
    trench_touch = np.bincount(labels[near_trench], minlength=count)
    keep = ridge_touch > trench_touch
    keep &= stats[:, cv2.CC_STAT_AREA] >= 60
    keep[0] = False
    filled = np.where(keep[labels], 255, 0).astype(np.uint8)
    return cv2.bitwise_or(filled, ridge)


def finished(mask):
    return filter_mask_components(apply_topology_repair(mask, POST), 60)


print("contact rule")
for sigma in (1.0, 1.5):
    cells = []
    for name in ("1111", "0000", "0001", "1079", "3104", "3195"):
        gray, masks, gt = load(name)
        rec, prec, miou, _good = score(finished(contact_mask(gray, sigma=sigma)), masks, gt)
        cells.append(f"{name}: {rec:.2f} {prec:.2f} {miou:.3f}")
    print(f"  sigma={sigma} | " + " | ".join(cells))
