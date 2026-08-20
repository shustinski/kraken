"""Prototype: segment by closed edge walls, then label each region by the polarity of its rim.

On 1111 the pour interior has the same brightness as the substrate, so no
intensity threshold can separate them.  The only signal is the topographic edge:
a bright ridge on the metal side and a dark trench on the substrate side.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(r"d:\code\kraken\plugins\contour\src")))

from contour.serializers import load_polygons_cif

JPG = Path(r"D:\OZI\Нейронка\jpg_metal")
CIF = Path(r"D:\OZI\Нейронка\cif_metal")


def load(name: str):
    gray = cv2.imdecode(np.fromfile(str(JPG / f"{name}.jpg"), np.uint8), cv2.IMREAD_GRAYSCALE)
    _, _, polys = load_polygons_cif(str(CIF / f"{name}.cif"))
    masks, gt = [], np.zeros(gray.shape, np.uint8)
    for p in polys:
        m = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(m, [np.array(p.points, np.int32).reshape(-1, 1, 2)], 255)
        masks.append(m)
        gt = cv2.bitwise_or(gt, m)
    return gray, masks, gt


def score(pred, masks, gt):
    inter = np.count_nonzero((pred > 0) & (gt > 0))
    recall = inter / max(np.count_nonzero(gt > 0), 1)
    precision = inter / max(np.count_nonzero(pred > 0), 1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((pred > 0).astype(np.uint8), 8)
    ious = []
    for m in masks:
        area = int(np.count_nonzero(m))
        sub = labels[m > 0]
        best = 0.0
        for idx in np.unique(sub):
            if idx == 0 or stats[idx, cv2.CC_STAT_AREA] < 120:
                continue
            i = int(np.count_nonzero(sub == idx))
            best = max(best, i / max(area + int(stats[idx, cv2.CC_STAT_AREA]) - i, 1))
        ious.append(best)
    arr = np.asarray(ious, float)
    return recall, precision, float(arr.mean()), float((arr > 0.7).mean())


def enclosure_mask(
    gray: np.ndarray,
    *,
    sigma: float = 1.5,
    background_sigma: float = 12.0,
    relief: float = 8.0,
    seal_px: int = 3,
    rim_band_px: int = 6,
) -> np.ndarray:
    smoothed = cv2.GaussianBlur(gray, (0, 0), sigma).astype(np.float32)
    background = cv2.GaussianBlur(gray, (0, 0), background_sigma).astype(np.float32)
    relief_field = smoothed - background
    ridge = relief_field >= relief
    trench = relief_field <= -relief
    wall = (ridge | trench).astype(np.uint8) * 255
    if seal_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * seal_px + 1, 2 * seal_px + 1))
        wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, kernel)

    interior = cv2.bitwise_not(wall)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(interior, connectivity=4)
    band_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * rim_band_px + 1, 2 * rim_band_px + 1)
    )
    near_wall = cv2.dilate(wall, band_kernel) > 0

    out = np.zeros(gray.shape, np.uint8)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 60:
            continue
        region = labels == index
        band = region & near_wall
        core = region & ~near_wall
        if not band.any() or not core.any():
            continue
        if float(np.median(smoothed[band])) > float(np.median(smoothed[core])):
            out[region] = 255
    return cv2.bitwise_or(out, (ridge.astype(np.uint8) * 255))


if __name__ == "__main__":
    for name in ("1111",):
        gray, masks, gt = load(name)
        for relief in (5.0, 8.0, 12.0):
            for seal in (2, 4, 6):
                mask = enclosure_mask(gray, relief=relief, seal_px=seal)
                rec, prec, miou, good = score(mask, masks, gt)
                print(f"{name} relief={relief:4.1f} seal={seal}: recall={rec:.3f} precision={prec:.3f} "
                      f"mIoU={miou:.3f} good={good:.2f}")
