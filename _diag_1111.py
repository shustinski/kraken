"""First look at frame 1111: intensity statistics, ground truth geometry, current result."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(r"d:\code\kraken\plugins\contour\src")))

from contour.serializers import load_polygons_cif
from contour.vision.metal_recovery import MetalRecoveryConfig, detect_metalization
from contour.vision.metal_recovery.detector import clear_metal_contour_cache
from contour.vision.metal_recovery.gradient_watershed import (
    GradientWatershedConfig,
    gradient_watershed_mask,
    intensity_class_limits,
)
from contour.vision.metal_recovery.pipeline_stages import clear_metal_segmentation_cache

JPG = Path(r"D:\OZI\Нейронка\jpg_metal")
CIF = Path(r"D:\OZI\Нейронка\cif_metal")
name = "1111"

gray = cv2.imdecode(np.fromfile(str(JPG / f"{name}.jpg"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
_, _, polys = load_polygons_cif(str(CIF / f"{name}.cif"))
gt = np.zeros(gray.shape, np.uint8)
for p in polys:
    cv2.fillPoly(gt, [np.array(p.points, np.int32).reshape(-1, 1, 2)], 255)

smoothed = cv2.GaussianBlur(gray, (0, 0), 1.0)
substrate_limit, metal_limit = intensity_class_limits(smoothed)

print(f"frame {name}: {gray.shape[1]}x{gray.shape[0]} polygons={len(polys)} gt_fill={np.mean(gt>0):.3f}")
print(f"  otsu limits: substrate={substrate_limit:.0f} metal={metal_limit:.0f}")
print(f"  substrate: p05={np.percentile(gray[gt==0],5):.0f} p50={np.percentile(gray[gt==0],50):.0f} "
      f"p95={np.percentile(gray[gt==0],95):.0f}")
print(f"  metal:     p05={np.percentile(gray[gt>0],5):.0f} p50={np.percentile(gray[gt>0],50):.0f} "
      f"p95={np.percentile(gray[gt>0],95):.0f}")

widths = np.array([p.bbox[2] for p in polys], float)
heights = np.array([p.bbox[3] for p in polys], float)
areas = np.array([abs(cv2.contourArea(np.array(p.points, np.int32))) for p in polys], float)
print(f"  polygon width  p10={np.percentile(widths,10):.0f} p50={np.percentile(widths,50):.0f} max={widths.max():.0f}")
print(f"  polygon height p10={np.percentile(heights,10):.0f} p50={np.percentile(heights,50):.0f} max={heights.max():.0f}")
print(f"  polygon area   p10={np.percentile(areas,10):.0f} p50={np.percentile(areas,50):.0f} max={areas.max():.0f}")

gap = cv2.distanceTransform((gt == 0).astype(np.uint8), cv2.DIST_L2, 3)
print(f"  gap half-width p50={np.percentile(gap[gt==0],50):.1f} p90={np.percentile(gap[gt==0],90):.1f}")


def score(pred, gt_mask, polygons):
    inter = np.count_nonzero((pred > 0) & (gt_mask > 0))
    recall = inter / max(np.count_nonzero(gt_mask > 0), 1)
    precision = inter / max(np.count_nonzero(pred > 0), 1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((pred > 0).astype(np.uint8), 8)
    ious = []
    for p in polygons:
        m = np.zeros(gt_mask.shape, np.uint8)
        cv2.fillPoly(m, [np.array(p.points, np.int32).reshape(-1, 1, 2)], 255)
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


mask = gradient_watershed_mask(gray, GradientWatershedConfig())
rec, prec, miou, good = score(mask, gt, polys)
print(f"\n  watershed mask: recall={rec:.3f} precision={prec:.3f} mIoU={miou:.3f} good={good:.2f}")

for enabled in (False, True):
    clear_metal_segmentation_cache()
    clear_metal_contour_cache()
    result = detect_metalization(gray, MetalRecoveryConfig(use_wide_conductor_gradient=enabled))
    pred = np.zeros(gray.shape, np.uint8)
    for p in result.accepted:
        pts = np.array(p.points, np.int32).reshape(-1, 1, 2)
        if pts.shape[0] >= 3:
            cv2.fillPoly(pred, [pts], 255)
    rec, prec, miou, good = score(pred, gt, polys)
    label = "watershed" if enabled else "otsu     "
    print(f"  detector {label}: polys={len(result.accepted):4d} recall={rec:.3f} precision={prec:.3f} "
          f"mIoU={miou:.3f} good={good:.2f}")

view = cv2.cvtColor(cv2.resize(gray, (800, 800)), cv2.COLOR_GRAY2BGR)
small_gt = cv2.resize(gt, (800, 800), interpolation=cv2.INTER_NEAREST) > 0
view[small_gt] = (0.5 * view[small_gt] + np.array([0, 0, 127])).astype(np.uint8)
cv2.imwrite(r"d:\code\kraken\_1111_overview.png", view)
crop = gray[700:1100, 700:1100]
overlay = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
sel = gt[700:1100, 700:1100] > 0
overlay[sel] = (0.5 * overlay[sel] + np.array([0, 0, 127])).astype(np.uint8)
cv2.imwrite(r"d:\code\kraken\_1111_crop.png",
            cv2.resize(overlay, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
print("\nwrote _1111_overview.png and _1111_crop.png")
