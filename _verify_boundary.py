"""Production closed-boundary strategy versus the tuned prototype."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(r"d:\code\kraken\plugins\contour\src")))
sys.path.insert(0, r"d:\code\kraken")

from _proto_enclosure import enclosure_mask, load, score  # noqa: E402

from contour.vision.metal_recovery.gradient_watershed import GradientWatershedConfig  # noqa: E402
from contour.vision.metal_recovery.seeded_segmentation import seeded_segmentation_mask  # noqa: E402
from contour.vision.metal_recovery.segmentation import (  # noqa: E402
    MetalSegmentationConfig,
    apply_topology_repair,
    filter_mask_components,
    is_seeded_segmentation_strategy,
    normalize_metal_segmentation_strategy,
)

for token in ("closed_boundary", "Замкнутые границы", "boundary", "closed-boundary"):
    normalized = normalize_metal_segmentation_strategy(token)
    print(f"  {token!r} -> {normalized!r} seeded={is_seeded_segmentation_strategy(token)}")

POST = MetalSegmentationConfig(
    gap_bridge_px=4, speckle_removal_px=0, min_component_area=60,
    segmentation_strategy="closed_boundary",
)


def finished(mask):
    return filter_mask_components(apply_topology_repair(mask, POST), POST.min_component_area)


config = GradientWatershedConfig()
print(f"\ndefaults: relief={config.boundary_relief} background_sigma={config.boundary_background_sigma}")
print("frame | production rec prec mIoU good (s) | prototype mIoU")
for name in ("1111", "0000", "0001", "1079", "3104", "3195"):
    gray, masks, gt = load(name)
    started = time.perf_counter()
    mask = finished(seeded_segmentation_mask(gray, "closed_boundary", config))
    elapsed = time.perf_counter() - started
    rec, prec, miou, good = score(mask, masks, gt)
    reference = score(
        finished(enclosure_mask(gray, relief=16.0, background_sigma=12.0, seal_px=2, rim_band_px=4)),
        masks,
        gt,
    )
    print(f"{name} | {rec:.2f} {prec:.2f} {miou:.3f} {good:.2f} ({elapsed:4.1f}) | {reference[2]:.3f}")
