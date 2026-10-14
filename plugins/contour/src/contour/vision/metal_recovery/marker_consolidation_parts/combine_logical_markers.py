"""marker_consolidation: combine logical markers."""
from __future__ import annotations
import cv2
import numpy as np
from contour.vision.metal_recovery.conductor_bands import (
    evidence_from_consolidation,
    region_has_internal_separator,
)

from contour.vision.metal_recovery.marker_consolidation_parts.render_debug import _compact_labels

def _combine_logical_markers(
    ridge_labels: np.ndarray,
    wide_labels: np.ndarray,
    evidence: ConsolidationEvidence | None = None,
    *,
    separator_aware: bool = False,
) -> tuple[np.ndarray, int, int]:
    """Keep a wide region unless a real interior separator splits it.

    Ridge count inside the region is evidence, not a reason to drop the plate.
    """

    if not np.any(ridge_labels):
        return _compact_labels(wide_labels), 0, 0
    if not np.any(wide_labels):
        return _compact_labels(ridge_labels), 0, 0

    ridge_max = int(ridge_labels.max())
    wide_max = int(wide_labels.max())
    keep_wide = np.ones(wide_max + 1, dtype=bool)
    keep_wide[0] = False
    eroded = cv2.erode((wide_labels > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
    interior = (ridge_labels > 0) & (eroded > 0)
    ridge_hits = np.zeros(wide_max + 1, dtype=np.int32)
    if np.any(interior):
        stride = ridge_max + 1
        encoded = wide_labels.astype(np.int64) * stride + ridge_labels.astype(np.int64)
        unique_pairs = np.unique(encoded[interior])
        wide_ids = (unique_pairs // stride).astype(np.int32)
        ridge_hits = np.bincount(wide_ids, minlength=wide_max + 1).astype(np.int32)
        if not separator_aware:
            keep_wide = ridge_hits <= 1
            keep_wide[0] = False

    kept_multi = 0
    dropped_sep = 0
    if separator_aware and evidence is not None:
        band_evidence = evidence_from_consolidation(evidence)
        for wide_id in range(1, wide_max + 1):
            if ridge_hits[wide_id] <= 1:
                continue
            region = wide_labels == wide_id
            if region_has_internal_separator(region, band_evidence):
                keep_wide[wide_id] = False
                dropped_sep += 1
            else:
                kept_multi += 1

    combined = np.zeros(wide_labels.shape, dtype=np.int32)
    wide_ids = np.unique(wide_labels)
    wide_ids = wide_ids[wide_ids > 0]
    wide_map = np.zeros(wide_max + 1, dtype=np.int32)
    next_id = 1
    for wide_id in wide_ids:
        if not keep_wide[int(wide_id)]:
            continue
        wide_map[int(wide_id)] = next_id
        next_id += 1
    kept_pixels = keep_wide[np.clip(wide_labels, 0, wide_max)]
    combined[kept_pixels] = wide_map[wide_labels[kept_pixels]]

    blocked = cv2.dilate((combined > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    remaining = ridge_labels.copy()
    remaining[blocked > 0] = 0
    ridge_ids = np.unique(remaining)
    ridge_ids = ridge_ids[ridge_ids > 0]
    ridge_map = np.zeros(ridge_max + 1, dtype=np.int32)
    for ridge_id in ridge_ids:
        ridge_map[int(ridge_id)] = next_id
        next_id += 1
    combined[remaining > 0] = ridge_map[remaining[remaining > 0]]
    return combined, kept_multi, dropped_sep
