"""Optional confidence and source-frame hints. Never change mask-defect scores."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def cell_low_confidence(
    mean_confidence: float | None,
    uncertain_pixel_ratio: float | None,
    *,
    threshold: float = 0.55,
) -> bool:
    if mean_confidence is None:
        return False
    uncertain = 0.0 if uncertain_pixel_ratio is None else float(uncertain_pixel_ratio)
    return float(mean_confidence) < float(threshold) or uncertain >= 0.35


def possible_missed_regions(
    probability: np.ndarray | None,
    mask: np.ndarray | None,
    *,
    high: float = 0.55,
    mask_threshold: float = 0.5,
    min_area: int = 16,
) -> list[tuple[int, int, int, int]]:
    """Regions where the network is almost sure but the binary mask is empty."""

    if probability is None or mask is None:
        return []
    prob = np.asarray(probability, dtype=np.float32)
    binary = np.asarray(mask)
    if binary.ndim == 3:
        binary = binary[:, :, 0]
    if prob.ndim == 3:
        prob = prob[:, :, 0]
    if prob.shape != binary.shape:
        return []
    if float(np.nanmax(prob)) > 1.5:
        prob = prob / 255.0
    candidate = (prob >= float(high)) & (binary <= float(mask_threshold) * (255.0 if binary.dtype != np.bool_ else 1.0))
    if not np.any(candidate):
        return []
    from cv2 import connectedComponentsWithStats

    count, _labels, stats, _centroids = connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    boxes = []
    for index in range(1, count):
        area = int(stats[index, 4])
        if area < int(min_area):
            continue
        x, y, width, height = (int(stats[index, axis]) for axis in range(4))
        boxes.append((x, y, width, height))
    return boxes


def source_cell_rule(source: np.ndarray, truth_mask: np.ndarray) -> dict[str, float]:
    """Brightness/contrast rule from marked cells on the grayscale photo."""

    gray = np.asarray(source, dtype=np.float32)
    truth = np.asarray(truth_mask) > 0
    if gray.ndim == 3:
        gray = gray[:, :, 0]
    if gray.shape != truth.shape or not np.any(truth) or not np.any(~truth):
        return {"mean": 0.0, "std": 0.0, "bg_mean": 0.0, "bg_std": 1.0, "usable": 0.0}
    cells = gray[truth]
    background = gray[~truth]
    return {
        "mean": float(np.mean(cells)),
        "std": float(max(np.std(cells), 1.0)),
        "bg_mean": float(np.mean(background)),
        "bg_std": float(max(np.std(background), 1.0)),
        "usable": 1.0,
    }


def source_mask_mismatches(
    source: np.ndarray,
    network_mask: np.ndarray,
    rule: Mapping[str, float],
    *,
    min_area: int = 25,
) -> list[dict[str, Any]]:
    """Find places that look like cells on the photo but are empty in the mask, and vice versa."""

    if float(rule.get("usable", 0.0)) < 1.0:
        return []
    gray = np.asarray(source, dtype=np.float32)
    mask = np.asarray(network_mask)
    if gray.ndim == 3:
        gray = gray[:, :, 0]
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if gray.shape != mask.shape:
        return []
    binary = mask > 0
    cell_like = np.abs(gray - float(rule["mean"])) <= 1.5 * float(rule["std"])
    photo_but_empty = cell_like & ~binary
    mask_but_unlike = binary & (np.abs(gray - float(rule["mean"])) > 2.5 * float(rule["std"]))
    from cv2 import connectedComponentsWithStats

    findings: list[dict[str, Any]] = []
    for array, label in ((photo_but_empty, "source_mask_mismatch"), (mask_but_unlike, "source_mask_mismatch")):
        count, _labels, stats, _centroids = connectedComponentsWithStats(array.astype(np.uint8), connectivity=8)
        for index in range(1, count):
            if int(stats[index, 4]) < int(min_area):
                continue
            x, y, width, height = (int(stats[index, axis]) for axis in range(4))
            findings.append({"label": label, "bbox": (x, y, width, height)})
    return findings


def frame_suspicion(cells: Sequence[Any]) -> dict[str, float]:
    scores = [float(getattr(cell, "score", 0.0) or 0.0) for cell in cells if getattr(cell, "reasons", ())]
    if not scores:
        return {"max": 0.0, "count": 0.0}
    return {"max": float(max(scores)), "count": float(len(scores))}
