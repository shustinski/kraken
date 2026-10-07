"""Validation metrics for the cell-defect analysis. Qt-free.

Pixel accuracy and Dice hide what matters here, so the measures are object-level:
per-class precision/recall/F1 on matched objects, false alarms per frame, Mask and
Boundary IoU for shape, Rand and information scores split into split and merge parts
(foreground-restricted, as in the SNEMI challenge), Brier score and calibration error
of the anomaly score. Frames are scored one by one and then averaged, so one dense
frame does not outweigh many sparse ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

from .error_classes import ERROR_CLASS_NAMES, OK


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    union = int(np.count_nonzero(a | b))
    return 1.0 if union == 0 else float(np.count_nonzero(a & b)) / float(union)


def _inner_boundary(mask: np.ndarray, width: int) -> np.ndarray:
    """Mask pixels within ``width`` of the contour (Cheng et al.: G_d ∩ G)."""

    if not mask.any():
        return mask
    padded = np.pad(mask.astype(np.uint8), int(width) + 1)
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(width) + 1, 2 * int(width) + 1))
        eroded = cv2.erode(padded, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    else:  # pragma: no cover - OpenCV is a hard dependency of the app
        from scipy.ndimage import binary_erosion

        eroded = binary_erosion(padded, iterations=int(width)).astype(np.uint8)
    pad = int(width) + 1
    return mask & ~eroded[pad:-pad, pad:-pad].astype(bool)


def boundary_iou(first: np.ndarray, second: np.ndarray, *, width: int = 2) -> float:
    """IoU of the two masks' boundary bands; reacts to contour errors that Mask IoU hides on small objects."""

    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    band_a = _inner_boundary(a, max(1, int(width)))
    band_b = _inner_boundary(b, max(1, int(width)))
    return mask_iou(band_a, band_b)


def _foreground_contingency(
    prediction_labels: np.ndarray, truth_labels: np.ndarray, ignore: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Joint counts on truth-foreground pixels: (pair counts, prediction-segment ids of the pairs, singletons, total).

    Prediction background inside the truth foreground counts as one-pixel segments.
    """

    region = np.asarray(truth_labels) > 0
    if ignore is not None:
        region &= ~np.asarray(ignore, dtype=bool)
    predicted = np.asarray(prediction_labels)[region].astype(np.int64)
    truth = np.asarray(truth_labels)[region].astype(np.int64)
    total = int(predicted.size)
    singletons = int(np.count_nonzero(predicted == 0))
    keep = predicted > 0
    if not np.any(keep):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), singletons, total
    stride = int(truth.max()) + 1
    pairs, counts = np.unique(predicted[keep] * stride + truth[keep], return_counts=True)
    return counts.astype(np.int64), (pairs // stride).astype(np.int64), singletons, total


@dataclass(frozen=True, slots=True)
class SplitMergeScores:
    """1.0 is perfect. ``*_split`` falls with split errors, ``*_merge`` with merge errors."""

    rand_split: float
    rand_merge: float
    rand_f: float
    info_split: float
    info_merge: float
    info_f: float
    vi_split: float  # H(prediction | truth), nats
    vi_merge: float  # H(truth | prediction), nats


def split_merge_scores(
    prediction_labels: np.ndarray, truth_labels: np.ndarray, *, ignore: np.ndarray | None = None
) -> SplitMergeScores:
    pair_counts, pair_segments, singletons, total = _foreground_contingency(prediction_labels, truth_labels, ignore)
    if total == 0:
        return SplitMergeScores(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    n = float(total)
    region = np.asarray(truth_labels) > 0
    if ignore is not None:
        region &= ~np.asarray(ignore, dtype=bool)
    truth_sizes = np.bincount(np.asarray(truth_labels)[region].astype(np.int64))
    truth_sizes = truth_sizes[truth_sizes > 0].astype(np.float64)
    prediction_sizes = (
        np.bincount(pair_segments, weights=pair_counts.astype(np.float64)) if pair_segments.size else np.zeros(0)
    )
    prediction_sizes = prediction_sizes[prediction_sizes > 0]

    joint_square = (float(np.sum(pair_counts.astype(np.float64) ** 2)) + singletons) / n**2
    prediction_square = (float(np.sum(prediction_sizes**2)) + singletons) / n**2
    truth_square = float(np.sum(truth_sizes**2)) / n**2
    rand_split = joint_square / truth_square if truth_square > 0 else 1.0
    rand_merge = joint_square / prediction_square if prediction_square > 0 else 1.0

    def entropy(sizes: np.ndarray, extra_singletons: int = 0) -> float:
        p = sizes / n
        value = -float(np.sum(p * np.log(p))) if p.size else 0.0
        if extra_singletons:
            value += extra_singletons * (math.log(n) / n)
        return value

    h_joint = entropy(pair_counts.astype(np.float64), singletons)
    h_prediction = entropy(prediction_sizes, singletons)
    h_truth = entropy(truth_sizes)
    mutual = max(0.0, h_prediction + h_truth - h_joint)
    info_split = mutual / h_prediction if h_prediction > 0 else 1.0
    info_merge = mutual / h_truth if h_truth > 0 else 1.0

    def harmonic(a: float, b: float) -> float:
        return 0.0 if a + b <= 0 else 2.0 * a * b / (a + b)

    return SplitMergeScores(
        rand_split=float(rand_split),
        rand_merge=float(rand_merge),
        rand_f=float(harmonic(rand_split, rand_merge)),
        info_split=float(info_split),
        info_merge=float(info_merge),
        info_f=float(harmonic(info_split, info_merge)),
        vi_split=float(max(0.0, h_joint - h_truth)),
        vi_merge=float(max(0.0, h_joint - h_prediction)),
    )


@dataclass(slots=True)
class ObjectConfusion:
    """Expected (rows) against predicted (columns) class of each prediction object."""

    classes: tuple[int, ...]
    counts: dict[tuple[int, int], int] = field(default_factory=dict)
    frames: int = 0

    def add(self, expected: Iterable[int], predicted: Iterable[int]) -> None:
        for want, got in zip(expected, predicted):
            key = (int(want), int(got))
            self.counts[key] = self.counts.get(key, 0) + 1
        self.frames += 1

    def matrix(self) -> np.ndarray:
        index = {value: position for position, value in enumerate(self.classes)}
        table = np.zeros((len(self.classes), len(self.classes)), dtype=np.int64)
        for (want, got), count in self.counts.items():
            if want in index and got in index:
                table[index[want], index[got]] += count
        return table

    def per_class(self) -> dict[str, dict[str, float]]:
        table = self.matrix()
        scores: dict[str, dict[str, float]] = {}
        for position, value in enumerate(self.classes):
            if value == OK:
                continue
            tp = float(table[position, position])
            fp = float(table[:, position].sum() - tp)
            fn = float(table[position, :].sum() - tp)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            scores[ERROR_CLASS_NAMES.get(value, str(value))] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn,
            }
        return scores

    def macro_f1(self) -> float:
        values = [item["f1"] for item in self.per_class().values() if item["support"] > 0]
        return float(np.mean(values)) if values else 0.0

    def detection(self) -> dict[str, float]:
        """Error versus OK, whatever the class: the question «is this object suspicious»."""

        tp = fp = fn = 0
        for (want, got), count in self.counts.items():
            if want != OK and got != OK:
                tp += count
            elif want == OK and got != OK:
                fp += count
            elif want != OK and got == OK:
                fn += count
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
            "false_alarms_per_frame": float(fp) / float(max(1, self.frames)),
        }


def brier_score(scores: Sequence[float], is_error: Sequence[bool]) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(is_error, dtype=np.float64)
    return float(np.mean((s - y) ** 2)) if s.size else 0.0


def expected_calibration_error(scores: Sequence[float], is_error: Sequence[bool], *, bins: int = 10) -> float:
    s = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(is_error, dtype=np.float64)
    if s.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    index = np.clip(np.digitize(s, edges[1:-1]), 0, int(bins) - 1)
    total = 0.0
    for bin_id in range(int(bins)):
        inside = index == bin_id
        if np.any(inside):
            total += float(np.count_nonzero(inside)) * abs(float(s[inside].mean()) - float(y[inside].mean()))
    return total / float(s.size)


def mean_over_frames(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average of per-frame values, each key over the frames that have it."""

    keys = sorted({key for row in rows for key in row})
    result: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
        if values:
            result[key] = float(np.mean(values))
    return result
