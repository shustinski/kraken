from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.metrics import contingency_table


def _to_binary(array: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(array, dtype=np.float32) >= threshold).astype(np.uint8)


def _extract_boundary(binary: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    return (binary - cv2.erode(binary, kernel, iterations=1)).astype(np.uint8)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 1.0 if denominator <= 0.0 else float(numerator / denominator)


def _dice_from_counts(tp: float, fp: float, fn: float) -> float:
    return _safe_ratio(2.0 * tp, 2.0 * tp + fp + fn)


def _iou_from_counts(tp: float, fp: float, fn: float) -> float:
    return _safe_ratio(tp, tp + fp + fn)


@dataclass
class SegmentationMetrics:
    dice: float = 0.0
    iou: float = 0.0
    boundary_iou: float = 0.0
    boundary_f1: float = 0.0
    hausdorff_distance: float = 0.0
    hausdorff_95: float | None = None
    connected_component_difference: int = 0
    wire_break_count: int = 0
    false_bridge_count: int = 0
    missed_component_count: int = 0
    spurious_component_count: int = 0
    foreground_component_delta: int = 0
    background_component_delta: int = 0
    topology_violation_count: int = 0
    brier_score: float | None = None
    expected_calibration_error: float | None = None
    error_detection_aurc: float | None = None
    confidence_histogram: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | int | None | dict[str, int]]:
        return dict(vars(self))


def _surface_distances(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    points = source > 0
    if not bool(points.any()):
        return np.empty((0,), dtype=np.float32)
    if not bool((destination > 0).any()):
        return np.full(int(points.sum()), float(np.hypot(*source.shape)), dtype=np.float32)
    distance = distance_transform_edt(destination == 0)
    return distance[points].astype(np.float32, copy=False)


def _hausdorff(boundary_a: np.ndarray, boundary_b: np.ndarray) -> tuple[float, float]:
    if not bool(boundary_a.any()) and not bool(boundary_b.any()):
        return 0.0, 0.0
    distances = np.concatenate(
        (_surface_distances(boundary_a, boundary_b), _surface_distances(boundary_b, boundary_a))
    )
    return float(distances.max(initial=0.0)), float(np.percentile(distances, 95))


def _component_labels(binary: np.ndarray) -> tuple[int, np.ndarray]:
    count, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    return max(0, int(count) - 1), labels


def _topology_counts(pred: np.ndarray, target: np.ndarray) -> dict[str, int]:
    pred_count, pred_labels = _component_labels(pred)
    target_count, target_labels = _component_labels(target)
    overlaps = contingency_table(target_labels, pred_labels, normalize=False).tocsr()
    foreground_overlaps = overlaps[1:, 1:]
    target_overlap_counts = np.diff(foreground_overlaps.indptr)
    pred_overlap_counts = np.diff(foreground_overlaps.tocsc().indptr)
    wire_breaks = int(np.maximum(target_overlap_counts - 1, 0).sum())
    missed = int(np.count_nonzero(target_overlap_counts == 0))
    false_bridges = int(np.maximum(pred_overlap_counts - 1, 0).sum())
    spurious = int(np.count_nonzero(pred_overlap_counts == 0))

    background_pred, _ = _component_labels(1 - pred)
    background_target, _ = _component_labels(1 - target)
    return {
        'wire_break_count': wire_breaks,
        'false_bridge_count': false_bridges,
        'missed_component_count': missed,
        'spurious_component_count': spurious,
        'foreground_component_delta': pred_count - target_count,
        'background_component_delta': background_pred - background_target,
    }


def _boundary_metrics(pred: np.ndarray, target: np.ndarray, tolerance: int) -> tuple[float, float, np.ndarray, np.ndarray]:
    pred_boundary = _extract_boundary(pred)
    target_boundary = _extract_boundary(target)
    radius = max(0, int(tolerance))
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    pred_dilated = cv2.dilate(pred_boundary, kernel)
    target_dilated = cv2.dilate(target_boundary, kernel)
    precision = _safe_ratio(float((pred_boundary & target_dilated).sum()), float(pred_boundary.sum()))
    recall = _safe_ratio(float((target_boundary & pred_dilated).sum()), float(target_boundary.sum()))
    boundary_f1 = _safe_ratio(2.0 * precision * recall, precision + recall)

    pred_band = pred & pred_dilated
    target_band = target & target_dilated
    intersection = float((pred_band & target_band).sum())
    union = float((pred_band | target_band).sum())
    return _safe_ratio(intersection, union), boundary_f1, pred_boundary, target_boundary


def _confidence_metrics(
    confidence: np.ndarray | None,
    correctness: np.ndarray,
    *,
    bins: int = 10,
) -> tuple[dict[str, int], float | None, float | None, float | None]:
    if confidence is None:
        return {}, None, None, None
    values = np.clip(np.asarray(confidence, dtype=np.float32).reshape(-1), 0.0, 1.0)
    correct = correctness.astype(np.float32, copy=False).reshape(-1)
    if values.size != correct.size:
        raise ValueError('Confidence map must have the same number of pixels as prediction.')
    edges = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    histogram: dict[str, int] = {}
    ece = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (values >= low) & (values < high if index < len(edges) - 2 else values <= high)
        count = int(selected.sum())
        histogram[f'{low:.1f}-{high:.1f}'] = count
        if count:
            ece += (count / values.size) * abs(float(values[selected].mean()) - float(correct[selected].mean()))
    brier = float(np.mean((values - correct) ** 2))
    order = np.argsort(-values, kind='stable')
    errors = 1.0 - correct[order]
    cumulative_risk = np.cumsum(errors) / np.arange(1, errors.size + 1)
    aurc = float(cumulative_risk.mean()) if cumulative_risk.size else 0.0
    return histogram, brier, float(ece), aurc


def compute_segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float = 0.5,
    confidence: np.ndarray | None = None,
    boundary_tolerance: int = 2,
    include_hd95: bool = True,
    confidence_bins: int = 10,
) -> SegmentationMetrics:
    pred_bin = _to_binary(prediction, threshold=threshold)
    target_bin = _to_binary(target, threshold=threshold)
    if pred_bin.shape != target_bin.shape:
        raise ValueError(f'Prediction and target shapes differ: {pred_bin.shape} != {target_bin.shape}.')
    tp = float((pred_bin & target_bin).sum())
    fp = float((pred_bin & (1 - target_bin)).sum())
    fn = float(((1 - pred_bin) & target_bin).sum())
    boundary_iou, boundary_f1, pred_boundary, target_boundary = _boundary_metrics(
        pred_bin, target_bin, boundary_tolerance
    )
    hausdorff, hd95 = _hausdorff(pred_boundary, target_boundary)
    topology = _topology_counts(pred_bin, target_bin)
    histogram, brier, ece, aurc = _confidence_metrics(
        confidence,
        pred_bin == target_bin,
        bins=max(2, int(confidence_bins)),
    )
    pred_count, _ = _component_labels(pred_bin)
    target_count, _ = _component_labels(target_bin)
    topology_violations = sum(
        topology[name]
        for name in ('wire_break_count', 'false_bridge_count', 'missed_component_count', 'spurious_component_count')
    )
    return SegmentationMetrics(
        dice=_dice_from_counts(tp, fp, fn),
        iou=_iou_from_counts(tp, fp, fn),
        boundary_iou=boundary_iou,
        boundary_f1=boundary_f1,
        hausdorff_distance=hausdorff,
        hausdorff_95=hd95 if include_hd95 else None,
        connected_component_difference=abs(pred_count - target_count),
        topology_violation_count=topology_violations,
        confidence_histogram=histogram,
        brier_score=brier,
        expected_calibration_error=ece,
        error_detection_aurc=aurc,
        **topology,
    )
