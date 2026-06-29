from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def _to_binary(array: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    return (values >= threshold).astype(np.uint8)


def _extract_boundary(binary: np.ndarray, *, kernel_size: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    return np.clip(dilated.astype(np.int16) - eroded.astype(np.int16), 0, 1).astype(np.uint8)


def _dice_from_counts(tp: float, fp: float, fn: float) -> float:
    eps = 1e-6
    return float((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps))


def _iou_from_counts(tp: float, fp: float, fn: float) -> float:
    eps = 1e-6
    return float((tp + eps) / (tp + fp + fn + eps))


@dataclass
class SegmentationMetrics:
    dice: float = 0.0
    iou: float = 0.0
    boundary_iou: float = 0.0
    boundary_f1: float = 0.0
    hausdorff_distance: float = 0.0
    connected_component_difference: int = 0
    wire_break_count: int = 0
    false_bridge_count: int = 0
    confidence_histogram: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | int | dict[str, float]]:
        return {
            'dice': self.dice,
            'iou': self.iou,
            'boundary_iou': self.boundary_iou,
            'boundary_f1': self.boundary_f1,
            'hausdorff_distance': self.hausdorff_distance,
            'connected_component_difference': self.connected_component_difference,
            'wire_break_count': self.wire_break_count,
            'false_bridge_count': self.false_bridge_count,
            'confidence_histogram': dict(self.confidence_histogram),
        }


def _hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    pred_points = np.column_stack(np.where(pred > 0))
    target_points = np.column_stack(np.where(target > 0))
    if pred_points.size == 0 or target_points.size == 0:
        return float(max(pred.shape))

    def directed_hausdorff(source: np.ndarray, destination: np.ndarray) -> float:
        max_distance = 0.0
        for point in source:
            distances = np.sqrt(((destination - point) ** 2).sum(axis=1))
            max_distance = max(max_distance, float(distances.min()))
        return max_distance

    return max(directed_hausdorff(pred_points, target_points), directed_hausdorff(target_points, pred_points))


def _count_components(binary: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(binary.astype(np.uint8))
    return max(0, int(count) - 1)


def _skeleton_break_count(binary: np.ndarray) -> int:
    if not bool(binary.any()):
        return 0
    skeleton = cv2.ximgproc.thinning(binary.astype(np.uint8) * 255) if hasattr(cv2, 'ximgproc') else binary
    skeleton = (skeleton > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton, cv2.CV_8U, kernel)
    return int(((neighbors == 1) & (skeleton > 0)).sum())


def _false_bridge_count(pred: np.ndarray, target: np.ndarray) -> int:
    pred_only = np.clip(pred.astype(np.int16) - target.astype(np.int16), 0, 1).astype(np.uint8)
    return _count_components(pred_only)


def _confidence_histogram(confidence: np.ndarray | None) -> dict[str, float]:
    if confidence is None:
        return {}
    values = np.asarray(confidence, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {}
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    histogram: dict[str, float] = {}
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (values >= low) & (values < high if high < 1.0 else values <= high)
        histogram[f'{low:.1f}-{high:.1f}'] = float(mask.mean())
    return histogram


def compute_segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float = 0.5,
    confidence: np.ndarray | None = None,
) -> SegmentationMetrics:
    pred_bin = _to_binary(prediction, threshold=threshold)
    target_bin = _to_binary(target, threshold=threshold)
    tp = float((pred_bin & target_bin).sum())
    fp = float((pred_bin & (1 - target_bin)).sum())
    fn = float(((1 - pred_bin) & target_bin).sum())

    pred_boundary = _extract_boundary(pred_bin)
    target_boundary = _extract_boundary(target_bin)
    b_tp = float((pred_boundary & target_boundary).sum())
    b_fp = float((pred_boundary & (1 - target_boundary)).sum())
    b_fn = float(((1 - pred_boundary) & target_boundary).sum())
    boundary_precision = (b_tp + 1e-6) / (b_tp + b_fp + 1e-6)
    boundary_recall = (b_tp + 1e-6) / (b_tp + b_fn + 1e-6)
    boundary_f1 = (2.0 * boundary_precision * boundary_recall + 1e-6) / (
        boundary_precision + boundary_recall + 1e-6
    )

    return SegmentationMetrics(
        dice=_dice_from_counts(tp, fp, fn),
        iou=_iou_from_counts(tp, fp, fn),
        boundary_iou=_iou_from_counts(b_tp, b_fp, b_fn),
        boundary_f1=float(boundary_f1),
        hausdorff_distance=_hausdorff_distance(pred_bin, target_bin),
        connected_component_difference=abs(_count_components(pred_bin) - _count_components(target_bin)),
        wire_break_count=_skeleton_break_count(target_bin) - _skeleton_break_count(pred_bin & target_bin),
        false_bridge_count=_false_bridge_count(pred_bin, target_bin),
        confidence_histogram=_confidence_histogram(confidence),
    )
