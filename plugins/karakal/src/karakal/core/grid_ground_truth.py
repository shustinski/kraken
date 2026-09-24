"""Match a network mask to an optional binary ground-truth mask. Qt-free."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .grid_calibration import decide_reasons, example_detector_scores, fit_sliders_from_examples
from .grid_scoring import slider_threshold

IOU_NORMAL = 0.55
FILLED_EXTRA = 0.45
PARTIAL_EXTRA = 0.15
COVER_FRACTION = 0.40
HELD_OUT_DROP = 0.25


def _binary(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[:, :, 0]
    return array > 0


def _labels(mask: np.ndarray) -> np.ndarray:
    binary = _binary(mask).astype(np.uint8)
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    if count <= 1:
        return np.zeros(binary.shape, dtype=np.int32)
    return labels.astype(np.int32, copy=False)


def _intersection_matrix(network: np.ndarray, truth: np.ndarray) -> np.ndarray:
    truth_count = int(truth.max()) + 1
    pair = network.astype(np.int64).ravel() * truth_count + truth.astype(np.int64).ravel()
    counts = np.bincount(pair, minlength=(int(network.max()) + 1) * truth_count)
    return counts.reshape((-1, truth_count))


def _bboxes(labels: np.ndarray) -> list[tuple[int, int, int, int]]:
    boxes = [(0, 0, 0, 0)]
    for component_id in range(1, int(labels.max()) + 1):
        ys, xs = np.nonzero(labels == component_id)
        if ys.size == 0:
            boxes.append((0, 0, 0, 0))
        else:
            boxes.append((int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))
    return boxes


def _box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter) / float(union) if union else 0.0


def _component_shape(labels: np.ndarray, component_id: int) -> dict[str, float]:
    ys, xs = np.nonzero(labels == component_id)
    if ys.size == 0:
        return {"width": 1.0, "height": 1.0, "area": 1.0, "solidity": 1.0, "extent": 1.0, "interior_fill": 0.0}
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    area = float(ys.size)
    extent = area / max(1.0, width * height)
    component = np.zeros(labels.shape, dtype=np.uint8)
    component[ys, xs] = 255
    contours, _hierarchy = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 1.0 else 1.0
    else:
        solidity = 1.0
    return {
        "width": width,
        "height": height,
        "area": area,
        "solidity": float(solidity),
        "extent": float(extent),
        "interior_fill": float(extent),
    }


def _relative_features(shapes: Sequence[dict[str, float]], labels: Sequence[str]) -> list[dict[str, float]]:
    normals = [shape for shape, label in zip(shapes, labels) if label == "normal"] or list(shapes)
    reference_width = float(np.median([item["width"] for item in normals]))
    reference_height = float(np.median([item["height"] for item in normals]))
    reference_area = float(np.median([item["area"] for item in normals]))
    reference_solidity = float(np.median([item["solidity"] for item in normals]))
    reference_extent = float(np.median([item["extent"] for item in normals]))
    reference_interior = float(np.median([item["interior_fill"] for item in normals]))
    features = []
    for shape in shapes:
        features.append(
            {
                "width_ratio": float(shape["width"]) / max(1.0, reference_width),
                "height_ratio": float(shape["height"]) / max(1.0, reference_height),
                "area_ratio": float(shape["area"]) / max(1.0, reference_area),
                "interior_fill": float(shape["interior_fill"]),
                "reference_interior": reference_interior,
                "solidity": float(shape["solidity"]),
                "reference_solidity": reference_solidity,
                "extent": float(shape["extent"]),
                "reference_extent": reference_extent,
            }
        )
    return features


def match_masks(network: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """Label each network component. A truth cell with no partner is only a miss count."""

    if _binary(network).shape != _binary(truth).shape:
        raise ValueError("ground-truth mask size does not match the network mask")
    network_labels = _labels(network)
    truth_labels = _labels(truth)
    if int(network_labels.max()) == 0:
        missed = int(truth_labels.max())
        return {"components": [], "missed": missed, "examples": []}
    matrix = _intersection_matrix(network_labels, truth_labels)
    network_area = matrix.sum(axis=1)
    truth_area = matrix.sum(axis=0)
    network_boxes = _bboxes(network_labels)
    truth_boxes = _bboxes(truth_labels)
    labels: list[str] = []
    shapes: list[dict[str, float]] = []
    paired_truth: set[int] = set()
    for component_id in range(1, matrix.shape[0]):
        intersections = matrix[component_id, 1:]
        area = float(network_area[component_id])
        box_ious = np.array(
            [_box_iou(network_boxes[component_id], truth_boxes[truth_id]) for truth_id in range(1, len(truth_boxes))],
            dtype=np.float64,
        )
        if intersections.size == 0 or area <= 0.0:
            labels.append("small_artifact")
            shapes.append(_component_shape(network_labels, component_id))
            continue
        truth_ids = np.arange(1, matrix.shape[1])
        covered = [
            int(truth_id)
            for truth_id, overlap, box_iou in zip(truth_ids, intersections, box_ious)
            if float(truth_area[truth_id]) > 0.0
            and (
                float(overlap) / float(truth_area[truth_id]) >= COVER_FRACTION
                or float(box_iou) >= IOU_NORMAL
            )
        ]
        if len(covered) >= 2:
            labels.append("merged_contour")
            paired_truth.update(covered)
            shapes.append(_component_shape(network_labels, component_id))
            continue
        unions = network_area[component_id] + truth_area[1:] - intersections
        ious = np.divide(intersections, np.maximum(unions, 1), dtype=np.float64)
        best = int(np.argmax(np.maximum(ious, box_ious))) if ious.size else 0
        pixel_iou = float(ious[best]) if ious.size else 0.0
        box_iou = float(box_ious[best]) if box_ious.size else 0.0
        overlap = float(intersections[best]) if intersections.size else 0.0
        extra = (area - overlap) / area if max(pixel_iou, box_iou) > 0.0 else 1.0
        if max(pixel_iou, box_iou) <= 0.05:
            labels.append("small_artifact")
        elif extra >= FILLED_EXTRA and box_iou >= 0.2:
            labels.append("filled_cell")
            paired_truth.add(int(best) + 1)
        elif extra >= PARTIAL_EXTRA and box_iou >= 0.2:
            labels.append("partial_filled_cell")
            paired_truth.add(int(best) + 1)
        elif pixel_iou >= IOU_NORMAL:
            labels.append("normal")
            paired_truth.add(int(best) + 1)
        else:
            labels.append("broken_geometry")
            paired_truth.add(int(best) + 1)
        shapes.append(_component_shape(network_labels, component_id))
    missed = sum(1 for truth_id in range(1, matrix.shape[1]) if truth_id not in paired_truth and truth_area[truth_id] > 0)
    features = _relative_features(shapes, labels)
    components = [
        {"label": label, "features": feature}
        for label, feature in zip(labels, features)
    ]
    return {"components": components, "missed": int(missed), "examples": components}


def fit_markup(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit sliders on labeled cells. Warn when a left-out frame loses a lot of F1."""

    rows = [dict(component) for frame in frames for component in frame.get("components", ())]
    if not rows:
        return {"sliders": {}, "f1": 0.0, "held_out_f1": 0.0, "warning": "", "metrics": markup_metrics((), {})}
    sliders = fit_sliders_from_examples(rows)
    in_sample = detection_f1(rows, sliders)
    held_out_scores = []
    if len(frames) >= 2:
        for index in range(len(frames)):
            train = [dict(component) for other, frame in enumerate(frames) if other != index for component in frame.get("components", ())]
            probe = [dict(component) for component in frames[index].get("components", ())]
            trained = fit_sliders_from_examples(train)
            held_out_scores.append(detection_f1(probe, trained or sliders))
    held_out = float(np.mean(held_out_scores)) if held_out_scores else in_sample
    warning = "grid_tuning.markup_overfit" if in_sample - held_out > HELD_OUT_DROP else ""
    return {
        "sliders": sliders,
        "f1": in_sample,
        "held_out_f1": held_out,
        "warning": warning,
        "metrics": markup_metrics(rows, sliders),
    }


def detection_f1(rows: Sequence[Mapping[str, Any]], sliders: Mapping[str, int]) -> float:
    if not rows:
        return 0.0
    true_positive = false_positive = false_negative = 0
    for row in rows:
        label = str(row.get("label") or "")
        expected = label not in {"normal", "good", "miss"}
        predicted = bool(_predicted_reasons(row, sliders))
        if expected and predicted and (label in _predicted_reasons(row, sliders) or label == "partial_filled_cell"):
            true_positive += 1
        elif expected and not predicted:
            false_negative += 1
        elif not expected and predicted:
            false_positive += 1
        elif expected and predicted:
            false_positive += 1
            false_negative += 1
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def markup_metrics(rows: Sequence[Mapping[str, Any]], sliders: Mapping[str, int]) -> dict[str, int]:
    errors = normals = found = false = 0
    for row in rows:
        label = str(row.get("label") or "")
        reasons = _predicted_reasons(row, sliders)
        if label in {"normal", "good"}:
            normals += 1
            if reasons:
                false += 1
        elif label not in {"miss", ""}:
            errors += 1
            if label in reasons or (label == "partial_filled_cell" and "filled_cell" in reasons):
                found += 1
    return {"found": found, "errors": errors, "false": false, "normals": normals}


def _predicted_reasons(row: Mapping[str, Any], sliders: Mapping[str, int]) -> tuple[str, ...]:
    features = row.get("features") if isinstance(row.get("features"), Mapping) else {}
    scores = example_detector_scores(features or {})
    thresholds = {
        "fill": slider_threshold(int(sliders.get("fill_sensitivity", 60))),
        "geometry": slider_threshold(int(sliders.get("geometry_sensitivity", 40))),
        "merge": slider_threshold(int(sliders.get("merge_sensitivity", 35))),
        "debris": slider_threshold(int(sliders.get("debris_sensitivity", 75))),
    }
    return decide_reasons(scores, thresholds, features=features or {})
