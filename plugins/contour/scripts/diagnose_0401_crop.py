"""Diagnose why frame 0401 is exact full-frame but has 35 false components after crop=50."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PLUGIN_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from contour.vision.metal_recovery import detect_metalization  # noqa: E402
from contour.vision.metal_recovery.gradient_watershed import GradientWatershedConfig  # noqa: E402
from scripts.benchmark_metal_segmentation import (  # noqa: E402
    REAL_DATASET_ROOT,
    _rasterize_detected_polygons,
    _ui_recovery_config,
    build_real_benchmark_cases,
    crop_evaluation_region,
    measure_segmentation,
    prepare_evaluation_masks,
)

CROP_PX = 50
FRAME_ID = "0401"
OUTPUT_PATH = PLUGIN_ROOT / "benchmarks" / "diagnose_0401_crop.json"


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _fragment_count(mask: np.ndarray) -> int:
    count, _labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return int(count - 1)


def _roi_mask(shape: tuple[int, int], crop_px: int) -> np.ndarray:
    inside = np.zeros(shape, dtype=bool)
    inside[crop_px : shape[0] - crop_px, crop_px : shape[1] - crop_px] = True
    return inside


def main() -> int:
    cases = {case.name: case for case in build_real_benchmark_cases(REAL_DATASET_ROOT)}
    case = cases[FRAME_ID]
    recovery = _ui_recovery_config("auto", GradientWatershedConfig())
    detection = detect_metalization(case.image, recovery, source_image=case.source_image)
    pred_labels_full = _rasterize_detected_polygons(detection.accepted, case.image.shape[:2])
    pred_mask_full = np.where(pred_labels_full > 0, 255, 0).astype(np.uint8)
    gt_full = case.labels
    roi = _roi_mask(gt_full.shape, CROP_PX)

    full_metrics = measure_segmentation(
        pred_mask_full,
        gt_full,
        elapsed_ms=0.0,
        predicted_labels=pred_labels_full,
    )
    pred_eval, gt_eval, pred_labels_eval = prepare_evaluation_masks(
        pred_mask_full,
        gt_full,
        crop_px=CROP_PX,
        predicted_labels=pred_labels_full,
        frame_id=FRAME_ID,
    )
    crop_metrics = measure_segmentation(
        pred_eval,
        gt_eval,
        elapsed_ms=0.0,
        predicted_labels=pred_labels_eval,
    )

    gt_ids_full = set(np.unique(gt_full[gt_full > 0]).tolist())
    pred_ids_full = set(np.unique(pred_labels_full[pred_labels_full > 0]).tolist())
    gt_ids_roi = set(np.unique(gt_full[roi][gt_full[roi] > 0]).tolist())
    pred_ids_roi = set(np.unique(pred_labels_full[roi][pred_labels_full[roi] > 0]).tolist())
    gt_removed_ids = sorted(gt_ids_full - gt_ids_roi)
    pred_removed_ids = sorted(pred_ids_full - pred_ids_roi)

    cropped_gt_keep_ids = crop_evaluation_region(gt_full, CROP_PX)
    cropped_pred_keep_ids = crop_evaluation_region(pred_labels_full, CROP_PX)
    keep_id_gt_count = int(np.unique(cropped_gt_keep_ids[cropped_gt_keep_ids > 0]).size)
    keep_id_pred_count = int(np.unique(cropped_pred_keep_ids[cropped_pred_keep_ids > 0]).size)
    binary_gt_count = int(np.unique(gt_eval[gt_eval > 0]).size)
    binary_pred_count = int(np.unique(pred_labels_eval[pred_labels_eval > 0]).size)

    per_id_pred_fragments = 0
    split_pred_ids: list[dict[str, int]] = []
    for pred_id in sorted(pred_ids_roi):
        fragments = _fragment_count(cropped_pred_keep_ids == pred_id)
        per_id_pred_fragments += fragments
        if fragments > 1:
            split_pred_ids.append({"id": int(pred_id), "fragments_after_crop": fragments})

    per_id_gt_fragments = 0
    split_gt_ids: list[dict[str, int]] = []
    for gt_id in sorted(gt_ids_roi):
        fragments = _fragment_count(cropped_gt_keep_ids == gt_id)
        per_id_gt_fragments += fragments
        if fragments > 1:
            split_gt_ids.append({"id": int(gt_id), "fragments_after_crop": fragments})

    keep_id_metrics = measure_segmentation(
        np.where(cropped_pred_keep_ids > 0, 255, 0).astype(np.uint8),
        cropped_gt_keep_ids,
        elapsed_ms=0.0,
        predicted_labels=cropped_pred_keep_ids,
    )

    false_pred_ids: list[int] = []
    for pred_id in np.unique(pred_labels_eval[pred_labels_eval > 0]):
        overlap = gt_eval[pred_labels_eval == pred_id]
        prediction_area = max(1, int(overlap.size))
        if np.count_nonzero(overlap > 0) / prediction_area < 0.10:
            false_pred_ids.append(int(pred_id))

    false_reports: list[dict[str, object]] = []
    for crop_pred_id in false_pred_ids:
        crop_mask = pred_labels_eval == crop_pred_id
        full_mask = np.zeros(gt_full.shape, dtype=bool)
        full_mask[CROP_PX : gt_full.shape[0] - CROP_PX, CROP_PX : gt_full.shape[1] - CROP_PX] = crop_mask
        source_pred_ids = [int(value) for value in np.unique(pred_labels_full[full_mask]) if value > 0]
        source_gt_ids = [int(value) for value in np.unique(gt_full[full_mask]) if value > 0]
        parent_summaries: list[dict[str, object]] = []
        for pred_id in source_pred_ids:
            pred_full = pred_labels_full == pred_id
            overlapping_gt = [int(value) for value in np.unique(gt_full[pred_full]) if value > 0]
            gt_union = np.zeros(gt_full.shape, dtype=bool)
            for gt_id in overlapping_gt:
                gt_union |= gt_full == gt_id
            parent_full_fragments = _fragment_count(pred_full)
            parent_roi_fragments = _fragment_count(pred_full & roi)
            parent_summaries.append(
                {
                    "full_frame_pred_id": pred_id,
                    "full_frame_pred_bbox": _bbox(pred_full),
                    "full_frame_pred_area": _area(pred_full),
                    "pred_area_in_roi": _area(pred_full & roi),
                    "pred_area_in_border": _area(pred_full & ~roi),
                    "full_frame_fragment_count": parent_full_fragments,
                    "roi_fragment_count": parent_roi_fragments,
                    "crop_created_split": parent_full_fragments == 1 and parent_roi_fragments > 1,
                    "already_disconnected_full_frame": parent_full_fragments > 1,
                    "overlapping_gt_ids": overlapping_gt,
                    "gt_area_in_roi": _area(gt_union & roi),
                    "gt_area_in_border": _area(gt_union & ~roi),
                    "gt_fully_removed_by_crop": bool(overlapping_gt)
                    and all(gt_id not in gt_ids_roi for gt_id in overlapping_gt),
                    "spatial_overlap_in_roi": _area(pred_full & gt_union & roi) > 0,
                }
            )
        gt_union_all = np.zeros(gt_full.shape, dtype=bool)
        for gt_id in source_gt_ids:
            gt_union_all |= gt_full == gt_id
        false_reports.append(
            {
                "crop_pred_id": crop_pred_id,
                "crop_bbox_xyxy": _bbox(crop_mask),
                "crop_area": _area(crop_mask),
                "pred_pixels_in_roi": _area(full_mask),
                "gt_pixels_in_roi_under_this_blob": _area(gt_full[full_mask] > 0),
                "source_full_frame_pred_ids": source_pred_ids,
                "source_full_frame_gt_ids": source_gt_ids,
                "parents": parent_summaries,
            }
        )

    removed_gt_reports: list[dict[str, object]] = []
    for gt_id in gt_removed_ids:
        gt_mask = gt_full == gt_id
        overlapping_pred = [int(value) for value in np.unique(pred_labels_full[gt_mask]) if value > 0]
        pred_union = np.zeros(gt_full.shape, dtype=bool)
        for pred_id in overlapping_pred:
            pred_union |= pred_labels_full == pred_id
        removed_gt_reports.append(
            {
                "gt_id": gt_id,
                "full_frame_gt_bbox": _bbox(gt_mask),
                "gt_area_full": _area(gt_mask),
                "gt_area_in_roi": _area(gt_mask & roi),
                "pred_ids": overlapping_pred,
                "pred_area_in_roi": _area(pred_union & roi),
                "pred_continues_into_roi": _area(pred_union & roi) > 0,
                "spatial_overlap_in_roi": _area(gt_mask & pred_union & roi) > 0,
            }
        )

    leak_count = sum(1 for item in removed_gt_reports if item["pred_continues_into_roi"])
    no_gt_in_roi = sum(1 for item in false_reports if int(item["gt_pixels_in_roi_under_this_blob"]) == 0)
    crop_created_splits = sum(
        1
        for item in false_reports
        for parent in item["parents"]
        if parent["crop_created_split"]
    )
    already_disconnected = sum(
        1
        for item in false_reports
        for parent in item["parents"]
        if parent["already_disconnected_full_frame"]
    )

    if leak_count == len(false_reports) and no_gt_in_roi == len(false_reports):
        verdict = "A_real_prediction_leak_into_roi"
    elif keep_id_metrics.topology_exact_match and not crop_metrics.topology_exact_match:
        verdict = "B_evaluation_binary_cc_changed_polygon_identity"
    elif crop_created_splits == len(false_reports):
        verdict = "A_crop_split_real_components"
    else:
        verdict = "mixed_needs_review"

    report = {
        "frame": FRAME_ID,
        "crop_px": CROP_PX,
        "verdict": verdict,
        "full_frame": {
            "gt_components": full_metrics.expected_components,
            "pred_components": full_metrics.predicted_components,
            "false_components": full_metrics.false_positive_components,
            "misses": full_metrics.missed_expected_components,
            "merges": full_metrics.false_merges,
            "splits": full_metrics.false_splits,
            "topology_exact_match": full_metrics.topology_exact_match,
        },
        "counts": {
            "gt_polygon_ids_removed_by_crop": len(gt_removed_ids),
            "pred_polygon_ids_removed_by_crop": len(pred_removed_ids),
            "gt_polygon_ids_remaining": keep_id_gt_count,
            "pred_polygon_ids_remaining": keep_id_pred_count,
            "gt_binary_cc_after_crop": binary_gt_count,
            "pred_binary_cc_after_crop": binary_pred_count,
            "pred_per_id_fragments_after_crop": per_id_pred_fragments,
            "gt_per_id_fragments_after_crop": per_id_gt_fragments,
            "pred_ids_split_by_cc": split_pred_ids,
            "gt_ids_split_by_cc": split_gt_ids,
        },
        "if_crop_keeps_polygon_ids": {
            "expected_components": keep_id_metrics.expected_components,
            "predicted_components": keep_id_metrics.predicted_components,
            "false_components": keep_id_metrics.false_positive_components,
            "misses": keep_id_metrics.missed_expected_components,
            "merges": keep_id_metrics.false_merges,
            "splits": keep_id_metrics.false_splits,
            "topology_exact_match": keep_id_metrics.topology_exact_match,
        },
        "current_binary_cc_crop": {
            "expected_components": crop_metrics.expected_components,
            "predicted_components": crop_metrics.predicted_components,
            "false_components": crop_metrics.false_positive_components,
            "misses": crop_metrics.missed_expected_components,
            "merges": crop_metrics.false_merges,
            "splits": crop_metrics.false_splits,
            "topology_exact_match": crop_metrics.topology_exact_match,
        },
        "summary": {
            "false_crop_components": len(false_reports),
            "false_blobs_with_zero_gt_in_roi": no_gt_in_roi,
            "removed_gt_with_pred_leak_into_roi": leak_count,
            "false_parents_split_by_crop": crop_created_splits,
            "false_parents_already_disconnected_full_frame": already_disconnected,
        },
        "false_components": false_reports,
        "removed_gt_components": removed_gt_reports,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("verdict", "full_frame", "counts", "if_crop_keeps_polygon_ids", "current_binary_cc_crop", "summary")}, indent=2))
    print(f"wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
