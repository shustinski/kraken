"""Benchmark-only transverse-profile diagnostic for the conductor-band hypothesis.

Selects local regions from hard frames using GT (selection/plots only), then
samples orientation-aware normal profiles with the runtime band machinery.
Ground truth, frame ids, and filenames are never passed into the algorithm.
"""

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

from contour.vision.metal_recovery.conductor_bands import (  # noqa: E402
    BandEvidence,
    BoundaryPair,
    TransverseProfile,
    detect_boundary_pair,
    sample_transverse_profile,
)
from contour.vision.metal_recovery.gradient_watershed import GradientWatershedConfig  # noqa: E402
from contour.vision.metal_recovery.structural_watershed import (  # noqa: E402
    clamped_structural_watershed_config,
    run_structural_watershed,
    _extract_structural_features,
)
from scripts.benchmark_metal_segmentation import (  # noqa: E402
    EVALUATION_BORDER_CROP_PX,
    REAL_DATASET_ROOT,
    build_real_benchmark_cases,
    crop_evaluation_region,
)
from scripts.benchmark_structural_watershed import remap_positive_ids  # noqa: E402

HARD_FRAMES = ("0175", "0580", "3242")
CROP_PX = EVALUATION_BORDER_CROP_PX
OUTPUT_ROOT = PLUGIN_ROOT / "benchmarks" / "structural_debug" / "conductor_bands"
ROI_MARGIN = 28
ROI_MIN = 80


def main() -> int:
    cases = {case.name: case for case in build_real_benchmark_cases(REAL_DATASET_ROOT)}
    report: dict[str, object] = {"crop_px": CROP_PX, "frames": {}}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for frame_id in HARD_FRAMES:
        case = cases[frame_id]
        frame_report = _diagnose_frame(case.image, case.labels, frame_id)
        report["frames"][frame_id] = frame_report
        print(
            f"{frame_id}: gt={frame_report['gt_count']} "
            f"logical={frame_report['logical_marker_count']} "
            f"multi={frame_report['gt_with_more_than_one_marker']} "
            f"enclosed={frame_report['multi_gt_enclosed_in_one_pair']} "
            f"median_ridges_in_pair={frame_report['median_same_gt_markers_in_pair']:.2f} "
            f"cross_in_pair={frame_report['pairs_covering_multiple_gt']}",
            flush=True,
        )
    path = PLUGIN_ROOT / "benchmarks" / "diagnose_conductor_bands.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)
    return 0


def _diagnose_frame(image: np.ndarray, gt_full: np.ndarray, frame_id: str) -> dict[str, object]:
    config = clamped_structural_watershed_config(variant="s10")
    result = run_structural_watershed(
        image,
        GradientWatershedConfig(),
        config,
        check_presence=False,
    )
    features = _extract_structural_features(image, config)
    evidence = BandEvidence(
        intensity=features.denoised.astype(np.float32),
        ridge_confidence=features.ridge_confidence,
        ridge_orientation=features.ridge_orientation,
        structure_orientation=features.structure_orientation,
        coherence=features.coherence,
        persistent_edge=features.persistent_edge,
        magnitude=features.magnitude,
        rim_response=features.rim_response,
        gradient_x=features.gradient_x,
        gradient_y=features.gradient_y,
    )
    logical_full = result.debug_images.get("metal_structural_logical_markers_i32")
    if logical_full is None:
        logical_full = np.zeros(image.shape[:2], dtype=np.int32)
    gt_eval = crop_evaluation_region(gt_full, CROP_PX, frame_id=frame_id)
    logical_eval = remap_positive_ids(
        crop_evaluation_region(logical_full, CROP_PX, frame_id=frame_id)
    )
    membership = _marker_membership(gt_eval, logical_eval)
    pair_stats = _pair_enclosure_stats(
        gt_eval,
        logical_eval,
        evidence,
        origin_offset=CROP_PX,
    )
    rois = _select_rois(gt_eval, logical_eval, membership)
    frame_dir = OUTPUT_ROOT / frame_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    roi_summaries = []
    for index, roi in enumerate(rois, start=1):
        summary = _save_roi(
            frame_dir,
            index,
            roi,
            image,
            features.structure_orientation,
            features.coherence,
            gt_full,
            logical_full,
            evidence,
        )
        roi_summaries.append(summary)
    return {
        "gt_count": int(len(membership["markers_per_gt"])),
        "logical_marker_count": int(np.unique(logical_eval[logical_eval > 0]).size),
        "gt_with_more_than_one_marker": int(sum(1 for count in membership["markers_per_gt"].values() if count > 1)),
        "gt_with_exactly_one_marker": int(sum(1 for count in membership["markers_per_gt"].values() if count == 1)),
        "gt_without_marker": int(sum(1 for count in membership["markers_per_gt"].values() if count == 0)),
        "cross_gt_markers": int(membership["cross_gt_markers"]),
        "mean_markers_per_gt": float(np.mean(list(membership["markers_per_gt"].values()) or [0])),
        "median_markers_per_gt": float(np.median(list(membership["markers_per_gt"].values()) or [0])),
        **pair_stats,
        "rois": roi_summaries,
        "consolidation": {
            "raw_ridge_count": int(result.consolidation.raw_ridge_count),
            "logical_ridge_count": int(result.consolidation.logical_ridge_count),
            "combined_logical_count": int(result.consolidation.combined_logical_count),
            "logical_wide_count": int(result.consolidation.logical_wide_count),
        },
    }


def _marker_membership(gt_labels: np.ndarray, logical_labels: np.ndarray) -> dict[str, object]:
    gt_ids = np.unique(gt_labels[gt_labels > 0])
    markers_per_gt = {int(gt_id): 0 for gt_id in gt_ids}
    marker_to_gt: dict[int, list[int]] = {}
    cross = 0
    marker_ids = np.unique(logical_labels[logical_labels > 0])
    for marker_id in marker_ids:
        covered = np.unique(gt_labels[logical_labels == marker_id])
        covered = covered[covered > 0]
        marker_to_gt[int(marker_id)] = [int(item) for item in covered]
        if covered.size > 1:
            cross += 1
        for gt_id in covered:
            markers_per_gt[int(gt_id)] += 1
    return {
        "markers_per_gt": markers_per_gt,
        "marker_to_gt": marker_to_gt,
        "cross_gt_markers": cross,
    }


def _pair_enclosure_stats(
    gt_eval: np.ndarray,
    logical_eval: np.ndarray,
    evidence: BandEvidence,
    *,
    origin_offset: int,
) -> dict[str, float | int]:
    gt_ids = np.unique(gt_eval[gt_eval > 0])
    centroids = _label_centroids(logical_eval)
    marker_gt = _marker_gt_ids(gt_eval, logical_eval)
    enclosed = 0
    multi = 0
    same_in_pair: list[int] = []
    extra_gt_in_pair = 0
    pair_found = 0
    for gt_id in gt_ids:
        region = gt_eval == gt_id
        marker_ids = np.unique(logical_eval[region])
        marker_ids = marker_ids[marker_ids > 0]
        if marker_ids.size <= 1:
            continue
        multi += 1
        ys, xs = np.nonzero(region)
        row = float(np.mean(ys)) + origin_offset
        col = float(np.mean(xs)) + origin_offset
        tangent_x, tangent_y = _tangent_from_mask(region)
        profile = sample_transverse_profile(evidence, row, col, tangent_x, tangent_y)
        pair = detect_boundary_pair(profile)
        if pair is None:
            continue
        pair_found += 1
        inside_same = 0
        other_gt = 0
        for marker_id, (mx, my) in centroids.items():
            across, along = _project(
                mx + origin_offset,
                my + origin_offset,
                col,
                row,
                profile,
            )
            if abs(along) > max(12.0, 0.8 * pair.width):
                continue
            if pair.left_offset <= across <= pair.right_offset:
                covered = marker_gt.get(int(marker_id), ())
                if len(covered) == 1 and int(covered[0]) == int(gt_id):
                    inside_same += 1
                elif covered:
                    other_gt += 1
        same_in_pair.append(inside_same)
        if inside_same >= marker_ids.size and other_gt == 0:
            enclosed += 1
        if other_gt > 0:
            extra_gt_in_pair += 1
    return {
        "multi_gt_count": multi,
        "multi_gt_with_detected_pair": pair_found,
        "multi_gt_enclosed_in_one_pair": enclosed,
        "median_same_gt_markers_in_pair": float(np.median(same_in_pair) if same_in_pair else 0.0),
        "mean_same_gt_markers_in_pair": float(np.mean(same_in_pair) if same_in_pair else 0.0),
        "pairs_covering_multiple_gt": extra_gt_in_pair,
    }


def _label_centroids(labels: np.ndarray) -> dict[int, tuple[float, float]]:
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if ids.size == 0:
        return {}
    height, width = labels.shape
    ys, xs = np.indices((height, width))
    max_id = int(ids.max())
    weights = (labels > 0).astype(np.float64)
    counts = np.bincount(labels.ravel(), minlength=max_id + 1).astype(np.float64)
    sum_x = np.bincount(labels.ravel(), weights=xs.ravel() * weights.ravel(), minlength=max_id + 1)
    sum_y = np.bincount(labels.ravel(), weights=ys.ravel() * weights.ravel(), minlength=max_id + 1)
    out: dict[int, tuple[float, float]] = {}
    for label_id in ids:
        count = max(float(counts[int(label_id)]), 1.0)
        out[int(label_id)] = (float(sum_x[int(label_id)] / count), float(sum_y[int(label_id)] / count))
    return out


def _marker_gt_ids(gt_labels: np.ndarray, logical_labels: np.ndarray) -> dict[int, tuple[int, ...]]:
    mapping: dict[int, tuple[int, ...]] = {}
    for marker_id in np.unique(logical_labels[logical_labels > 0]):
        covered = np.unique(gt_labels[logical_labels == marker_id])
        covered = covered[covered > 0]
        mapping[int(marker_id)] = tuple(int(item) for item in covered)
    return mapping


def _select_rois(
    gt_eval: np.ndarray,
    logical_eval: np.ndarray,
    membership: dict[str, object],
) -> list[dict[str, object]]:
    markers_per_gt: dict[int, int] = membership["markers_per_gt"]  # type: ignore[assignment]
    rois: list[dict[str, object]] = []
    multi = sorted(
        ((gt_id, count) for gt_id, count in markers_per_gt.items() if count > 1),
        key=lambda item: (abs(item[1] - 2), -item[1]),
    )
    for gt_id, count in multi[:6]:
        rois.append(_gt_roi(gt_eval, gt_id, f"multi_{count}m"))
    singles = [gt_id for gt_id, count in markers_per_gt.items() if count == 1]
    for gt_id in singles[:3]:
        rois.append(_gt_roi(gt_eval, gt_id, "single"))
    areas = []
    for gt_id in markers_per_gt:
        area = int(np.count_nonzero(gt_eval == gt_id))
        areas.append((area, gt_id))
    for area, gt_id in sorted(areas, reverse=True)[:2]:
        rois.append(_gt_roi(gt_eval, gt_id, f"wide_{area}"))
    centroids = {
        gt_id: _centroid(gt_eval == gt_id) for gt_id in markers_per_gt if markers_per_gt[gt_id] > 0
    }
    neighbors: list[tuple[float, int, int]] = []
    ids = list(centroids.keys())
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            dy = centroids[left][0] - centroids[right][0]
            dx = centroids[left][1] - centroids[right][1]
            dist = float(np.hypot(dx, dy))
            if 12.0 <= dist <= 48.0:
                neighbors.append((dist, left, right))
    neighbors.sort()
    for dist, left, right in neighbors[:3]:
        union = (gt_eval == left) | (gt_eval == right)
        rois.append(
            {
                "kind": "between",
                "gt_ids": [left, right],
                "bbox": _bbox(union),
                "label": f"between_{left}_{right}_{dist:.0f}",
            }
        )
    return [roi for roi in rois if roi["bbox"] is not None]


def _gt_roi(gt_eval: np.ndarray, gt_id: int, kind: str) -> dict[str, object]:
    region = gt_eval == gt_id
    return {
        "kind": kind,
        "gt_ids": [int(gt_id)],
        "bbox": _bbox(region),
        "label": f"{kind}_{gt_id}",
    }


def _save_roi(
    frame_dir: Path,
    index: int,
    roi: dict[str, object],
    image: np.ndarray,
    orientation: np.ndarray,
    coherence: np.ndarray,
    gt_full: np.ndarray,
    logical_full: np.ndarray,
    evidence: BandEvidence,
) -> dict[str, object]:
    x0, y0, x1, y1 = _expand_bbox(roi["bbox"], image.shape[:2])  # type: ignore[arg-type]
    gray = image[y0:y1, x0:x1]
    gt_crop = gt_full[y0:y1, x0:x1]
    logical_crop = logical_full[y0:y1, x0:x1]
    center_row = 0.5 * (y0 + y1 - 1)
    center_col = 0.5 * (x0 + x1 - 1)
    if roi["gt_ids"]:
        gt_id = int(roi["gt_ids"][0])  # type: ignore[index]
        region = gt_full == gt_id
        ys, xs = np.nonzero(region)
        if ys.size:
            center_row = float(np.mean(ys))
            center_col = float(np.mean(xs))
            tangent_x, tangent_y = _tangent_from_mask(region)
        else:
            tangent_x, tangent_y = 1.0, 0.0
    else:
        tangent_x, tangent_y = 1.0, 0.0
    profile = sample_transverse_profile(evidence, center_row, center_col, tangent_x, tangent_y)
    pair = detect_boundary_pair(profile)
    prefix = f"{index:02d}_{roi['label']}"
    cv2.imwrite(str(frame_dir / f"{prefix}_gray.png"), gray)
    cv2.imwrite(
        str(frame_dir / f"{prefix}_orientation.png"),
        _orientation_crop(orientation, coherence, y0, y1, x0, x1),
    )
    cv2.imwrite(
        str(frame_dir / f"{prefix}_overlay.png"),
        _roi_overlay(gray, gt_crop, logical_crop, profile, pair, x0, y0),
    )
    cv2.imwrite(str(frame_dir / f"{prefix}_profile.png"), _draw_profile_plot(profile, pair))
    markers_in_pair = 0
    if pair is not None:
        for marker_id in np.unique(logical_crop[logical_crop > 0]):
            my, mx = np.nonzero(logical_crop == marker_id)
            across, _along = _project(
                float(np.mean(mx)) + x0,
                float(np.mean(my)) + y0,
                center_col,
                center_row,
                profile,
            )
            if pair.left_offset <= across <= pair.right_offset:
                markers_in_pair += 1
    return {
        "label": str(roi["label"]),
        "kind": str(roi["kind"]),
        "gt_ids": list(roi["gt_ids"]),  # type: ignore[arg-type]
        "bbox_full": [x0, y0, x1, y1],
        "pair_found": pair is not None,
        "pair_width": None if pair is None else float(pair.width),
        "pair_confidence": None if pair is None else float(pair.confidence),
        "markers_in_pair": markers_in_pair,
        "separator_absence": None if pair is None else float(pair.separator_absence),
    }


def _orientation_crop(
    orientation: np.ndarray,
    coherence: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    hue = ((orientation[y0:y1, x0:x1] % np.pi) / np.pi * 180.0).astype(np.uint8)
    hsv = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = 220
    hsv[:, :, 2] = np.clip(coherence[y0:y1, x0:x1] * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _roi_overlay(
    gray: np.ndarray,
    gt_crop: np.ndarray,
    logical_crop: np.ndarray,
    profile: TransverseProfile,
    pair: BoundaryPair | None,
    x0: int,
    y0: int,
) -> np.ndarray:
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canvas = (canvas.astype(np.float32) * 0.65).astype(np.uint8)
    gt_edge = cv2.morphologyEx(
        (gt_crop > 0).astype(np.uint8) * 255,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    )
    canvas[gt_edge > 0] = (40, 40, 220)
    for marker_id in np.unique(logical_crop[logical_crop > 0]):
        my, mx = np.nonzero(logical_crop == marker_id)
        canvas[my, mx] = (40, 220, 40)
    start = (
        int(round(profile.origin_col + float(profile.offsets[0]) * profile.normal_x - x0)),
        int(round(profile.origin_row + float(profile.offsets[0]) * profile.normal_y - y0)),
    )
    end = (
        int(round(profile.origin_col + float(profile.offsets[-1]) * profile.normal_x - x0)),
        int(round(profile.origin_row + float(profile.offsets[-1]) * profile.normal_y - y0)),
    )
    cv2.line(canvas, start, end, (40, 200, 255), 1, lineType=cv2.LINE_AA)
    if pair is not None:
        left = (
            int(round(profile.origin_col + pair.left_offset * profile.normal_x - x0)),
            int(round(profile.origin_row + pair.left_offset * profile.normal_y - y0)),
        )
        right = (
            int(round(profile.origin_col + pair.right_offset * profile.normal_x - x0)),
            int(round(profile.origin_row + pair.right_offset * profile.normal_y - y0)),
        )
        cv2.line(canvas, left, right, (0, 220, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, left, 3, (0, 220, 255), -1)
        cv2.circle(canvas, right, 3, (0, 220, 255), -1)
    origin = (
        int(round(profile.origin_col - x0)),
        int(round(profile.origin_row - y0)),
    )
    cv2.circle(canvas, origin, 3, (0, 255, 255), -1)
    return canvas


def _draw_profile_plot(profile: TransverseProfile, pair: BoundaryPair | None) -> np.ndarray:
    width, height = 720, 320
    canvas = np.full((height, width, 3), 18, np.uint8)
    legend = (
        ("I", profile.intensity, (200, 200, 200)),
        ("grad", profile.magnitude, (90, 180, 255)),
        ("ridge", profile.ridge, (70, 220, 90)),
        ("rim", profile.rim, (80, 90, 255)),
        ("boundary", profile.boundary, (40, 170, 230)),
    )
    left_pad, right_pad, top_pad, bottom_pad = 48, 16, 16, 36
    plot_w = width - left_pad - right_pad
    plot_h = height - top_pad - bottom_pad
    offsets = profile.offsets.astype(np.float32)
    x_min = float(offsets[0])
    x_span = max(float(offsets[-1] - offsets[0]), 1e-3)
    if pair is not None:
        x_left = int(left_pad + (pair.left_offset - x_min) / x_span * plot_w)
        x_right = int(left_pad + (pair.right_offset - x_min) / x_span * plot_w)
        cv2.rectangle(canvas, (x_left, top_pad), (x_right, top_pad + plot_h), (40, 40, 28), -1)
        cv2.line(canvas, (x_left, top_pad), (x_left, top_pad + plot_h), (0, 220, 255), 1)
        cv2.line(canvas, (x_right, top_pad), (x_right, top_pad + plot_h), (0, 220, 255), 1)
    x_zero = int(left_pad + (0.0 - x_min) / x_span * plot_w)
    cv2.line(canvas, (x_zero, top_pad), (x_zero, top_pad + plot_h), (60, 60, 60), 1)
    for index, (name, values, color) in enumerate(legend):
        scaled = _unit_plot(values)
        points = np.column_stack(
            [
                left_pad + ((offsets - x_min) / x_span * plot_w),
                top_pad + (1.0 - scaled) * plot_h,
            ]
        ).astype(np.int32)
        cv2.polylines(canvas, [points], False, color, 1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            name,
            (left_pad + 8 + index * 90, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "offset along local normal (px)",
        (left_pad, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _unit_plot(values: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values, (5.0, 95.0)) if values.size else (0.0, 1.0)
    span = float(high - low)
    if span <= 1e-6:
        return np.full(values.shape, 0.5, dtype=np.float32)
    return np.clip((values - float(low)) / span, 0.0, 1.0).astype(np.float32)


def _tangent_from_mask(region: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(region)
    if ys.size < 8:
        return 1.0, 0.0
    coords = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=0)
    centered = coords - coords.mean(axis=1, keepdims=True)
    cov = centered @ centered.T / max(ys.size - 1, 1)
    _vals, vecs = np.linalg.eigh(cov)
    tangent = vecs[:, -1]
    norm = max(float(np.hypot(tangent[0], tangent[1])), 1e-6)
    return float(tangent[0] / norm), float(tangent[1] / norm)


def _project(
    col: float,
    row: float,
    origin_col: float,
    origin_row: float,
    profile: TransverseProfile,
) -> tuple[float, float]:
    dx = col - origin_col
    dy = row - origin_row
    across = dx * profile.normal_x + dy * profile.normal_y
    along = dx * profile.tangent_x + dy * profile.tangent_y
    return float(across), float(along)


def _centroid(region: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(region)
    if ys.size == 0:
        return 0.0, 0.0
    return float(np.mean(ys)), float(np.mean(xs))


def _bbox(region: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(region)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - ROI_MARGIN)
    y0 = max(0, y0 - ROI_MARGIN)
    x1 = min(shape[1], x1 + ROI_MARGIN)
    y1 = min(shape[0], y1 + ROI_MARGIN)
    if x1 - x0 < ROI_MIN:
        extra = ROI_MIN - (x1 - x0)
        x0 = max(0, x0 - extra // 2)
        x1 = min(shape[1], x1 + extra - extra // 2)
    if y1 - y0 < ROI_MIN:
        extra = ROI_MIN - (y1 - y0)
        y0 = max(0, y0 - extra // 2)
        y1 = min(shape[0], y1 + extra - extra // 2)
    return x0, y0, x1, y1


if __name__ == "__main__":
    raise SystemExit(main())
