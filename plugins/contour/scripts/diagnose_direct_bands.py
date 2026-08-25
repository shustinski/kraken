"""Falsification experiment: direct conductor bands from continuous maps.

Builds candidate bands on frame 3242 from orientation / ridge / boundary fields.
Ridge connected-component IDs, marker IDs, S7 propagation, watershed fill, and
GT are not used during inference.  GT is applied only after bands exist.

No production recognition path is modified.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
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
    has_separating_boundary,
    sample_transverse_profile,
)
from contour.vision.metal_recovery.gradient_watershed import GradientWatershedConfig  # noqa: E402
from contour.vision.metal_recovery.structural_watershed import (  # noqa: E402
    clamped_structural_watershed_config,
    run_structural_watershed,
    _extract_structural_features,
    _non_maximum_suppress,
)
from scripts.benchmark_metal_segmentation import (  # noqa: E402
    EVALUATION_BORDER_CROP_PX,
    REAL_DATASET_ROOT,
    build_real_benchmark_cases,
    crop_evaluation_region,
    relabel_connected_components,
)
from scripts.benchmark_structural_watershed import remap_positive_ids  # noqa: E402

FRAME_ID = "3242"
CROP_PX = EVALUATION_BORDER_CROP_PX
OUTPUT_JSON = PLUGIN_ROOT / "benchmarks" / "diagnose_direct_bands_3242.json"
OUTPUT_DIR = PLUGIN_ROOT / "benchmarks" / "structural_debug" / "direct_bands" / FRAME_ID
RNG = np.random.default_rng(3242)
ROI_MARGIN = 28
ALONG_STAMP_PX = 5.0
MAX_SAMPLES = 6000


@dataclass(frozen=True, slots=True)
class BandSample:
    row: float
    col: float
    profile: TransverseProfile
    pair: BoundaryPair
    ridge_peaks: tuple[float, ...]
    separator: bool


def main() -> int:
    cases = {case.name: case for case in build_real_benchmark_cases(REAL_DATASET_ROOT)}
    case = cases[FRAME_ID]
    config = clamped_structural_watershed_config(variant="s7")
    features = _extract_structural_features(case.image, config)
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
    samples = _detect_band_samples(features, evidence, config)
    band_labels = _rasterize_bands(features.denoised.shape, samples)
    current = run_structural_watershed(
        case.image,
        GradientWatershedConfig(),
        clamped_structural_watershed_config(variant="s10"),
        check_presence=False,
    )
    logical = current.debug_images.get("metal_structural_logical_markers_i32")
    if logical is None:
        logical = np.zeros(case.image.shape[:2], dtype=np.int32)

    gt_eval = relabel_connected_components(
        crop_evaluation_region(case.labels, CROP_PX, frame_id=FRAME_ID)
    )
    band_eval = remap_positive_ids(
        crop_evaluation_region(band_labels, CROP_PX, frame_id=FRAME_ID)
    )
    logical_eval = remap_positive_ids(
        crop_evaluation_region(logical, CROP_PX, frame_id=FRAME_ID)
    )
    band_stats = _overlap_stats(gt_eval, band_eval)
    marker_stats = _overlap_stats(gt_eval, logical_eval)
    multi_gt = _gt_ids_with_count(marker_stats["bands_per_gt"], minimum=2)
    multi_table = [
        {
            "gt_id": int(gt_id),
            "current_markers": int(marker_stats["bands_per_gt"][gt_id]),
            "direct_bands": int(band_stats["bands_per_gt"].get(gt_id, 0)),
        }
        for gt_id in sorted(multi_gt)
    ]
    report = {
        "frame": FRAME_ID,
        "evaluation_border_crop_px": CROP_PX,
        "inference": {
            "sample_sites": len(samples),
            "candidate_bands": int(band_eval.max()) if band_eval.size else 0,
            "used_ridge_cc_ids": False,
            "used_marker_ids": False,
            "used_gt": False,
        },
        "current_s10_markers": {
            "gt_count": marker_stats["gt_count"],
            "gt_with_0": marker_stats["gt_with_0"],
            "gt_with_exactly_1": marker_stats["gt_with_exactly_1"],
            "gt_with_more_than_1": marker_stats["gt_with_more_than_1"],
            "median_per_gt": marker_stats["median_per_gt"],
        },
        "direct_bands": {
            "gt_count": band_stats["gt_count"],
            "gt_with_0_candidate_bands": band_stats["gt_with_0"],
            "gt_with_exactly_1_candidate_band": band_stats["gt_with_exactly_1"],
            "gt_with_more_than_1_candidate_band": band_stats["gt_with_more_than_1"],
            "bands_overlapping_0_gt": band_stats["item_overlapping_0"],
            "bands_overlapping_exactly_1_gt": band_stats["item_overlapping_1"],
            "bands_overlapping_more_than_1_gt": band_stats["item_overlapping_multi"],
            "mean_bands_per_gt": band_stats["mean_per_gt"],
            "median_bands_per_gt": band_stats["median_per_gt"],
            "p90_bands_per_gt": band_stats["p90_per_gt"],
        },
        "current_multi_marker_gt": {
            "count": len(multi_table),
            "direct_bands_on_those_gt": _count_histogram(
                [row["direct_bands"] for row in multi_table]
            ),
            "median_direct_bands": float(
                np.median([row["direct_bands"] for row in multi_table]) if multi_table else 0.0
            ),
            "rows": multi_table,
        },
        "gate": _gate(band_stats),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rois = _save_rois(
        case.image,
        features,
        evidence,
        samples,
        band_labels,
        case.labels,
        gt_eval,
        band_eval,
        band_stats,
    )
    report["rois"] = rois
    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_report(report)
    print(f"wrote {OUTPUT_JSON}", flush=True)
    print(f"rois {OUTPUT_DIR}", flush=True)
    return 0


def _detect_band_samples(features, evidence: BandEvidence, config) -> list[BandSample]:
    along = features.structure_orientation + (0.5 * np.pi)
    nms = _non_maximum_suppress(features.ridge_confidence, along)
    coherent = features.coherence >= float(config.min_orientation_coherence)
    if np.any(nms > 0):
        floor = float(np.percentile(features.ridge_confidence[nms > 0], 55.0))
    else:
        floor = float(config.min_ridge_confidence)
    keep = (
        (nms > 0)
        & coherent
        & (features.ridge_confidence >= max(floor, float(config.min_ridge_confidence)))
    )
    ys, xs = np.nonzero(keep)
    if ys.size == 0:
        return []
    scores = features.ridge_confidence[ys, xs]
    order = np.argsort(scores)[::-1]
    ys = ys[order]
    xs = xs[order]
    packed = _pack(evidence)
    occupied = np.zeros(keep.shape, dtype=np.uint8)
    samples: list[BandSample] = []
    for row, col in zip(ys.tolist(), xs.tolist(), strict=True):
        if occupied[row, col]:
            continue
        angle = float(features.structure_orientation[row, col]) + 0.5 * np.pi
        profile = sample_transverse_profile(
            evidence,
            float(row),
            float(col),
            float(np.cos(angle)),
            float(np.sin(angle)),
            packed=packed,
        )
        pair = detect_boundary_pair(profile)
        if pair is None:
            continue
        occupied[
            max(0, row - 2) : row + 3,
            max(0, col - 2) : col + 3,
        ] = 1
        peaks = _ridge_peak_offsets(profile, pair)
        separator = has_separating_boundary(profile, pair.left_offset, pair.right_offset)
        samples.append(
            BandSample(
                row=float(row),
                col=float(col),
                profile=profile,
                pair=pair,
                ridge_peaks=peaks,
                separator=separator,
            )
        )
        if len(samples) >= MAX_SAMPLES:
            break
    return samples


def _rasterize_bands(shape: tuple[int, ...], samples: list[BandSample]) -> np.ndarray:
    painted = np.zeros(shape[:2], dtype=np.uint8)
    for sample in samples:
        _stamp_interval(painted, sample)
    count, labels = cv2.connectedComponents(painted, connectivity=8)
    del count
    return labels.astype(np.int32)


def _stamp_interval(canvas: np.ndarray, sample: BandSample) -> None:
    height, width = canvas.shape
    for along in (-ALONG_STAMP_PX, -2.0, 0.0, 2.0, ALONG_STAMP_PX):
        left = (
            int(round(sample.col + along * sample.profile.tangent_x + sample.pair.left_offset * sample.profile.normal_x)),
            int(round(sample.row + along * sample.profile.tangent_y + sample.pair.left_offset * sample.profile.normal_y)),
        )
        right = (
            int(round(sample.col + along * sample.profile.tangent_x + sample.pair.right_offset * sample.profile.normal_x)),
            int(round(sample.row + along * sample.profile.tangent_y + sample.pair.right_offset * sample.profile.normal_y)),
        )
        if 0 <= left[0] < width and 0 <= left[1] < height and 0 <= right[0] < width and 0 <= right[1] < height:
            cv2.line(canvas, left, right, 255, 2, lineType=cv2.LINE_8)


def _overlap_stats(gt_labels: np.ndarray, item_labels: np.ndarray) -> dict[str, object]:
    gt_ids = np.unique(gt_labels[gt_labels > 0])
    item_ids = np.unique(item_labels[item_labels > 0])
    bands_per_gt: dict[int, int] = {int(gt_id): 0 for gt_id in gt_ids}
    gt_per_item: dict[int, int] = {int(item_id): 0 for item_id in item_ids}
    valid = (gt_labels > 0) & (item_labels > 0)
    if np.any(valid):
        packed = item_labels.astype(np.int64) * (int(gt_labels.max()) + 1) + gt_labels.astype(np.int64)
        pairs = np.unique(packed[valid])
        stride = int(gt_labels.max()) + 1
        item_from = (pairs // stride).astype(np.int32)
        gt_from = (pairs % stride).astype(np.int32)
        for item_id, gt_id in zip(item_from.tolist(), gt_from.tolist(), strict=True):
            if item_id > 0 and gt_id > 0:
                gt_per_item[int(item_id)] = gt_per_item.get(int(item_id), 0) + 1
                bands_per_gt[int(gt_id)] = bands_per_gt.get(int(gt_id), 0) + 1
        for item_id in item_ids:
            if int(item_id) not in gt_per_item:
                gt_per_item[int(item_id)] = 0
    values = list(bands_per_gt.values()) if bands_per_gt else [0]
    item_counts = list(gt_per_item.values()) if gt_per_item else [0]
    return {
        "gt_count": int(gt_ids.size),
        "item_count": int(item_ids.size),
        "gt_with_0": int(sum(1 for count in values if count == 0)),
        "gt_with_exactly_1": int(sum(1 for count in values if count == 1)),
        "gt_with_more_than_1": int(sum(1 for count in values if count > 1)),
        "item_overlapping_0": int(sum(1 for count in item_counts if count == 0)),
        "item_overlapping_1": int(sum(1 for count in item_counts if count == 1)),
        "item_overlapping_multi": int(sum(1 for count in item_counts if count > 1)),
        "mean_per_gt": float(np.mean(values)),
        "median_per_gt": float(np.median(values)),
        "p90_per_gt": float(np.percentile(values, 90.0)),
        "bands_per_gt": bands_per_gt,
        "gt_per_item": gt_per_item,
    }


def _gate(band_stats: dict[str, object]) -> dict[str, object]:
    median = float(band_stats["median_per_gt"])
    multi = int(band_stats["gt_with_more_than_1"])
    cross = int(band_stats["item_overlapping_multi"])
    passed = median <= 1.0 and multi < 200 and cross <= 10
    return {
        "continue_band_native": passed,
        "median_bands_per_gt": median,
        "gt_with_more_than_1": multi,
        "cross_gt_bands": cross,
        "reason": (
            "direct detector assigns about one band per GT with low cross-GT overlap"
            if passed
            else "direct detector still over-segments or cross-links GTs; close band-native"
        ),
    }


def _save_rois(
    image: np.ndarray,
    features,
    evidence: BandEvidence,
    samples: list[BandSample],
    band_labels: np.ndarray,
    gt_full: np.ndarray,
    gt_eval: np.ndarray,
    band_eval: np.ndarray,
    band_stats: dict[str, object],
) -> list[dict[str, object]]:
    bands_per_gt: dict[int, int] = band_stats["bands_per_gt"]  # type: ignore[assignment]
    gt_per_item: dict[int, int] = band_stats["gt_per_item"]  # type: ignore[assignment]
    one = _gt_ids_with_count(bands_per_gt, exact=1)
    multi = _gt_ids_with_count(bands_per_gt, minimum=2)
    zero = _gt_ids_with_count(bands_per_gt, exact=0)
    cross_ids = [item_id for item_id, count in gt_per_item.items() if count > 1]
    chosen: list[tuple[str, str, int]] = []
    chosen.extend(("gt_one", "gt", int(gt_id)) for gt_id in _take_random(one, 30))
    chosen.extend(("gt_multi", "gt", int(gt_id)) for gt_id in _take_random(multi, 30))
    chosen.extend(("gt_zero", "gt", int(gt_id)) for gt_id in _take_random(zero, 8))
    chosen.extend(("cross_gt", "band", int(item_id)) for item_id in cross_ids[:40])
    sample_by_pixel = {(int(round(s.row)), int(round(s.col))): s for s in samples}
    summaries: list[dict[str, object]] = []
    for index, (kind, target, ident) in enumerate(chosen, start=1):
        if target == "gt":
            region = gt_eval == ident
            bbox = _bbox(region)
            if bbox is None:
                continue
            x0, y0, x1, y1 = _shift_bbox(bbox, CROP_PX, image.shape[:2])
        else:
            region = band_eval == ident
            bbox = _bbox(region)
            if bbox is None:
                continue
            x0, y0, x1, y1 = _shift_bbox(bbox, CROP_PX, image.shape[:2])
        sample = _nearest_sample(sample_by_pixel, y0, x0, y1, x1)
        prefix = f"{index:03d}_{kind}_{ident}"
        gray = image[y0:y1, x0:x1]
        cv2.imwrite(str(OUTPUT_DIR / f"{prefix}_gray.png"), gray)
        cv2.imwrite(
            str(OUTPUT_DIR / f"{prefix}_orientation.png"),
            _orientation_crop(features.structure_orientation, features.coherence, y0, y1, x0, x1),
        )
        cv2.imwrite(str(OUTPUT_DIR / f"{prefix}_ridge.png"), _to_u8(features.ridge_response[y0:y1, x0:x1]))
        cv2.imwrite(str(OUTPUT_DIR / f"{prefix}_boundary.png"), _to_u8(features.persistent_edge[y0:y1, x0:x1]))
        algo = _algorithm_overlay(gray, band_labels[y0:y1, x0:x1], sample, x0, y0)
        cv2.imwrite(str(OUTPUT_DIR / f"{prefix}_band.png"), algo)
        gt_vis = _gt_overlay(gray, gt_full[y0:y1, x0:x1], band_labels[y0:y1, x0:x1])
        cv2.imwrite(str(OUTPUT_DIR / f"{prefix}_gt.png"), gt_vis)
        if sample is not None:
            cv2.imwrite(
                str(OUTPUT_DIR / f"{prefix}_profile.png"),
                _profile_plot(sample.profile, sample.pair, sample.ridge_peaks, sample.separator),
            )
        summaries.append(
            {
                "file_prefix": prefix,
                "kind": kind,
                "id": ident,
                "bbox_full": [x0, y0, x1, y1],
                "pair_width": None if sample is None else float(sample.pair.width),
                "ridge_peaks_in_pair": None if sample is None else len(sample.ridge_peaks),
                "separator": None if sample is None else bool(sample.separator),
            }
        )
    return summaries


def _algorithm_overlay(
    gray: np.ndarray,
    band_crop: np.ndarray,
    sample: BandSample | None,
    x0: int,
    y0: int,
) -> np.ndarray:
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canvas = (canvas.astype(np.float32) * 0.55).astype(np.uint8)
    band_edge = cv2.morphologyEx(
        (band_crop > 0).astype(np.uint8) * 255,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    )
    canvas[band_edge > 0] = (0, 180, 255)
    if sample is None:
        return canvas
    start = _pt(sample, float(sample.profile.offsets[0]), 0.0, x0, y0)
    end = _pt(sample, float(sample.profile.offsets[-1]), 0.0, x0, y0)
    cv2.line(canvas, start, end, (180, 180, 40), 1, lineType=cv2.LINE_AA)
    left = _pt(sample, sample.pair.left_offset, 0.0, x0, y0)
    right = _pt(sample, sample.pair.right_offset, 0.0, x0, y0)
    cv2.line(canvas, left, right, (0, 220, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(canvas, left, 3, (0, 220, 255), -1)
    cv2.circle(canvas, right, 3, (0, 220, 255), -1)
    color = (0, 0, 220) if sample.separator else (40, 220, 90)
    for peak in sample.ridge_peaks:
        cv2.circle(canvas, _pt(sample, peak, 0.0, x0, y0), 3, color, -1)
    origin = (
        int(round(sample.col - x0)),
        int(round(sample.row - y0)),
    )
    cv2.circle(canvas, origin, 3, (0, 255, 255), -1)
    return canvas


def _gt_overlay(gray: np.ndarray, gt_crop: np.ndarray, band_crop: np.ndarray) -> np.ndarray:
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canvas = (canvas.astype(np.float32) * 0.55).astype(np.uint8)
    gt_edge = cv2.morphologyEx(
        (gt_crop > 0).astype(np.uint8) * 255,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    )
    band_edge = cv2.morphologyEx(
        (band_crop > 0).astype(np.uint8) * 255,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    )
    canvas[band_edge > 0] = (0, 180, 255)
    canvas[gt_edge > 0] = (40, 40, 220)
    return canvas


def _profile_plot(
    profile: TransverseProfile,
    pair: BoundaryPair,
    ridge_peaks: tuple[float, ...],
    separator: bool,
) -> np.ndarray:
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
    x_left = int(left_pad + (pair.left_offset - x_min) / x_span * plot_w)
    x_right = int(left_pad + (pair.right_offset - x_min) / x_span * plot_w)
    cv2.rectangle(canvas, (x_left, top_pad), (x_right, top_pad + plot_h), (40, 40, 28), -1)
    cv2.line(canvas, (x_left, top_pad), (x_left, top_pad + plot_h), (0, 220, 255), 1)
    cv2.line(canvas, (x_right, top_pad), (x_right, top_pad + plot_h), (0, 220, 255), 1)
    for peak in ridge_peaks:
        x_peak = int(left_pad + (peak - x_min) / x_span * plot_w)
        color = (0, 0, 220) if separator else (70, 220, 90)
        cv2.line(canvas, (x_peak, top_pad), (x_peak, top_pad + plot_h), color, 1)
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
    return canvas


def _ridge_peak_offsets(profile: TransverseProfile, pair: BoundaryPair) -> tuple[float, ...]:
    inside = (profile.offsets >= pair.left_offset) & (profile.offsets <= pair.right_offset)
    if int(np.count_nonzero(inside)) < 3:
        return ()
    ridge = profile.ridge
    peak_level = max(float(np.max(ridge[inside])) * 0.35, 1e-3)
    found: list[float] = []
    for index in range(1, int(ridge.size) - 1):
        if not inside[index]:
            continue
        if ridge[index] >= ridge[index - 1] and ridge[index] >= ridge[index + 1] and ridge[index] >= peak_level:
            found.append(float(profile.offsets[index]))
    return tuple(found)


def _pack(evidence: BandEvidence) -> np.ndarray:
    return np.stack(
        [
            evidence.intensity.astype(np.float32),
            evidence.magnitude,
            evidence.ridge_confidence,
            evidence.rim_response,
            evidence.persistent_edge,
            evidence.coherence,
            evidence.gradient_x,
            evidence.gradient_y,
        ],
        axis=-1,
    )


def _gt_ids_with_count(
    counts: dict[int, int],
    *,
    exact: int | None = None,
    minimum: int | None = None,
) -> list[int]:
    found: list[int] = []
    for gt_id, count in counts.items():
        if exact is not None and count == exact:
            found.append(int(gt_id))
        elif minimum is not None and count >= minimum:
            found.append(int(gt_id))
    return found


def _take_random(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return values
    chosen = RNG.choice(np.asarray(values, dtype=np.int32), size=count, replace=False)
    return [int(item) for item in chosen]


def _count_histogram(values: list[int]) -> dict[str, int]:
    return {
        "with_0": int(sum(1 for item in values if item == 0)),
        "with_1": int(sum(1 for item in values if item == 1)),
        "with_more_than_1": int(sum(1 for item in values if item > 1)),
    }


def _bbox(region: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(region)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _shift_bbox(
    bbox: tuple[int, int, int, int],
    crop_px: int,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 + crop_px - ROI_MARGIN)
    y0 = max(0, y0 + crop_px - ROI_MARGIN)
    x1 = min(shape[1], x1 + crop_px + ROI_MARGIN)
    y1 = min(shape[0], y1 + crop_px + ROI_MARGIN)
    if x1 - x0 < 80:
        extra = 80 - (x1 - x0)
        x0 = max(0, x0 - extra // 2)
        x1 = min(shape[1], x1 + extra - extra // 2)
    if y1 - y0 < 80:
        extra = 80 - (y1 - y0)
        y0 = max(0, y0 - extra // 2)
        y1 = min(shape[0], y1 + extra - extra // 2)
    return x0, y0, x1, y1


def _nearest_sample(
    sample_by_pixel: dict[tuple[int, int], BandSample],
    y0: int,
    x0: int,
    y1: int,
    x1: int,
) -> BandSample | None:
    best: BandSample | None = None
    best_dist = 1e9
    center_y = 0.5 * (y0 + y1)
    center_x = 0.5 * (x0 + x1)
    for (row, col), sample in sample_by_pixel.items():
        if not (y0 <= row < y1 and x0 <= col < x1):
            continue
        dist = (row - center_y) ** 2 + (col - center_x) ** 2
        if dist < best_dist:
            best = sample
            best_dist = dist
    return best


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


def _to_u8(values: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(values.astype(np.float32), nan=0.0)
    low, high = np.percentile(finite, (2.0, 98.0)) if finite.size else (0.0, 1.0)
    span = float(high - low)
    if span <= 1e-6:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip((finite - float(low)) / span * 255.0, 0, 255).astype(np.uint8)


def _unit_plot(values: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values, (5.0, 95.0)) if values.size else (0.0, 1.0)
    span = float(high - low)
    if span <= 1e-6:
        return np.full(values.shape, 0.5, dtype=np.float32)
    return np.clip((values - float(low)) / span, 0.0, 1.0).astype(np.float32)


def _pt(sample: BandSample, across: float, along: float, x0: int, y0: int) -> tuple[int, int]:
    col = sample.col + along * sample.profile.tangent_x + across * sample.profile.normal_x
    row = sample.row + along * sample.profile.tangent_y + across * sample.profile.normal_y
    return int(round(col - x0)), int(round(row - y0))


def _print_report(report: dict[str, object]) -> None:
    bands = report["direct_bands"]
    current = report["current_s10_markers"]
    gate = report["gate"]
    multi = report["current_multi_marker_gt"]
    print(
        f"3242 GT={bands['gt_count']} "
        f"direct_bands={report['inference']['candidate_bands']} "
        f"samples={report['inference']['sample_sites']}",
        flush=True,
    )
    print(
        f"direct: gt0={bands['gt_with_0_candidate_bands']} "
        f"gt1={bands['gt_with_exactly_1_candidate_band']} "
        f"gt>1={bands['gt_with_more_than_1_candidate_band']} "
        f"mean={bands['mean_bands_per_gt']:.2f} "
        f"median={bands['median_bands_per_gt']:.2f} "
        f"p90={bands['p90_bands_per_gt']:.2f}",
        flush=True,
    )
    print(
        f"direct overlap: 0GT={bands['bands_overlapping_0_gt']} "
        f"1GT={bands['bands_overlapping_exactly_1_gt']} "
        f">1GT={bands['bands_overlapping_more_than_1_gt']}",
        flush=True,
    )
    print(
        f"S10 markers: gt>1={current['gt_with_more_than_1']} "
        f"median={current['median_per_gt']:.2f}",
        flush=True,
    )
    print(
        f"on those multi-marker GT: "
        f"direct 0/1/>1 = {multi['direct_bands_on_those_gt']['with_0']}/"
        f"{multi['direct_bands_on_those_gt']['with_1']}/"
        f"{multi['direct_bands_on_those_gt']['with_more_than_1']} "
        f"median={multi['median_direct_bands']:.2f}",
        flush=True,
    )
    print(
        f"gate continue_band_native={gate['continue_band_native']} "
        f"({gate['reason']})",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
