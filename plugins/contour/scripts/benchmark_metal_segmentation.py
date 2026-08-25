"""Deterministic topology-sensitive benchmark for classical metal segmentation.

The synthetic scenes intentionally contain blurred edges, shot/read noise,
illumination drift, bright rims, dark conductor interiors, and elevated narrow
gaps.  They are not intended to replace CIF golden tests; they isolate failure
classes whose ground-truth object topology is known exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PLUGIN_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from contour.application.processing import ContourExtractionSettings  # noqa: E402
from contour.domain import PolygonData  # noqa: E402
from contour.serializers import load_polygons_cif  # noqa: E402
from contour.ui.metal_presets import standard_metal_preset_payload  # noqa: E402
from contour.vision.metal_recovery import (  # noqa: E402
    detect_metalization,
    metal_recovery_config_from_settings,
)
from contour.vision.metal_recovery.gradient_watershed import (  # noqa: E402
    GradientWatershedConfig,
    analyze_metal_presence,
    build_conductor_seeds,
    gradient_watershed_mask_from_seeds,
    intensity_class_limits,
    selective_conductor_recovery,
)
from contour.vision.metal_recovery.pipeline_stages import _segment  # noqa: E402
from contour.vision.preprocessing import (  # noqa: E402
    metal_preprocess_config_from_settings,
    preprocess_for_sem,
)

STRATEGIES = (
    "auto",
    "legacy_otsu",
    "local_adaptive",
    "gradient_watershed",
    "random_walker",
    "graph_cut",
    "reconstruction",
    "closed_boundary",
    "structural_watershed",
)

REAL_DATASET_ROOT = PLUGIN_ROOT / "tests" / "test_metal"
# Evaluation-only ROI. Recognition always runs on the full SEM frame.
EVALUATION_BORDER_CROP_PX = 50
BORDER_AUDIT_FRAMES = ("1514", "2497", "0284")


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    category: str
    image: np.ndarray
    labels: np.ndarray
    source_image: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class SegmentationMetrics:
    iou: float
    precision: float
    recall: float
    boundary_f1: float
    expected_components: int
    predicted_components: int
    false_merges: int
    false_splits: int
    false_positive_components: int
    missed_expected_components: int
    matched_expected_components: int
    component_count_absolute_error: int
    component_precision: float
    component_recall: float
    component_f1: float
    false_merge_indicator: bool
    false_split_indicator: bool
    predicted_foreground_fraction: float
    gt_foreground_fraction: float
    false_positive_area_fraction: float
    false_negative_area_fraction: float
    false_metal_fraction: float
    segmentation_ms: float
    refinement_ms: float
    elapsed_ms: float
    topology_exact_match: bool


@dataclass(frozen=True, slots=True)
class SeedDiagnostics:
    substrate_intensity_limit: float
    metal_intensity_limit: float
    core_seed_fraction: float
    groove_seed_fraction: float
    core_connected_components: int
    groove_connected_components: int
    intensity_p01: float
    intensity_p10: float
    intensity_p50: float
    intensity_p90: float
    intensity_p99: float
    gradient_mean: float
    gradient_p90: float
    gradient_p99: float
    local_contrast_mean: float
    local_contrast_p90: float
    local_contrast_p99: float
    presence_has_metal: bool
    presence_robust_intensity_span: float
    presence_coherent_contrast_fraction: float
    presence_largest_coherent_contrast_fraction: float
    presence_local_contrast_limit: float


def _draw_line(labels: np.ndarray, object_id: int, points: Iterable[tuple[int, int]], width: int) -> None:
    vertices = np.asarray(tuple(points), dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(labels, [vertices], False, int(object_id), thickness=int(width), lineType=cv2.LINE_8)


def _draw_rect(labels: np.ndarray, object_id: int, rect: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = rect
    labels[y0:y1, x0:x1] = int(object_id)


def _render_sem(
    labels: np.ndarray,
    *,
    seed: int,
    substrate: float = 46.0,
    metal: float = 105.0,
    rim: float = 205.0,
    rim_width: int = 2,
    dark_center: float | None = None,
    dark_center_depth: float = 8.0,
    noise_sigma: float = 4.0,
    illumination_amplitude: float = 8.0,
    blur_sigma: float = 0.9,
    elevated_regions: tuple[tuple[slice, slice, float], ...] = (),
) -> np.ndarray:
    height, width = labels.shape
    yy, xx = np.indices(labels.shape, dtype=np.float32)
    illumination = illumination_amplitude * (
        0.55 * np.sin(xx / max(24.0, width * 0.31)) + 0.45 * np.cos(yy / max(24.0, height * 0.37))
    )
    image = np.full(labels.shape, substrate, dtype=np.float32) + illumination
    foreground = labels > 0
    image[foreground] = metal + 0.35 * illumination[foreground]

    if dark_center is not None:
        distance = cv2.distanceTransform(foreground.astype(np.uint8), cv2.DIST_L2, 5)
        center = foreground & (distance >= float(dark_center_depth))
        image[center] = float(dark_center) + 0.25 * illumination[center]

    if rim_width > 0:
        kernel_size = 2 * int(rim_width) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        interior_edge = (cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0) & foreground
        image[interior_edge] = float(rim)

    for rows, columns, value in elevated_regions:
        region = labels[rows, columns] == 0
        patch = image[rows, columns]
        patch[region] = float(value)

    rng = np.random.default_rng(seed)
    image += rng.normal(0.0, float(noise_sigma), size=image.shape).astype(np.float32)
    # A scaled Poisson draw approximates shot noise without overwhelming the
    # deliberately weak contrast cases.
    image = rng.poisson(np.clip(image, 0.0, 255.0) * 4.0).astype(np.float32) * 0.25
    if blur_sigma > 0.0:
        image = cv2.GaussianBlur(image, (0, 0), float(blur_sigma))
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def build_benchmark_cases() -> list[BenchmarkCase]:
    shape = (256, 256)
    cases: list[BenchmarkCase] = []

    def add(
        name: str,
        category: str,
        labels: np.ndarray,
        **render_options: object,
    ) -> None:
        image = _render_sem(labels, seed=100 + len(cases), **render_options)
        cases.append(BenchmarkCase(name, category, image, labels))

    labels = np.zeros(shape, np.int32)
    _draw_rect(labels, 1, (32, 48, 224, 208))
    add("large_isolated", "scale", labels)

    labels = np.zeros(shape, np.int32)
    _draw_line(labels, 1, ((20, 30), (235, 226)), 4)
    add("narrow_4px", "scale", labels, metal=88.0, rim=150.0, rim_width=1)

    for gap in (1, 2, 3):
        labels = np.zeros(shape, np.int32)
        left_end = 126 - gap // 2
        right_start = left_end + gap
        _draw_rect(labels, 1, (36, 12, left_end, 244))
        _draw_rect(labels, 2, (right_start, 12, 220, 244))
        add(
            f"parallel_gap_{gap}px",
            "close_conductors",
            labels,
            elevated_regions=((slice(12, 244), slice(left_end, right_start), 105.0),),
        )

    labels = np.zeros(shape, np.int32)
    _draw_rect(labels, 1, (24, 30, 232, 226))
    add(
        "wide_dark_center",
        "dark_interior",
        labels,
        metal=104.0,
        dark_center=49.0,
        dark_center_depth=13.0,
    )

    labels = np.zeros(shape, np.int32)
    _draw_rect(labels, 1, (34, 35, 222, 221))
    add("bright_rims", "rim", labels, metal=76.0, rim=224.0, rim_width=3)

    labels = np.zeros(shape, np.int32)
    _draw_line(labels, 1, ((18, 70), (238, 70), (238, 190), (35, 190)), 10)
    add("weak_contrast", "contrast", labels, metal=72.0, rim=92.0, rim_width=1)

    labels = np.zeros(shape, np.int32)
    _draw_rect(labels, 1, (0, 52, 116, 184))
    add("touches_one_border", "border", labels, dark_center=58.0, dark_center_depth=11.0)

    labels = np.zeros(shape, np.int32)
    _draw_line(labels, 1, ((0, 128), (128, 128), (128, 0)), 28)
    add("crosses_multiple_borders", "border", labels, dark_center=57.0, dark_center_depth=9.0)

    labels = np.zeros(shape, np.int32)
    for object_id, x_coord in enumerate((28, 63, 98, 133, 168, 203), start=1):
        _draw_line(labels, object_id, ((x_coord, 0), (x_coord, 256)), 9)
    add("parallel_bundle", "topology", labels, metal=92.0, rim=178.0, rim_width=2)

    labels = np.zeros(shape, np.int32)
    _draw_line(labels, 1, ((20, 128), (236, 128)), 20)
    _draw_line(labels, 1, ((128, 22), (128, 234)), 20)
    add("junction", "topology", labels, dark_center=62.0, dark_center_depth=7.0)

    labels = np.zeros(shape, np.int32)
    _draw_line(labels, 1, ((15, 55), (240, 55)), 11)
    _draw_line(labels, 2, ((15, 155), (240, 205)), 13)
    add("noisy_background", "noise", labels, noise_sigma=11.0, rim=175.0)

    labels = np.zeros(shape, np.int32)
    _draw_line(labels, 1, ((8, 45), (245, 45)), 14)
    _draw_line(labels, 2, ((8, 210), (245, 210)), 14)
    add("illumination_drift", "illumination", labels, illumination_amplitude=30.0)

    labels = np.zeros(shape, np.int32)
    _draw_line(labels, 1, ((15, 80), (240, 80)), 8)
    _draw_line(labels, 2, ((15, 145), (240, 145)), 16)
    add("blurred_boundaries", "blur", labels, blur_sigma=2.0, rim=170.0)

    labels = np.zeros(shape, np.int32)
    _draw_line(labels, 1, ((0, 35), (218, 35), (218, 112)), 7)
    _draw_line(labels, 2, ((18, 124), (238, 124)), 17)
    _draw_line(labels, 3, ((18, 143), (238, 143)), 17)
    _draw_rect(labels, 4, (32, 174, 224, 256))
    add(
        "combined_stress",
        "combined",
        labels,
        metal=91.0,
        rim=185.0,
        dark_center=54.0,
        dark_center_depth=9.0,
        noise_sigma=8.0,
        illumination_amplitude=22.0,
        blur_sigma=1.4,
        elevated_regions=((slice(124, 144), slice(18, 238), 101.0),),
    )
    return cases


def _preprocess_real_sem(gray: np.ndarray) -> np.ndarray:
    return preprocess_for_sem(gray, metal_preprocess_config_from_settings(ContourExtractionSettings()))


def _rasterize_polygon_labels(
    polygons: Iterable[PolygonData],
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize each conductor independently, subtracting only its own holes."""
    selected = [
        polygon
        for polygon in polygons
        if polygon.shape_hint != "box" and len(polygon.points) >= 3
    ]
    holes_by_parent: dict[int, list[PolygonData]] = {}
    conductors: list[tuple[float, PolygonData]] = []
    for polygon in selected:
        if polygon.is_hole:
            if polygon.parent_id is not None:
                holes_by_parent.setdefault(int(polygon.parent_id), []).append(polygon)
            continue
        points = np.asarray(polygon.points, dtype=np.int32).reshape(-1, 1, 2)
        conductors.append((abs(float(cv2.contourArea(points))), polygon))

    labels = np.zeros(shape, dtype=np.int32)
    object_id = 0
    for _area, polygon in sorted(conductors, key=lambda item: item[0], reverse=True):
        region = np.zeros(shape, dtype=np.uint8)
        points = np.asarray(polygon.points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(region, [points], 255)
        for hole in holes_by_parent.get(int(polygon.id), ()):
            hole_points = np.asarray(hole.points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(region, [hole_points], 0)
        if not np.any(region):
            continue
        object_id += 1
        labels[region > 0] = object_id
    return labels


def _rasterize_reference_labels(cif_path: Path, shape: tuple[int, int]) -> np.ndarray:
    _image_name, _image_size, polygons = load_polygons_cif(cif_path)
    return _rasterize_polygon_labels(polygons, shape)


def build_real_benchmark_cases(
    dataset_root: Path = REAL_DATASET_ROOT,
) -> list[BenchmarkCase]:
    image_root = dataset_root / "images"
    cif_root = dataset_root / "cif"
    cases: list[BenchmarkCase] = []
    for image_path in sorted(image_root.glob("*.jpg")):
        cif_path = cif_root / f"{image_path.stem}.cif"
        if not cif_path.is_file():
            raise FileNotFoundError(f"Missing CIF pair for {image_path}: {cif_path}")
        raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise ValueError(f"Could not read SEM frame: {image_path}")
        labels = _rasterize_reference_labels(cif_path, raw.shape[:2])
        category = "empty" if not np.any(labels) else "real_sem"
        cases.append(
            BenchmarkCase(
                name=image_path.stem,
                category=category,
                image=_preprocess_real_sem(raw),
                labels=labels,
                source_image=raw,
            )
        )
    if not cases:
        raise FileNotFoundError(f"No JPG/CIF benchmark pairs found under {dataset_root}")
    return cases


def _boundary(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0


def evaluation_crop_slices(
    shape: tuple[int, int],
    crop_px: int,
    *,
    frame_id: str | None = None,
) -> tuple[slice, slice]:
    """Return row/column slices for the evaluation ROI.

    Recognition must already have run on the full image.  A frame that is too
    small for the requested crop fails instead of silently shrinking the ROI.
    """
    if crop_px < 0:
        raise ValueError(f"evaluation_border_crop_px must be >= 0, got {crop_px}")
    height, width = int(shape[0]), int(shape[1])
    if crop_px == 0:
        return slice(None), slice(None)
    if height <= 2 * crop_px or width <= 2 * crop_px:
        frame = f"Frame {frame_id}: " if frame_id else ""
        raise ValueError(
            f"{frame}image size {height}x{width} is too small for "
            f"evaluation_border_crop_px={crop_px}"
        )
    return slice(crop_px, height - crop_px), slice(crop_px, width - crop_px)


def crop_evaluation_region(
    array: np.ndarray,
    crop_px: int,
    *,
    frame_id: str | None = None,
) -> np.ndarray:
    rows, columns = evaluation_crop_slices(array.shape[:2], crop_px, frame_id=frame_id)
    return array[rows, columns]


def relabel_connected_components(mask_or_labels: np.ndarray) -> np.ndarray:
    """Assign fresh 8-connected IDs to a cropped mask.

    Distinct full-frame IDs are not reused.  A remnant that still reaches the
    new virtual border stays a normal component.
    """
    _count, relabeled = cv2.connectedComponents(
        (mask_or_labels > 0).astype(np.uint8),
        connectivity=8,
    )
    return relabeled.astype(np.int32)


def prepare_evaluation_masks(
    predicted: np.ndarray,
    expected_labels: np.ndarray,
    *,
    crop_px: int,
    predicted_labels: np.ndarray | None = None,
    frame_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Crop prediction and GT after recognition, then relabel topology.

    ``crop_px == 0`` preserves the original full-frame masks and IDs.
    """
    predicted_eval = crop_evaluation_region(predicted, crop_px, frame_id=frame_id)
    expected_eval = crop_evaluation_region(expected_labels, crop_px, frame_id=frame_id)
    predicted_labels_eval = (
        None
        if predicted_labels is None
        else crop_evaluation_region(predicted_labels, crop_px, frame_id=frame_id)
    )
    if crop_px == 0:
        return predicted_eval, expected_eval, predicted_labels_eval
    expected_eval = relabel_connected_components(expected_eval)
    if predicted_labels_eval is None:
        predicted_labels_eval = relabel_connected_components(predicted_eval)
    else:
        predicted_labels_eval = relabel_connected_components(predicted_labels_eval)
    return predicted_eval, expected_eval, predicted_labels_eval


def _component_count(labels: np.ndarray) -> int:
    return int(np.unique(labels[labels > 0]).size)


def _component_records(labels: np.ndarray) -> list[dict[str, int | list[int]]]:
    records: list[dict[str, int | list[int]]] = []
    for object_id in np.unique(labels[labels > 0]):
        ys, xs = np.nonzero(labels == object_id)
        records.append(
            {
                "id": int(object_id),
                "area": int(ys.size),
                "bbox_xyxy": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()) + 1,
                    int(ys.max()) + 1,
                ],
            }
        )
    return records


def components_removed_by_crop(
    full_labels: np.ndarray,
    crop_px: int,
    *,
    frame_id: str | None = None,
) -> list[dict[str, int | list[int]]]:
    """Full-frame components whose pixels all lie in the discarded border."""
    if crop_px == 0:
        return []
    remaining = set(
        np.unique(
            crop_evaluation_region(full_labels, crop_px, frame_id=frame_id)
        ).tolist()
    )
    remaining.discard(0)
    return [
        record
        for record in _component_records(full_labels)
        if int(record["id"]) not in remaining
    ]


def labels_for_component_audit(
    mask: np.ndarray,
    labels: np.ndarray | None,
) -> np.ndarray:
    if labels is not None:
        return labels
    return relabel_connected_components(mask)


def _boundary_f1(predicted: np.ndarray, expected: np.ndarray, tolerance_px: int = 2) -> float:
    pred_boundary = _boundary(predicted)
    true_boundary = _boundary(expected)
    if not pred_boundary.any() and not true_boundary.any():
        return 1.0
    kernel_size = 2 * max(0, int(tolerance_px)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    pred_near_true = cv2.dilate(true_boundary.astype(np.uint8), kernel) > 0
    true_near_pred = cv2.dilate(pred_boundary.astype(np.uint8), kernel) > 0
    precision = float(np.count_nonzero(pred_boundary & pred_near_true)) / max(1, int(np.count_nonzero(pred_boundary)))
    recall = float(np.count_nonzero(true_boundary & true_near_pred)) / max(1, int(np.count_nonzero(true_boundary)))
    return 2.0 * precision * recall / max(1e-12, precision + recall)


def _topology_errors(
    predicted: np.ndarray,
    expected_labels: np.ndarray,
    predicted_labels: np.ndarray | None = None,
) -> tuple[int, int, int, int, int, int]:
    if predicted_labels is None:
        _count, predicted_labels = cv2.connectedComponents(
            (predicted > 0).astype(np.uint8),
            connectivity=8,
        )
    predicted_ids = np.unique(predicted_labels[predicted_labels > 0])
    expected_ids = np.unique(expected_labels[expected_labels > 0])
    expected_areas = np.bincount(expected_labels.ravel())
    predicted_areas = np.bincount(predicted_labels.ravel())
    supported_predictions = 0
    false_merges = 0
    for predicted_id in predicted_ids:
        overlaps = expected_labels[predicted_labels == predicted_id]
        object_ids = np.unique(overlaps[overlaps > 0])
        prediction_area = max(1, int(overlaps.size))
        supported_predictions += int(np.count_nonzero(overlaps > 0) / prediction_area >= 0.10)
        materially_covered_objects = 0
        for object_id in object_ids:
            intersection = int(np.count_nonzero(overlaps == object_id))
            expected_area = max(1, int(expected_areas[int(object_id)]))
            if intersection / expected_area >= 0.10:
                materially_covered_objects += 1
        false_merges += max(0, materially_covered_objects - 1)

    false_splits = 0
    matched_expected = 0
    for object_id in expected_ids:
        overlaps = predicted_labels[expected_labels == object_id]
        expected_area = max(1, int(overlaps.size))
        matched_expected += int(np.count_nonzero(overlaps > 0) / expected_area >= 0.50)
        component_ids = np.unique(overlaps[overlaps > 0])
        material_overlaps = 0
        for component_id in component_ids:
            intersection = int(np.count_nonzero(overlaps == component_id))
            prediction_area = max(1, int(predicted_areas[int(component_id)]))
            if intersection / expected_area >= 0.01 and intersection / prediction_area >= 0.10:
                material_overlaps += 1
        false_splits += max(0, material_overlaps - 1)
    predicted_components = int(predicted_ids.size)
    false_positive_components = predicted_components - supported_predictions
    missed_expected_components = int(expected_ids.size) - matched_expected
    return (
        predicted_components,
        false_merges,
        false_splits,
        false_positive_components,
        missed_expected_components,
        matched_expected,
    )


def measure_segmentation(
    predicted: np.ndarray,
    expected_labels: np.ndarray,
    *,
    elapsed_ms: float,
    segmentation_ms: float | None = None,
    refinement_ms: float = 0.0,
    predicted_labels: np.ndarray | None = None,
) -> SegmentationMetrics:
    predicted_active = predicted > 0
    expected_active = expected_labels > 0
    true_positive = int(np.count_nonzero(predicted_active & expected_active))
    false_positive = int(np.count_nonzero(predicted_active & ~expected_active))
    false_negative = int(np.count_nonzero(~predicted_active & expected_active))
    total_pixels = max(1, int(predicted_active.size))
    expected_pixels = int(np.count_nonzero(expected_active))
    predicted_pixels = int(np.count_nonzero(predicted_active))
    union = true_positive + false_positive + false_negative
    (
        predicted_components,
        false_merges,
        false_splits,
        false_positive_components,
        missed_expected_components,
        matched_expected_components,
    ) = _topology_errors(predicted, expected_labels, predicted_labels)
    supported_components = max(0, predicted_components - false_positive_components)
    expected_components = int(np.unique(expected_labels[expected_labels > 0]).size)
    component_precision = (
        1.0 if predicted_components == 0 and expected_components == 0 else supported_components / max(1, predicted_components)
    )
    component_recall = (
        1.0 if expected_components == 0 else matched_expected_components / expected_components
    )
    component_f1 = (
        0.0
        if component_precision + component_recall <= 0.0
        else 2.0 * component_precision * component_recall / (component_precision + component_recall)
    )
    topology_exact_match = (
        false_positive_components == 0
        and missed_expected_components == 0
        and false_merges == 0
        and false_splits == 0
        and predicted_components == expected_components
    )
    return SegmentationMetrics(
        iou=1.0 if union == 0 else true_positive / union,
        precision=(1.0 if predicted_pixels == 0 and expected_pixels == 0 else true_positive / max(1, predicted_pixels)),
        recall=(1.0 if expected_pixels == 0 and predicted_pixels == 0 else true_positive / max(1, expected_pixels)),
        boundary_f1=_boundary_f1(predicted, expected_active.astype(np.uint8) * 255),
        expected_components=expected_components,
        predicted_components=predicted_components,
        false_merges=false_merges,
        false_splits=false_splits,
        false_positive_components=false_positive_components,
        missed_expected_components=missed_expected_components,
        matched_expected_components=matched_expected_components,
        component_count_absolute_error=abs(predicted_components - expected_components),
        component_precision=component_precision,
        component_recall=component_recall,
        component_f1=component_f1,
        false_merge_indicator=false_merges > 0,
        false_split_indicator=false_splits > 0,
        predicted_foreground_fraction=predicted_pixels / total_pixels,
        gt_foreground_fraction=expected_pixels / total_pixels,
        false_positive_area_fraction=false_positive / total_pixels,
        false_negative_area_fraction=false_negative / total_pixels,
        false_metal_fraction=predicted_pixels / total_pixels if expected_pixels == 0 else 0.0,
        segmentation_ms=float(elapsed_ms if segmentation_ms is None else segmentation_ms),
        refinement_ms=float(refinement_ms),
        elapsed_ms=float(elapsed_ms),
        topology_exact_match=topology_exact_match,
    )


def seed_diagnostics(
    image: np.ndarray,
    config: GradientWatershedConfig,
    *,
    check_presence: bool = True,
) -> SeedDiagnostics:
    smoothed = cv2.GaussianBlur(
        image,
        (0, 0),
        max(0.1, float(config.smoothing_sigma)),
    )
    substrate_limit, metal_limit = intensity_class_limits(smoothed)
    seeds = build_conductor_seeds(image, config, check_presence=check_presence)
    if seeds is None:
        core = np.zeros(image.shape, dtype=np.uint8)
        groove = np.zeros(image.shape, dtype=np.uint8)
    else:
        core = seeds.core_seeds
        groove = seeds.groove_seeds
    core_count, _core_labels = cv2.connectedComponents(
        (core > 0).astype(np.uint8),
        connectivity=8,
    )
    groove_count, _groove_labels = cv2.connectedComponents(
        (groove > 0).astype(np.uint8),
        connectivity=8,
    )
    gradient_x = cv2.Scharr(smoothed, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Scharr(smoothed, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    local_contrast = cv2.absdiff(
        smoothed,
        cv2.GaussianBlur(smoothed, (0, 0), 6.0),
    )
    intensity_percentiles = np.percentile(smoothed, (1, 10, 50, 90, 99))
    gradient_percentiles = np.percentile(gradient, (90, 99))
    contrast_percentiles = np.percentile(local_contrast, (90, 99))
    presence = analyze_metal_presence(
        image,
        smoothing_sigma=float(config.smoothing_sigma),
    )
    return SeedDiagnostics(
        substrate_intensity_limit=float(substrate_limit),
        metal_intensity_limit=float(metal_limit),
        core_seed_fraction=float(np.mean(core > 0)),
        groove_seed_fraction=float(np.mean(groove > 0)),
        core_connected_components=core_count - 1,
        groove_connected_components=groove_count - 1,
        intensity_p01=float(intensity_percentiles[0]),
        intensity_p10=float(intensity_percentiles[1]),
        intensity_p50=float(intensity_percentiles[2]),
        intensity_p90=float(intensity_percentiles[3]),
        intensity_p99=float(intensity_percentiles[4]),
        gradient_mean=float(np.mean(gradient)),
        gradient_p90=float(gradient_percentiles[0]),
        gradient_p99=float(gradient_percentiles[1]),
        local_contrast_mean=float(np.mean(local_contrast)),
        local_contrast_p90=float(contrast_percentiles[0]),
        local_contrast_p99=float(contrast_percentiles[1]),
        presence_has_metal=presence.has_metal,
        presence_robust_intensity_span=presence.robust_intensity_span,
        presence_coherent_contrast_fraction=presence.coherent_contrast_fraction,
        presence_largest_coherent_contrast_fraction=(presence.largest_coherent_contrast_fraction),
        presence_local_contrast_limit=presence.local_contrast_limit,
    )


def resolution_probe(source_width: int = 2000, working_width: int = 640) -> list[dict[str, float | int | str]]:
    results: list[dict[str, float | int | str]] = []
    for feature in ("gap", "conductor"):
        background, foreground = (255, 0) if feature == "gap" else (0, 255)
        for width_px in range(1, 9):
            coverage: list[float] = []
            vanished = 0
            for phase in range(10):
                mask = np.full((32, source_width), background, np.uint8)
                start = source_width // 2 + phase
                mask[:, start : start + width_px] = foreground
                working_height = max(1, round(mask.shape[0] * working_width / source_width))
                reduced = cv2.resize(mask, (working_width, working_height), interpolation=cv2.INTER_NEAREST)
                restored = cv2.resize(reduced, (source_width, mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                retained = float(np.mean(restored[:, start : start + width_px] == foreground))
                coverage.append(retained)
                vanished += int(retained == 0.0)
            results.append(
                {
                    "feature": feature,
                    "width_px": width_px,
                    "mean_retained_fraction": float(np.mean(coverage)),
                    "vanished_phases_of_10": vanished,
                }
            )
    return results


def _write_diagnostic_overlay(
    output_path: Path,
    case: BenchmarkCase,
    predicted: np.ndarray,
    config: GradientWatershedConfig,
    *,
    check_presence: bool,
) -> None:
    expected = case.labels > 0
    seeds = build_conductor_seeds(
        case.image,
        config,
        check_presence=check_presence,
    )
    core = np.zeros(case.image.shape, dtype=np.uint8) if seeds is None else seeds.core_seeds
    groove = np.zeros(case.image.shape, dtype=np.uint8) if seeds is None else seeds.groove_seeds
    false_negative = expected & (predicted == 0)
    false_positive = ~expected & (predicted > 0)
    base = cv2.cvtColor(case.image, cv2.COLOR_GRAY2BGR)
    comparison = base.copy()
    comparison[expected] = (0, 170, 0)
    comparison[predicted > 0] = (0, 0, 220)
    seeds_overlay = base.copy()
    seeds_overlay[core > 0] = (255, 0, 255)
    seeds_overlay[groove > 0] = (255, 255, 0)
    errors = base.copy()
    errors[false_negative] = (255, 0, 0)
    errors[false_positive] = (0, 0, 255)
    panel = np.hstack((base, comparison, seeds_overlay, errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), panel):
        raise OSError(f"Could not write diagnostic overlay: {output_path}")


def _rasterize_detected_polygons(
    polygons: Iterable[PolygonData],
    shape: tuple[int, int],
) -> np.ndarray:
    return _rasterize_polygon_labels(polygons, shape)


def _ui_recovery_config(strategy: str, config: GradientWatershedConfig):
    base = ContourExtractionSettings(
        recognition_mode="conductors",
        extraction_profile="conductors",
        object_type="conductor",
        output_mode="polygon",
        algorithm_backend="legacy",
        metal_structural_pipeline=True,
    )
    settings = ContourExtractionSettings.from_dict(
        {
            **base.to_dict(),
            **standard_metal_preset_payload(),
            "metal_segmentation_strategy": strategy,
            "metal_use_wide_conductor_gradient": strategy == "gradient_watershed",
            "metal_watershed_smoothing_sigma": config.smoothing_sigma,
            "metal_watershed_core_margin": config.core_margin,
            "metal_watershed_groove_margin": config.groove_margin,
            "metal_watershed_rim_probe_px": config.rim_probe_px,
            "metal_watershed_seed_speckle_px": config.seed_speckle_px,
            "metal_watershed_valley_span_px": config.valley_span_px,
            "metal_watershed_valley_depth": config.valley_depth,
            "metal_random_walker_beta": config.random_walker_beta,
            "metal_random_walker_iterations": config.random_walker_iterations,
            "metal_graph_cut_iterations": config.graph_cut_iterations,
            "metal_reconstruction_erode_px": config.reconstruction_erode_px,
            "metal_boundary_relief": config.boundary_relief,
            "metal_boundary_background_sigma": config.boundary_background_sigma,
        }
    )
    return metal_recovery_config_from_settings(settings)


def _aggregate_strategy_metrics(
    case_results: dict[str, dict[str, dict[str, object]]],
    case_categories: dict[str, str],
    strategy_names: tuple[str, ...],
) -> dict[str, dict[str, float | int]]:
    aggregates: dict[str, dict[str, float | int]] = {}
    for strategy in strategy_names:
        metrics = [case_results[name][strategy] for name in case_results]
        positive_metrics = [
            case_results[name][strategy]
            for name in case_results
            if case_categories[name] != "empty"
        ]
        empty_metrics = [
            case_results[name][strategy]
            for name in case_results
            if case_categories[name] == "empty"
        ]
        aggregates[strategy] = {
            "mean_iou": float(np.mean([item["iou"] for item in metrics])),
            "positive_mean_iou": float(np.mean([item["iou"] for item in positive_metrics])),
            "positive_median_iou": float(np.median([item["iou"] for item in positive_metrics])),
            "mean_precision": float(np.mean([item["precision"] for item in metrics])),
            "positive_mean_precision": float(np.mean([item["precision"] for item in positive_metrics])),
            "mean_recall": float(np.mean([item["recall"] for item in metrics])),
            "positive_mean_recall": float(np.mean([item["recall"] for item in positive_metrics])),
            "mean_boundary_f1": float(np.mean([item["boundary_f1"] for item in metrics])),
            "positive_mean_boundary_f1": float(np.mean([item["boundary_f1"] for item in positive_metrics])),
            "false_merges": int(sum(item["false_merges"] for item in metrics)),
            "false_splits": int(sum(item["false_splits"] for item in metrics)),
            "false_positive_components": int(
                sum(item["false_positive_components"] for item in metrics)
            ),
            "missed_expected_components": int(
                sum(item["missed_expected_components"] for item in metrics)
            ),
            "component_count_absolute_error": int(
                sum(item["component_count_absolute_error"] for item in metrics)
            ),
            "positive_mean_component_precision": float(
                np.mean([item["component_precision"] for item in positive_metrics])
            ),
            "positive_mean_component_recall": float(
                np.mean([item["component_recall"] for item in positive_metrics])
            ),
            "positive_mean_component_f1": float(
                np.mean([item["component_f1"] for item in positive_metrics])
            ),
            "empty_false_metal_fraction": float(
                np.mean([item["false_metal_fraction"] for item in empty_metrics]) if empty_metrics else 0.0
            ),
            "exact_topology_frames": int(sum(bool(item["topology_exact_match"]) for item in metrics)),
            "positive_exact_topology_frames": int(
                sum(bool(item["topology_exact_match"]) for item in positive_metrics)
            ),
            "elapsed_ms": float(sum(item["elapsed_ms"] for item in metrics)),
            "mean_elapsed_ms": float(np.mean([item["elapsed_ms"] for item in metrics])),
            "mean_segmentation_ms": float(np.mean([item["segmentation_ms"] for item in metrics])),
            "mean_refinement_ms": float(np.mean([item["refinement_ms"] for item in metrics])),
        }
    return aggregates


def _evaluation_record(
    metrics: SegmentationMetrics,
    *,
    original_shape: tuple[int, int],
    evaluation_shape: tuple[int, int],
    full_metrics: SegmentationMetrics,
    removed_gt_component_count: int,
    removed_predicted_component_count: int,
) -> dict[str, float | int | bool]:
    original_height, original_width = original_shape
    evaluation_height, evaluation_width = evaluation_shape
    return {
        **asdict(metrics),
        "original_height": int(original_height),
        "original_width": int(original_width),
        "evaluation_height": int(evaluation_height),
        "evaluation_width": int(evaluation_width),
        "expected_components_full_frame": full_metrics.expected_components,
        "predicted_components_full_frame": full_metrics.predicted_components,
        "removed_gt_component_count": int(removed_gt_component_count),
        "removed_predicted_component_count": int(removed_predicted_component_count),
        "iou_full_frame": full_metrics.iou,
        "false_positive_components_full_frame": full_metrics.false_positive_components,
        "missed_expected_components_full_frame": full_metrics.missed_expected_components,
        "false_merges_full_frame": full_metrics.false_merges,
        "false_splits_full_frame": full_metrics.false_splits,
        "component_count_absolute_error_full_frame": full_metrics.component_count_absolute_error,
        "topology_exact_match_full_frame": full_metrics.topology_exact_match,
    }


def run_benchmark(
    strategies: Iterable[str] = STRATEGIES,
    *,
    cases: Iterable[BenchmarkCase] | None = None,
    suite: str = "synthetic",
    diagnostics_dir: Path | None = None,
    check_presence: bool = True,
    evaluation_stage: str = "ui",
    evaluation_border_crop_px: int = 0,
) -> dict[str, object]:
    config = GradientWatershedConfig()
    case_results: dict[str, dict[str, dict[str, object]]] = {}
    full_frame_case_results: dict[str, dict[str, dict[str, object]]] = {}
    case_categories: dict[str, str] = {}
    seed_results: dict[str, dict[str, float | int | bool]] = {}
    border_audit: dict[str, dict[str, object]] = {}
    strategy_names = tuple(strategies)
    selected_cases = tuple(build_benchmark_cases() if cases is None else cases)
    crop_px = int(evaluation_border_crop_px)
    for case in selected_cases:
        evaluation_crop_slices(case.image.shape[:2], crop_px, frame_id=case.name)
        case_categories[case.name] = case.category
        seed_results[case.name] = asdict(seed_diagnostics(case.image, config, check_presence=check_presence))
        per_strategy: dict[str, dict[str, object]] = {}
        per_strategy_full: dict[str, dict[str, object]] = {}
        for strategy in strategy_names:
            started = perf_counter()
            predicted_labels: np.ndarray | None = None
            if evaluation_stage == "ui":
                detection = detect_metalization(
                    case.image,
                    _ui_recovery_config(strategy, config),
                    source_image=case.source_image,
                )
                predicted_labels = _rasterize_detected_polygons(
                    detection.accepted,
                    case.image.shape[:2],
                )
                mask = np.where(predicted_labels > 0, 255, 0).astype(np.uint8)
                segmentation_ms = (perf_counter() - started) * 1000.0
                refinement_ms = 0.0
            elif strategy == "gradient_watershed":
                seeds = build_conductor_seeds(
                    case.image,
                    config,
                    check_presence=check_presence,
                )
                if seeds is None:
                    baseline_mask = np.zeros(case.image.shape, dtype=np.uint8)
                else:
                    baseline_mask = gradient_watershed_mask_from_seeds(case.image, seeds)
                segmentation_ms = (perf_counter() - started) * 1000.0
                refinement_started = perf_counter()
                mask = (
                    selective_conductor_recovery(baseline_mask, seeds, config)
                    if check_presence and seeds is not None
                    else baseline_mask
                )
                refinement_ms = (perf_counter() - refinement_started) * 1000.0
            else:
                mask, _polarity, _selected = _segment(
                    case.image,
                    strategy=strategy,
                    watershed_config=config,
                )
                segmentation_ms = (perf_counter() - started) * 1000.0
                refinement_ms = 0.0
            elapsed_ms = (perf_counter() - started) * 1000.0
            full_metrics = measure_segmentation(
                mask,
                case.labels,
                elapsed_ms=elapsed_ms,
                segmentation_ms=segmentation_ms,
                refinement_ms=refinement_ms,
                predicted_labels=predicted_labels,
            )
            predicted_eval, expected_eval, predicted_labels_eval = prepare_evaluation_masks(
                mask,
                case.labels,
                crop_px=crop_px,
                predicted_labels=predicted_labels,
                frame_id=case.name,
            )
            eval_metrics = (
                full_metrics
                if crop_px == 0
                else measure_segmentation(
                    predicted_eval,
                    expected_eval,
                    elapsed_ms=elapsed_ms,
                    segmentation_ms=segmentation_ms,
                    refinement_ms=refinement_ms,
                    predicted_labels=predicted_labels_eval,
                )
            )
            predicted_audit_labels = labels_for_component_audit(mask, predicted_labels)
            removed_gt = components_removed_by_crop(
                case.labels,
                crop_px,
                frame_id=case.name,
            )
            removed_predicted = components_removed_by_crop(
                predicted_audit_labels,
                crop_px,
                frame_id=case.name,
            )
            per_strategy[strategy] = _evaluation_record(
                eval_metrics,
                original_shape=case.image.shape[:2],
                evaluation_shape=predicted_eval.shape[:2],
                full_metrics=full_metrics,
                removed_gt_component_count=len(removed_gt),
                removed_predicted_component_count=len(removed_predicted),
            )
            per_strategy_full[strategy] = asdict(full_metrics)
            if case.name in BORDER_AUDIT_FRAMES:
                border_audit.setdefault(case.name, {})[strategy] = {
                    "removed_gt_components": removed_gt,
                    "removed_predicted_components": removed_predicted,
                    "remaining": {
                        "false_positive_components": eval_metrics.false_positive_components,
                        "missed_expected_components": eval_metrics.missed_expected_components,
                        "false_merges": eval_metrics.false_merges,
                        "false_splits": eval_metrics.false_splits,
                        "component_count_absolute_error": eval_metrics.component_count_absolute_error,
                        "expected_components": eval_metrics.expected_components,
                        "predicted_components": eval_metrics.predicted_components,
                        "topology_exact_match": eval_metrics.topology_exact_match,
                    },
                }
            if diagnostics_dir is not None and strategy == "gradient_watershed":
                _write_diagnostic_overlay(
                    diagnostics_dir / f"{case.name}.png",
                    case,
                    mask,
                    config,
                    check_presence=check_presence,
                )
        case_results[case.name] = per_strategy
        full_frame_case_results[case.name] = per_strategy_full

    aggregates = _aggregate_strategy_metrics(case_results, case_categories, strategy_names)
    exact_topology_frames = {
        strategy: [
            name
            for name, per_strategy in case_results.items()
            if bool(per_strategy[strategy]["topology_exact_match"])
        ]
        for strategy in strategy_names
    }
    return {
        "schema_version": 4,
        "suite": suite,
        "evaluation_stage": evaluation_stage,
        "evaluation_border_crop_px": crop_px,
        "case_count": len(case_results),
        "positive_case_count": sum(category != "empty" for category in case_categories.values()),
        "empty_case_count": sum(category == "empty" for category in case_categories.values()),
        "strategies": list(strategy_names),
        "config": asdict(config),
        "recovery_configs": {
            strategy: asdict(_ui_recovery_config(strategy, config))
            for strategy in strategy_names
        },
        "preprocess_config": asdict(
            metal_preprocess_config_from_settings(ContourExtractionSettings())
        ),
        "metal_presence_check": check_presence,
        "aggregates": aggregates,
        "full_frame_aggregates": _aggregate_strategy_metrics(
            full_frame_case_results,
            case_categories,
            strategy_names,
        ),
        "exact_topology_frames": exact_topology_frames,
        "border_audit": border_audit,
        "case_categories": case_categories,
        "seed_diagnostics": seed_results,
        "cases": case_results,
        "resolution_probe": resolution_probe(),
    }


def _write_csv(report: dict[str, object], output_path: Path) -> None:
    cases = report["cases"]
    diagnostics = report["seed_diagnostics"]
    categories = report["case_categories"]
    assert isinstance(cases, dict)
    assert isinstance(diagnostics, dict)
    assert isinstance(categories, dict)
    rows: list[dict[str, object]] = []
    for case_name, raw_strategies in cases.items():
        assert isinstance(raw_strategies, dict)
        raw_diagnostics = diagnostics[case_name]
        assert isinstance(raw_diagnostics, dict)
        for strategy, raw_metrics in raw_strategies.items():
            assert isinstance(raw_metrics, dict)
            scalar_metrics = {
                key: value
                for key, value in raw_metrics.items()
                if not isinstance(value, (dict, list, tuple))
            }
            rows.append(
                {
                    "case": case_name,
                    "category": categories[case_name],
                    "strategy": strategy,
                    "evaluation_border_crop_px": report.get("evaluation_border_crop_px", 0),
                    **scalar_metrics,
                    **raw_diagnostics,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value: object, digits: int = 3) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(report: dict[str, object]) -> str:
    aggregates = report["aggregates"]
    assert isinstance(aggregates, dict)
    crop_px = int(report.get("evaluation_border_crop_px", 0) or 0)
    lines = [
        f"evaluation_border_crop_px = {crop_px}",
        "",
        "| strategy | mean IoU | boundary F1 | component F1 | false components | missed objects | count error | merges | splits | exact topology | empty false metal | mean ms/frame |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, raw_metrics in aggregates.items():
        assert isinstance(raw_metrics, dict)
        lines.append(
            f"| {strategy} | {raw_metrics['positive_mean_iou']:.3f} | "
            f"{raw_metrics['positive_mean_boundary_f1']:.3f} | "
            f"{raw_metrics['positive_mean_component_f1']:.3f} | "
            f"{raw_metrics['false_positive_components']} | "
            f"{raw_metrics['missed_expected_components']} | "
            f"{raw_metrics['component_count_absolute_error']} | "
            f"{raw_metrics['false_merges']} | {raw_metrics['false_splits']} | "
            f"{raw_metrics['exact_topology_frames']} | "
            f"{raw_metrics['empty_false_metal_fraction']:.4f} | "
            f"{raw_metrics['mean_elapsed_ms']:.1f} |"
        )

    cases = report["cases"]
    exact_frames = report.get("exact_topology_frames", {})
    assert isinstance(cases, dict)
    strategy_names = [str(name) for name in report.get("strategies", ())]
    if len(strategy_names) == 1:
        strategy = strategy_names[0]
        lines.extend(
            [
                "",
                f"Per-frame evaluation ({strategy}):",
                "",
                "| frame | original | eval size | GT full | GT crop | pred crop | IoU | precision | recall | Boundary F1 | false | misses | merges | splits | count err | exact |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for frame_id, per_strategy in cases.items():
            assert isinstance(per_strategy, dict)
            raw_metrics = per_strategy[strategy]
            assert isinstance(raw_metrics, dict)
            original = f"{raw_metrics['original_height']}x{raw_metrics['original_width']}"
            evaluation = f"{raw_metrics['evaluation_height']}x{raw_metrics['evaluation_width']}"
            lines.append(
                f"| {frame_id} | {original} | {evaluation} | "
                f"{raw_metrics['expected_components_full_frame']} | "
                f"{raw_metrics['expected_components']} | "
                f"{raw_metrics['predicted_components']} | "
                f"{raw_metrics['iou']:.3f} | {raw_metrics['precision']:.3f} | "
                f"{raw_metrics['recall']:.3f} | {raw_metrics['boundary_f1']:.3f} | "
                f"{raw_metrics['false_positive_components']} | "
                f"{raw_metrics['missed_expected_components']} | "
                f"{raw_metrics['false_merges']} | {raw_metrics['false_splits']} | "
                f"{raw_metrics['component_count_absolute_error']} | "
                f"{'yes' if raw_metrics['topology_exact_match'] else 'no'} |"
            )
        matched = exact_frames.get(strategy, []) if isinstance(exact_frames, dict) else []
        lines.extend(["", f"Exact topology frames: {', '.join(str(item) for item in matched) or '(none)'}"])

    full_frame_aggregates = report.get("full_frame_aggregates")
    if crop_px > 0 and isinstance(full_frame_aggregates, dict) and len(strategy_names) == 1:
        strategy = strategy_names[0]
        old_metrics = full_frame_aggregates[strategy]
        new_metrics = aggregates[strategy]
        assert isinstance(old_metrics, dict)
        assert isinstance(new_metrics, dict)
        rows = (
            ("Mean IoU", "positive_mean_iou", 3),
            ("Median IoU", "positive_median_iou", 3),
            ("Boundary F1", "positive_mean_boundary_f1", 3),
            ("False components", "false_positive_components", 0),
            ("Misses", "missed_expected_components", 0),
            ("Count error", "component_count_absolute_error", 0),
            ("Merges", "false_merges", 0),
            ("Splits", "false_splits", 0),
            ("Exact topology frames", "exact_topology_frames", 0),
        )
        lines.extend(
            [
                "",
                f"Metric                     old full-frame    new crop-{crop_px}",
                "--------------------------------------------------------",
            ]
        )
        for label, key, digits in rows:
            old_value = _format_metric(old_metrics[key], digits)
            new_value = _format_metric(new_metrics[key], digits)
            lines.append(f"{label:<26} {old_value:>16} {new_value:>14}")

    border_audit = report.get("border_audit", {})
    if isinstance(border_audit, dict) and border_audit:
        lines.extend(["", "Border-frame audit:"])
        for frame_id in BORDER_AUDIT_FRAMES:
            frame_audit = border_audit.get(frame_id)
            if not isinstance(frame_audit, dict):
                continue
            for strategy, payload in frame_audit.items():
                assert isinstance(payload, dict)
                removed_gt = payload["removed_gt_components"]
                removed_pred = payload["removed_predicted_components"]
                remaining = payload["remaining"]
                assert isinstance(removed_gt, list)
                assert isinstance(removed_pred, list)
                assert isinstance(remaining, dict)
                lines.extend(
                    [
                        "",
                        f"### {frame_id} ({strategy})",
                        f"GT components removed by crop: {len(removed_gt)}",
                    ]
                )
                for record in removed_gt:
                    lines.append(f"- GT id={record['id']} area={record['area']} bbox={record['bbox_xyxy']}")
                lines.append(f"Predicted components removed by crop: {len(removed_pred)}")
                for record in removed_pred:
                    lines.append(
                        f"- pred id={record['id']} area={record['area']} bbox={record['bbox_xyxy']}"
                    )
                lines.append(
                    "Remaining topology: "
                    f"false={remaining['false_positive_components']}, "
                    f"misses={remaining['missed_expected_components']}, "
                    f"merges={remaining['false_merges']}, "
                    f"splits={remaining['false_splits']}, "
                    f"count_error={remaining['component_count_absolute_error']}, "
                    f"pred/gt={remaining['predicted_components']}/{remaining['expected_components']}, "
                    f"exact={'yes' if remaining['topology_exact_match'] else 'no'}"
                )
    return "\n".join(lines)


def _default_evaluation_border_crop_px(suite: str, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    return EVALUATION_BORDER_CROP_PX if suite == "real" else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    parser.add_argument("--output", type=Path, help="write the full JSON report to this path")
    parser.add_argument("--csv", type=Path, help="write one flattened row per frame and strategy")
    parser.add_argument(
        "--suite",
        choices=("synthetic", "real"),
        default="synthetic",
        help="benchmark deterministic synthetic scenes or tests/test_metal JPG/CIF pairs",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REAL_DATASET_ROOT,
        help="root containing images/ and cif/ for --suite real",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        help="write gradient-watershed GT/prediction/seed/error overlays",
    )
    parser.add_argument(
        "--skip-presence-check",
        action="store_true",
        help="benchmark the pre-presence-rejection gradient-watershed baseline",
    )
    parser.add_argument(
        "--evaluation-stage",
        choices=("ui", "segmentation"),
        default="ui",
        help="score final UI polygons by default, or the raw segmentation mask for diagnostics",
    )
    parser.add_argument(
        "--evaluation-border-crop-px",
        type=int,
        default=None,
        help=(
            "Discard this many pixels on each side after recognition, before metrics. "
            f"Default: {EVALUATION_BORDER_CROP_PX} for --suite real, 0 for synthetic. "
            "Use 0 to restore full-frame scoring."
        ),
    )
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    args = parser.parse_args()
    cases = build_real_benchmark_cases(args.dataset_root) if args.suite == "real" else build_benchmark_cases()
    report = run_benchmark(
        args.strategies,
        cases=cases,
        suite=args.suite,
        diagnostics_dir=args.diagnostics_dir,
        check_presence=not args.skip_presence_check,
        evaluation_stage=args.evaluation_stage,
        evaluation_border_crop_px=_default_evaluation_border_crop_px(
            args.suite,
            args.evaluation_border_crop_px,
        ),
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.csv is not None:
        _write_csv(report, args.csv)
    print(json.dumps(report, indent=2) if args.json else _markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
