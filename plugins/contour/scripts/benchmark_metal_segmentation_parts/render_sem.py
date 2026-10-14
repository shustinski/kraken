"""benchmark_metal_segmentation: render sem."""
from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
import cv2
import numpy as np

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
