"""benchmark_metal_segmentation: rasterize polygon labels."""
from __future__ import annotations
from collections.abc import Iterable
from pathlib import Path
import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings  # noqa: E402
from contour.domain import PolygonData  # noqa: E402
from contour.serializers import load_polygons_cif  # noqa: E402
from contour.vision.preprocessing import (  # noqa: E402
    metal_preprocess_config_from_settings,
    preprocess_for_sem,
)

from benchmark_metal_segmentation_parts.render_sem import BenchmarkCase

def _preprocess_real_sem(gray: np.ndarray) -> np.ndarray:
    return preprocess_for_sem(gray, metal_preprocess_config_from_settings(ContourExtractionSettings()))

def _rasterize_polygon_labels(
    polygons: Iterable[PolygonData],
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize each conductor independently, subtracting only its own holes."""
    selected = [polygon for polygon in polygons if polygon.shape_hint != "box" and len(polygon.points) >= 3]
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
        raise ValueError(f"{frame}image size {height}x{width} is too small for evaluation_border_crop_px={crop_px}")
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
        None if predicted_labels is None else crop_evaluation_region(predicted_labels, crop_px, frame_id=frame_id)
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
    remaining = set(np.unique(crop_evaluation_region(full_labels, crop_px, frame_id=frame_id)).tolist())
    remaining.discard(0)
    return [record for record in _component_records(full_labels) if int(record["id"]) not in remaining]

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
