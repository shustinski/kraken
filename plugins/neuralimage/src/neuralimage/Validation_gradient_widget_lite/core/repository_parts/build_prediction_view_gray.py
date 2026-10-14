"""repository: build prediction view gray."""
from __future__ import annotations
import numpy as np

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import EPS

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.available_memory_bytes import (
    _PredictionPoint,
    _PredictionRegionSummary,
    _PredictionView,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.extract_patch import _extract_patch

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import (
    _boundary_mask,
    _label_components,
)

def build_prediction_view_from_gray(model_name: str, pred_gray: np.ndarray, *, threshold: int = 128) -> _PredictionView:
    """Build lightweight prediction view from one grayscale output."""

    grayscale = np.asarray(pred_gray, dtype=np.uint8)
    pred_bin = grayscale >= int(threshold)
    labels, count = _label_components(pred_bin)
    points: list[_PredictionPoint] = []
    areas: list[int] = []
    for label in range(1, int(count) + 1):
        ys, xs = np.where(labels == label)
        if xs.size == 0:
            continue
        area = int(xs.size)
        areas.append(area)
        centroid_x = float(np.mean(xs, dtype=np.float64))
        centroid_y = float(np.mean(ys, dtype=np.float64))
        px = int(np.clip(round(centroid_x), 0, grayscale.shape[1] - 1))
        py = int(np.clip(round(centroid_y), 0, grayscale.shape[0] - 1))
        patch = _extract_patch(grayscale, centroid_x, centroid_y)
        peak = float(grayscale[py, px] / 255.0)
        patch_mean = float(patch.mean(dtype=np.float32) / 255.0) if patch.size else peak
        patch_std = float(patch.std(dtype=np.float32) / 255.0) if patch.size else 0.0
        local_contrast = float(max(0.0, peak - patch_mean))
        local_snr = float(local_contrast / max(1e-6, patch_std))
        radius = float(np.sqrt(area / np.pi))
        boundary = _boundary_mask(labels == label)
        perimeter = float(max(1, np.count_nonzero(boundary)))
        compactness = float(np.clip((4.0 * np.pi * area) / max(EPS, perimeter * perimeter), 0.0, 1.0))
        points.append(
            _PredictionPoint(
                x=centroid_x,
                y=centroid_y,
                score=peak,
                peak_intensity=peak * 255.0,
                local_contrast=local_contrast,
                blob_score=compactness,
                local_snr=local_snr,
                radius=radius,
                spot_area=float(area),
            )
        )
    region_summary = _PredictionRegionSummary(
        area_fraction=float(np.count_nonzero(pred_bin) / max(1, pred_bin.size)),
        mean_area=float(np.mean(np.asarray(areas, dtype=np.float64))) if areas else 0.0,
    )
    return _PredictionView(
        model_name=str(model_name),
        pred_gray=grayscale,
        pred_bin=np.asarray(pred_bin, dtype=bool),
        points=tuple(points),
        region_summary=region_summary,
    )
