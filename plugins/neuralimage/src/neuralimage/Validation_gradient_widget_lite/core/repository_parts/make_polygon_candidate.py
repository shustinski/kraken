"""repository: make polygon candidate."""
from __future__ import annotations
import numpy as np

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import EPS

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_confidence_config import _PolygonConfidenceCandidate

def _make_polygon_candidate(
    probability: np.ndarray,
    mask: np.ndarray,
    source_branches: tuple[str, ...],
    *,
    roi_bbox: tuple[int, int, int, int] | None = None,
) -> _PolygonConfidenceCandidate | None:
    """Build one candidate while optionally restricting geometry extraction to a known ROI."""

    mask_bool = np.asarray(mask, dtype=bool)
    if roi_bbox is not None:
        roi_x0, roi_y0, roi_width, roi_height = roi_bbox
        roi_x0 = max(0, int(roi_x0))
        roi_y0 = max(0, int(roi_y0))
        roi_x1 = min(mask_bool.shape[1], roi_x0 + max(0, int(roi_width)))
        roi_y1 = min(mask_bool.shape[0], roi_y0 + max(0, int(roi_height)))
        roi_mask = mask_bool[roi_y0:roi_y1, roi_x0:roi_x1]
    else:
        roi_x0 = 0
        roi_y0 = 0
        roi_mask = mask_bool

    area = int(np.count_nonzero(roi_mask))
    if area <= 0:
        return None

    ys, xs = np.nonzero(roi_mask)
    if ys.size == 0 or xs.size == 0:
        return None
    bbox_x0 = roi_x0 + int(np.min(xs))
    bbox_y0 = roi_y0 + int(np.min(ys))
    bbox_x1 = roi_x0 + int(np.max(xs)) + 1
    bbox_y1 = roi_y0 + int(np.max(ys)) + 1
    bbox = (bbox_x0, bbox_y0, bbox_x1 - bbox_x0, bbox_y1 - bbox_y0)
    width = max(1, int(bbox[2]))
    height = max(1, int(bbox[3]))
    aspect_ratio = float(max(width, height) / max(1, min(width, height)))
    extent = float(area / max(1, width * height))
    elongation = aspect_ratio
    if ys.size >= 3:
        coords = np.column_stack((ys.astype(np.float32), xs.astype(np.float32)))
        centered = coords - np.mean(coords, axis=0, keepdims=True)
        covariance = np.matmul(centered.T, centered) / max(1.0, float(coords.shape[0] - 1))
        eigvals = np.linalg.eigvalsh(covariance)
        if eigvals.size >= 2:
            major = float(max(eigvals[-1], EPS))
            minor = float(max(eigvals[0], EPS))
            elongation = float(np.sqrt(major / minor))

    object_prob = np.asarray(probability[bbox_y0:bbox_y1, bbox_x0:bbox_x1], dtype=np.float32)[mask_bool[bbox_y0:bbox_y1, bbox_x0:bbox_x1]]
    if object_prob.size == 0:
        return None
    return _PolygonConfidenceCandidate(
        mask=mask_bool,
        source_branches=tuple(sorted({str(branch) for branch in source_branches if branch})),
        area=area,
        bbox=bbox,
        aspect_ratio=float(aspect_ratio),
        elongation=float(elongation),
        extent=float(extent),
        peak_probability=float(np.max(object_prob)),
        mean_probability=float(np.mean(object_prob, dtype=np.float64)),
    )
