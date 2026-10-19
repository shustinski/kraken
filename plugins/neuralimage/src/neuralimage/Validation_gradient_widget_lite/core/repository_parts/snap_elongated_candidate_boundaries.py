"""repository: snap elongated candidate boundaries."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidencePipelineConfig,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import _label_components

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.make_polygon_candidate import _make_polygon_candidate

def _snap_elongated_candidate_boundaries(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    low_threshold: float,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
    source_branches: tuple[str, ...],
) -> np.ndarray:
    """Refine large elongated candidates by snapping the minor-axis boundaries to local profiles."""

    def _smooth_trace(values: np.ndarray, valid_mask: np.ndarray, window: int) -> np.ndarray:
        from neuralimage.Validation_gradient_widget_lite.core.repository import ndi

        result = np.asarray(values, dtype=np.float32).copy()
        valid = np.asarray(valid_mask, dtype=bool)
        if not np.any(valid):
            return result
        coords = np.flatnonzero(valid)
        samples = np.asarray(result[coords], dtype=np.float32)
        if samples.size >= 3:
            kernel = max(3, int(window) | 1)
            if ndi is not None:
                samples = np.asarray(ndi.median_filter(samples, size=kernel, mode='nearest'), dtype=np.float32)
            else:
                radius = kernel // 2
                padded = np.pad(samples, (radius, radius), mode='edge')
                smoothed = np.empty_like(samples)
                for index in range(samples.size):
                    smoothed[index] = float(np.median(padded[index:index + kernel]))
                samples = smoothed
        result[coords] = samples
        return result

    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return mask_bool
    candidate = _make_polygon_candidate(probability, mask_bool, source_branches)
    if candidate is None:
        return mask_bool
    if float(max(candidate.aspect_ratio, candidate.elongation)) < float(config.boundary_snap_min_aspect):
        return mask_bool
    x0, y0, width, height = candidate.bbox
    if width < 6 or height < 6:
        return mask_bool

    pad_y = max(2, min(6, height))
    pad_x = max(2, min(6, width))
    roi_x0 = max(0, x0 - pad_x)
    roi_y0 = max(0, y0 - pad_y)
    roi_x1 = min(mask_bool.shape[1], x0 + width + pad_x)
    roi_y1 = min(mask_bool.shape[0], y0 + height + pad_y)

    roi_mask = np.asarray(mask_bool[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    roi_prob = np.asarray(probability[roi_y0:roi_y1, roi_x0:roi_x1], dtype=np.float32)
    inner_x0 = x0 - roi_x0
    inner_y0 = y0 - roi_y0
    inner_x1 = inner_x0 + width
    inner_y1 = inner_y0 + height
    inner_mask = np.asarray(roi_mask[inner_y0:inner_y1, inner_x0:inner_x1], dtype=bool)
    object_prob = np.asarray(roi_prob[roi_mask], dtype=np.float32)
    if object_prob.size == 0:
        return mask_bool

    refined_roi = np.zeros_like(roi_mask, dtype=bool)
    horizontal = width >= height
    if horizontal:
        active_cols = np.any(inner_mask, axis=0)
        if not np.any(active_cols):
            return mask_bool
        center_profile = np.mean(roi_prob[:, inner_x0:inner_x1][:, active_cols], axis=1, dtype=np.float32)
        active_rows = np.any(inner_mask, axis=1)
        if np.count_nonzero(active_rows) <= 2:
            return mask_bool
        active_row_indices = inner_y0 + np.flatnonzero(active_rows)
        edge_drop = float(np.max(center_profile[active_row_indices])) - float(max(center_profile[active_row_indices[0]], center_profile[active_row_indices[-1]]))
        if edge_drop < float(config.boundary_snap_min_drop):
            return mask_bool
        center_row = int(np.argmax(center_profile))
        top_trace = np.full(width, np.nan, dtype=np.float32)
        bottom_trace = np.full(width, np.nan, dtype=np.float32)
        valid_trace = np.zeros(width, dtype=bool)
        for local_col in np.flatnonzero(active_cols):
            col = inner_x0 + int(local_col)
            col_profile = np.asarray(roi_prob[:, col], dtype=np.float32)
            baseline = float(np.quantile(col_profile, 0.2))
            peak_row = int(np.argmax(col_profile))
            peak_value = float(col_profile[peak_row])
            profile_threshold = max(
                float(low_threshold),
                min(
                    float(strong_threshold),
                    baseline + 0.45 * max(0.0, peak_value - baseline),
                ),
            )
            keep_rows = col_profile >= profile_threshold
            if not np.any(keep_rows):
                continue
            row_labels, row_count = _label_components(keep_rows[:, None])
            selected_rows = None
            preferred_row = peak_row if keep_rows[peak_row] else center_row
            for label_id in range(1, int(row_count) + 1):
                component = (row_labels[:, 0] == label_id)
                if component[preferred_row]:
                    selected_rows = component
                    break
            if selected_rows is None:
                continue
            selected_indices = np.flatnonzero(selected_rows & roi_mask[:, col])
            if selected_indices.size == 0:
                continue
            top_trace[int(local_col)] = float(selected_indices[0])
            bottom_trace[int(local_col)] = float(selected_indices[-1])
            valid_trace[int(local_col)] = True
        if not np.any(valid_trace):
            return mask_bool
        active_col_indices = np.flatnonzero(active_cols)
        valid_col_indices = np.flatnonzero(valid_trace)
        top_trace = np.interp(active_col_indices, valid_col_indices, top_trace[valid_col_indices]).astype(np.float32)
        bottom_trace = np.interp(active_col_indices, valid_col_indices, bottom_trace[valid_col_indices]).astype(np.float32)
        smooth_window = max(3, min(11, width // 6 if width >= 6 else 3))
        top_smoothed = _smooth_trace(top_trace, np.ones_like(top_trace, dtype=bool), smooth_window)
        bottom_smoothed = _smooth_trace(bottom_trace, np.ones_like(bottom_trace, dtype=bool), smooth_window)
        for position, local_col in enumerate(active_col_indices.tolist()):
            col = inner_x0 + int(local_col)
            top = int(np.clip(round(float(top_smoothed[position])), 0, roi_mask.shape[0] - 1))
            bottom = int(np.clip(round(float(bottom_smoothed[position])), top, roi_mask.shape[0] - 1))
            refined_roi[top:bottom + 1, col] = roi_mask[top:bottom + 1, col]
    else:
        active_rows = np.any(inner_mask, axis=1)
        if not np.any(active_rows):
            return mask_bool
        center_profile = np.mean(roi_prob[inner_y0:inner_y1, :][active_rows, :], axis=0, dtype=np.float32)
        active_cols = np.any(inner_mask, axis=0)
        if np.count_nonzero(active_cols) <= 2:
            return mask_bool
        active_col_indices = inner_x0 + np.flatnonzero(active_cols)
        edge_drop = float(np.max(center_profile[active_col_indices])) - float(max(center_profile[active_col_indices[0]], center_profile[active_col_indices[-1]]))
        if edge_drop < float(config.boundary_snap_min_drop):
            return mask_bool
        center_col = int(np.argmax(center_profile))
        left_trace = np.full(height, np.nan, dtype=np.float32)
        right_trace = np.full(height, np.nan, dtype=np.float32)
        valid_trace = np.zeros(height, dtype=bool)
        for local_row in np.flatnonzero(active_rows):
            row = inner_y0 + int(local_row)
            row_profile = np.asarray(roi_prob[row, :], dtype=np.float32)
            baseline = float(np.quantile(row_profile, 0.2))
            peak_col = int(np.argmax(row_profile))
            peak_value = float(row_profile[peak_col])
            profile_threshold = max(
                float(low_threshold),
                min(
                    float(strong_threshold),
                    baseline + 0.45 * max(0.0, peak_value - baseline),
                ),
            )
            keep_cols = row_profile >= profile_threshold
            if not np.any(keep_cols):
                continue
            col_labels, col_count = _label_components(keep_cols[None, :])
            selected_cols = None
            preferred_col = peak_col if keep_cols[peak_col] else center_col
            for label_id in range(1, int(col_count) + 1):
                component = (col_labels[0, :] == label_id)
                if component[preferred_col]:
                    selected_cols = component
                    break
            if selected_cols is None:
                continue
            selected_indices = np.flatnonzero(selected_cols & roi_mask[row, :])
            if selected_indices.size == 0:
                continue
            left_trace[int(local_row)] = float(selected_indices[0])
            right_trace[int(local_row)] = float(selected_indices[-1])
            valid_trace[int(local_row)] = True
        if not np.any(valid_trace):
            return mask_bool
        active_row_indices = np.flatnonzero(active_rows)
        valid_row_indices = np.flatnonzero(valid_trace)
        left_trace = np.interp(active_row_indices, valid_row_indices, left_trace[valid_row_indices]).astype(np.float32)
        right_trace = np.interp(active_row_indices, valid_row_indices, right_trace[valid_row_indices]).astype(np.float32)
        smooth_window = max(3, min(11, height // 6 if height >= 6 else 3))
        left_smoothed = _smooth_trace(left_trace, np.ones_like(left_trace, dtype=bool), smooth_window)
        right_smoothed = _smooth_trace(right_trace, np.ones_like(right_trace, dtype=bool), smooth_window)
        for position, local_row in enumerate(active_row_indices.tolist()):
            row = inner_y0 + int(local_row)
            left = int(np.clip(round(float(left_smoothed[position])), 0, roi_mask.shape[1] - 1))
            right = int(np.clip(round(float(right_smoothed[position])), left, roi_mask.shape[1] - 1))
            refined_roi[row, left:right + 1] = roi_mask[row, left:right + 1]

    if not np.any(refined_roi):
        return mask_bool
    original_area = int(np.count_nonzero(roi_mask))
    refined_area = int(np.count_nonzero(refined_roi))
    if refined_area <= 0:
        return mask_bool
    retained_fraction = float(refined_area / max(1, original_area))
    if retained_fraction < float(config.boundary_snap_min_retained_fraction):
        return mask_bool
    refined_mask = mask_bool.copy()
    refined_mask[roi_y0:roi_y1, roi_x0:roi_x1] = refined_roi
    return refined_mask
