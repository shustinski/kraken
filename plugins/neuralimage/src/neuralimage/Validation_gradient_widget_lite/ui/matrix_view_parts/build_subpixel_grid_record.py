"""matrix_view: build subpixel grid record."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ...core.analysis_modes import confidence_metric_family
from ...core.domain import FrameRecord
from ...core.repository import (
    POLYGON_SUPPORT_THRESHOLD,
    _frame_uncertainty_components_from_probability,
    compute_comparison_score,
    load_frame_layers,
    load_grayscale_image,
)
from ...core.subpixel_grid import (
    SubpixelGrid,
    SubpixelGridSpec,
    build_subpixel_grid_from_array,
    build_subpixel_grid_from_pair,
)

def _build_subpixel_grid_for_record(
    record: FrameRecord,
    spec: SubpixelGridSpec,
    *,
    aggregation: str,
    metric_key: str | None,
    comparison_mode,
) -> SubpixelGrid:
    metric_family = confidence_metric_family(metric_key)
    if metric_family is not None:
        family, model_id = metric_family
        probability_path = str((getattr(record, "model_prob_paths", {}) or {}).get(model_id) or "")
        probability_array = None
        try:
            if probability_path:
                probability_array = np.asarray(load_grayscale_image(Path(probability_path)), dtype=np.float32) / 255.0
            elif model_id and model_id in (getattr(record, "model_mask_paths", {}) or {}):
                mask_path = str((getattr(record, "model_mask_paths", {}) or {}).get(model_id) or "")
                if mask_path:
                    probability_array = np.asarray(load_grayscale_image(Path(mask_path)), dtype=np.float32) / 255.0
        except Exception:
            probability_array = None
        if probability_array is not None and probability_array.ndim == 2 and probability_array.size > 0:
            try:
                if family == "model_confidence":
                    return build_subpixel_grid_from_array(
                        probability_array,
                        spec,
                        score_fn=lambda prob_tile: _frame_uncertainty_components_from_probability(
                            np.asarray(prob_tile, dtype=np.float32),
                            support_threshold=float(POLYGON_SUPPORT_THRESHOLD),
                        )[0],
                        aggregation=aggregation,
                        value_kind="risk",
                    )
                if family == "model_uncertain_fraction":
                    return build_subpixel_grid_from_array(
                        probability_array,
                        spec,
                        score_fn=lambda prob_tile: _frame_uncertainty_components_from_probability(
                            np.asarray(prob_tile, dtype=np.float32),
                            support_threshold=float(POLYGON_SUPPORT_THRESHOLD),
                        )[2],
                        aggregation=aggregation,
                        value_kind="risk",
                    )
            except Exception:
                pass
    try:
        layers = load_frame_layers(record)
        first_layer = np.asarray(layers.get("first_binary"), dtype=bool)
        second_layer = np.asarray(layers.get("second_binary"), dtype=bool)
        if first_layer.shape == second_layer.shape and first_layer.ndim == 2 and first_layer.size > 0:
            return build_subpixel_grid_from_pair(
                first_layer,
                second_layer,
                spec,
                score_fn=lambda first_tile, second_tile: compute_comparison_score(first_tile, second_tile, comparison_mode),
                aggregation=aggregation,
                value_kind="risk",
            )
    except Exception:
        pass
    parent_score = float(record.score if bool(getattr(record, "score_ready", False)) else 0.0)
    values = np.full((max(1, int(spec.rows)), max(1, int(spec.columns))), parent_score, dtype=np.float32)
    confidences = np.ones_like(values, dtype=np.float32)
    return SubpixelGrid(spec=spec.normalized(), values=values, confidences=confidences, aggregation=aggregation, value_kind="score")
