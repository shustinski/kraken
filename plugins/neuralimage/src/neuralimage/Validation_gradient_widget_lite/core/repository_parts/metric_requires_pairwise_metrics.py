"""repository: metric requires pairwise metrics."""
from __future__ import annotations

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.compute_record_payloads import _parse_model_metric_key

def _metric_requires_model_confidence(metric_key: str | None) -> bool:
    parsed = _parse_model_metric_key(str(metric_key or ''))
    if parsed is None:
        return False
    family, _model_id = parsed
    return family in {'model_confidence', 'model_uncertain_fraction', 'model_point_contrast'}

def _metric_requires_pairwise_metrics(metric_key: str | None) -> bool:
    metric_key_text = str(metric_key or 'overall_frame_score')
    parsed = _parse_model_metric_key(metric_key_text)
    if parsed is not None:
        family, _model_id = parsed
        return family not in {'model_confidence', 'model_uncertain_fraction', 'model_point_contrast'}
    return metric_key_text in {
        'overall_frame_score',
        'export_priority_score',
        'model_model_score',
        'disagreement_score',
        'overall_polygon_score',
        'iou_score',
        'dice_score',
        'polygon_bce_score',
        'iou',
        'dice',
        'bce',
        'overall_point_score',
        'precision_score',
        'recall_score',
        'f1_score',
        'localization_score',
        'precision',
        'recall',
        'f1',
        'mean_localization_distance',
    }

def _metric_requires_model_metrics(metric_key: str | None) -> bool:
    metric_key_text = str(metric_key or 'overall_frame_score')
    if _parse_model_metric_key(metric_key_text) is not None:
        return False
    return metric_key_text in {
        'overall_frame_score',
        'export_priority_score',
        'model_labeled_score',
        'labeled_best_quality',
        'labeled_mean_quality',
    }
