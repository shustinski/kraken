"""presenter: metric hint fallback."""
from __future__ import annotations
from ...core.domain import BuildResult


class MetricHintFallbackMixin:
    def _metric_hint(self, metric_key: str, summary) -> str | None:
        metric_key_text = str(metric_key)
        if '::' in metric_key_text:
            family, _model_id = metric_key_text.split('::', 1)
            if family in {'model_confidence', 'model_uncertain_fraction', 'model_point_contrast'}:
                if summary.frame_type == 'point':
                    return self._t('hint.intra_model_point')
                return self._t('hint.confidence_polygon')
        if metric_key_text == 'overall_frame_score':
            return self._t('hint.overall_labeled') if summary.is_labeled else self._t('hint.overall_unlabeled')
        if metric_key_text in {'overall_polygon_score', 'iou_score', 'dice_score', 'polygon_bce_score', 'iou', 'dice', 'bce'}:
            return self._t('hint.inter_model_polygon')
        if metric_key_text in {'overall_point_score', 'precision_score', 'recall_score', 'f1_score', 'localization_score', 'precision', 'recall', 'f1', 'mean_localization_distance'}:
            return self._t('hint.inter_model_point')
        if metric_key_text == 'model_model_score':
            return self._t('hint.model_model_point') if summary.frame_type == 'point' else self._t('hint.model_model_polygon')
        if metric_key_text in {'model_labeled_score', 'labeled_best_quality', 'labeled_mean_quality'}:
            return self._t('hint.model_labeled_point') if summary.frame_type == 'point' else self._t('hint.model_labeled_polygon')
        return None

    def _metric_hint_fallback(self, metric_key: str, build_result: BuildResult | None = None) -> str:
        metric_key_text = str(metric_key)
        sample_summary = None
        if build_result is not None:
            for record in build_result.records:
                if record.summary is not None:
                    sample_summary = record.summary
                    break
        if sample_summary is not None:
            hint = self._metric_hint(metric_key_text, sample_summary)
            if hint:
                return hint
        family = metric_key_text.split('::', 1)[0]
        defaults = {
            'overall_frame_score': self._t('hint.overall_unlabeled'),
            'model_model_score': self._t('hint.model_model_polygon'),
            'model_labeled_score': self._t('hint.model_labeled_polygon'),
            'labeled_best_quality': self._t('hint.model_labeled_polygon'),
            'labeled_mean_quality': self._t('hint.model_labeled_polygon'),
            'model_confidence': self._t('hint.confidence_polygon'),
            'model_uncertain_fraction': self._t('hint.confidence_polygon'),
            'model_point_contrast': self._t('hint.confidence_point'),
            'overall_polygon_score': self._t('hint.inter_model_polygon'),
            'iou_score': self._t('hint.inter_model_polygon'),
            'dice_score': self._t('hint.inter_model_polygon'),
            'polygon_bce_score': self._t('hint.inter_model_polygon'),
            'iou': self._t('hint.inter_model_polygon'),
            'dice': self._t('hint.inter_model_polygon'),
            'bce': self._t('hint.inter_model_polygon'),
            'overall_point_score': self._t('hint.inter_model_point'),
            'precision_score': self._t('hint.inter_model_point'),
            'recall_score': self._t('hint.inter_model_point'),
            'f1_score': self._t('hint.inter_model_point'),
            'localization_score': self._t('hint.inter_model_point'),
            'precision': self._t('hint.inter_model_point'),
            'recall': self._t('hint.inter_model_point'),
            'f1': self._t('hint.inter_model_point'),
            'mean_localization_distance': self._t('hint.inter_model_point'),
        }
        return defaults.get(family, self._metric_label(metric_key_text, build_result))
