"""presenter: metric component summary."""
from __future__ import annotations


class MetricComponentSummaryMixin:
    def _metric_component_summary(self, metric_key: str, summary) -> str:
        metric_key_text = str(metric_key)
        family = metric_key_text.split('::', 1)[0]
        is_ru = getattr(self._i18n, 'language', 'en') == 'ru'
        if family == 'overall_frame_score':
            return self._t('metric.model_labeled_score') if summary.is_labeled else self._t('metric.disagreement_score')
        if family in {'overall_polygon_score', 'iou_score', 'dice_score', 'polygon_bce_score', 'iou', 'dice', 'bce'}:
            return 'iou + dice + bce score'
        if family in {'overall_point_score', 'precision_score', 'recall_score', 'f1_score', 'localization_score', 'precision', 'recall', 'f1', 'mean_localization_distance'}:
            return 'precision + recall + f1 + localization + tp/fp/fn'
        if family == 'model_model_score':
            return 'soft_dice + soft_iou + ssim + dice + iou + hausdorff + centroid' if summary.frame_type != 'point' else ('precision + recall + f1_at_r + локализация + количество' if is_ru else 'precision + recall + f1_at_r + localization + count')
        if family in {'model_labeled_score', 'labeled_best_quality', 'labeled_mean_quality'}:
            return 'soft_dice + soft_iou + ssim + dice + iou + hausdorff + centroid' if summary.frame_type != 'point' else ('precision + recall + f1_at_r + локализация + количество' if is_ru else 'precision + recall + f1_at_r + localization + count')
        if family == 'model_confidence':
            return 'средняя уверенность внутри объекта' if (is_ru and summary.frame_type != 'point') else ('средняя уверенность по точкам' if is_ru else ('mean object confidence' if summary.frame_type != 'point' else 'mean point confidence'))
        if family == 'model_uncertain_fraction':
            return 'доля сомнительных пикселей объекта' if is_ru else 'uncertain object fraction'
        if family == 'model_point_contrast':
            return 'средний контраст точек' if is_ru else 'mean point contrast'
        if family == 'disagreement_score':
            return '1 - согласие моделей' if is_ru else '1 - model-to-model agreement'
        return '-'
