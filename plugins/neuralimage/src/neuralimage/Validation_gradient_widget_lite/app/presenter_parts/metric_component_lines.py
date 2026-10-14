"""presenter: metric component lines."""
from __future__ import annotations
from ...core.domain import FrameRecord
from neuralimage.Validation_gradient_widget_lite.app.state import ExtendMatrixTabState


class MetricComponentLinesMixin:
    def _metric_component_lines(self, state: ExtendMatrixTabState, record: FrameRecord, metric_key: str) -> list[str]:
        summary = record.summary
        if summary is None:
            return []
        if metric_key in {"overall_frame_score", "export_priority_score"}:
            if summary.labeled_best_quality is not None:
                return [
                    "source: labeled frame",
                    "formula: 1 - labeled_best_quality",
                    f"labeled_best_quality: {summary.labeled_best_quality:.4f}",
                ]
            return [
                "source: unlabeled frame",
                "formula: disagreement_score",
                f"disagreement_score: {summary.disagreement_score:.4f}",
            ]
        if metric_key == "model_model_score":
            lines = []
            for row in summary.pairwise_metrics[:8]:
                left = self._model_display_name(state, str(row.get("model_a", "-")))
                right = self._model_display_name(state, str(row.get("model_b", "-")))
                agreement = float(row.get("agreement_score", 0.0))
                lines.append(f"{left} vs {right}: {agreement:.4f}")
                if summary.frame_type == "polygon":
                    lines.append(f"  active soft_dice: {float(row.get('soft_dice', 0.0)):.4f}")
                    lines.append(f"  active soft_iou: {float(row.get('soft_iou', 0.0)):.4f}")
                    lines.append(f"  active ssim: {float(row.get('ssim', 0.0)):.4f}")
                    lines.append(f"  active dice: {float(row.get('dice', 0.0)):.4f}")
                    lines.append(f"  active iou: {float(row.get('iou', 0.0)):.4f}")
                    lines.append(f"  active hausdorff_distance: {float(row.get('hausdorff_distance', 0.0)):.4f}")
                    lines.append(f"  active centroid_distance: {float(row.get('centroid_distance', 0.0)):.4f}")
                    lines.append(f"  auxiliary mae: {float(row.get('mae', 0.0)):.4f}")
                    lines.append(f"  auxiliary rmse: {float(row.get('rmse', 0.0)):.4f}")
                    lines.append(f"  auxiliary count_agreement: {float(row.get('count_agreement', 0.0)):.4f}")
                else:
                    lines.append(f"  active precision: {float(row.get('precision', 0.0)):.4f}")
                    lines.append(f"  active recall: {float(row.get('recall', 0.0)):.4f}")
                    lines.append(f"  active f1_at_r: {float(row.get('f1', 0.0)):.4f}")
                    lines.append(f"  active mean_localization_error: {float(row.get('mean_localization_error', 0.0)):.4f}")
                    lines.append(f"  active localization_agreement: {float(row.get('localization_agreement', 0.0)):.4f}")
                    lines.append(f"  active count_agreement: {float(row.get('count_agreement', 0.0)):.4f}")
            return lines or [f"disagreement_score: {summary.disagreement_score:.4f}"]
        if metric_key == "disagreement_score":
            return [
                "formula: 1 - model_model_score",
                f"model_model_score: {summary.metric_values.get('model_model_score', 0.0):.4f}",
            ]
        if metric_key in {"overall_polygon_score", "iou_score", "dice_score", "polygon_bce_score", "iou", "dice", "bce"}:
            return [
                f"iou: {self._format_component_value(summary.metric_values.get('iou'))}",
                f"dice: {self._format_component_value(summary.metric_values.get('dice'))}",
                f"bce: {self._format_component_value(summary.metric_values.get('bce'))}",
                f"iou_score: {self._format_component_value(summary.metric_values.get('iou_score'))}",
                f"dice_score: {self._format_component_value(summary.metric_values.get('dice_score'))}",
                f"polygon_bce_score: {self._format_component_value(summary.metric_values.get('polygon_bce_score'))}",
                f"overall_polygon_score: {self._format_component_value(summary.metric_values.get('overall_polygon_score'))}",
            ]
        if metric_key in {"overall_point_score", "precision_score", "recall_score", "f1_score", "localization_score", "precision", "recall", "f1", "mean_localization_distance"}:
            return [
                f"precision: {self._format_component_value(summary.metric_values.get('precision'))}",
                f"recall: {self._format_component_value(summary.metric_values.get('recall'))}",
                f"f1: {self._format_component_value(summary.metric_values.get('f1'))}",
                f"mean_localization_distance: {self._format_component_value(summary.metric_values.get('mean_localization_distance'))}",
                f"tp: {self._format_component_value(summary.metric_values.get('tp'))}",
                f"fp: {self._format_component_value(summary.metric_values.get('fp'))}",
                f"fn: {self._format_component_value(summary.metric_values.get('fn'))}",
                f"precision_score: {self._format_component_value(summary.metric_values.get('precision_score'))}",
                f"recall_score: {self._format_component_value(summary.metric_values.get('recall_score'))}",
                f"f1_score: {self._format_component_value(summary.metric_values.get('f1_score'))}",
                f"localization_score: {self._format_component_value(summary.metric_values.get('localization_score'))}",
                f"overall_point_score: {self._format_component_value(summary.metric_values.get('overall_point_score'))}",
            ]
        parsed_metric = metric_key.split('::', 1) if '::' in str(metric_key) else None
        if parsed_metric is not None:
            family, model_id = parsed_metric
            confidence_row = summary.model_confidence.get(model_id) if summary.model_confidence is not None else None
            model_name = self._model_display_name(state, model_id)
            if family == 'model_confidence' and confidence_row is not None:
                if hasattr(confidence_row, 'mean_object_confidence'):
                    return [
                        f"model: {model_name}",
                        f"frame_uncertainty_score: {self._format_component_value(getattr(confidence_row, 'frame_uncertainty_score', None))}",
                        f"summary_metric: {self._format_component_value(getattr(confidence_row, 'summary_metric', None))}",
                        f"mean_uncertainty: {self._format_component_value(getattr(confidence_row, 'mean_uncertainty', None))}",
                        f"low_conf_fraction: {self._format_component_value(getattr(confidence_row, 'low_conf_fraction', None))}",
                        f"worst_tail_uncertainty: {self._format_component_value(getattr(confidence_row, 'worst_tail_uncertainty', None))}",
                        f"largest_low_conf_component: {self._format_component_value(getattr(confidence_row, 'largest_low_conf_component', None))}",
                        f"mean_object_confidence: {self._format_component_value(getattr(confidence_row, 'mean_object_confidence', None))}",
                        f"uncertain_support_fraction: {self._format_component_value(getattr(confidence_row, 'uncertain_support_fraction', None))}",
                        f"top_uncertainty_mean: {self._format_component_value(getattr(confidence_row, 'top_uncertainty_mean', None))}",
                        f"largest_uncertain_region_fraction: {self._format_component_value(getattr(confidence_row, 'largest_uncertain_region_fraction', None))}",
                        f"mean_core_confidence: {self._format_component_value(getattr(confidence_row, 'mean_core_confidence', None))}",
                        f"mean_boundary_uncertainty: {self._format_component_value(getattr(confidence_row, 'mean_boundary_uncertainty', None))}",
                        f"mean_weighted_confidence: {self._format_component_value(getattr(confidence_row, 'mean_weighted_confidence', None))}",
                        f"mean_object_probability: {self._format_component_value(getattr(confidence_row, 'mean_object_probability', None))}",
                        f"uncertain_fraction: {self._format_component_value(getattr(confidence_row, 'uncertain_fraction', None))}",
                        f"mean_transition_width: {self._format_component_value(getattr(confidence_row, 'mean_transition_width', None))}",
                        f"polygon_count: {self._format_component_value(getattr(confidence_row, 'polygon_count', None))}",
                    ]
                return [
                    f"model: {model_name}",
                    f"frame_uncertainty_score: {self._format_component_value(getattr(confidence_row, 'frame_uncertainty_score', None))}",
                    f"mean_uncertainty: {self._format_component_value(getattr(confidence_row, 'mean_uncertainty', None))}",
                    f"low_conf_fraction: {self._format_component_value(getattr(confidence_row, 'low_conf_fraction', None))}",
                    f"worst_tail_uncertainty: {self._format_component_value(getattr(confidence_row, 'worst_tail_uncertainty', None))}",
                    f"largest_low_conf_component: {self._format_component_value(getattr(confidence_row, 'largest_low_conf_component', None))}",
                    f"mean_point_confidence: {self._format_component_value(getattr(confidence_row, 'mean_point_confidence', None))}",
                    f"uncertain_support_fraction: {self._format_component_value(getattr(confidence_row, 'uncertain_support_fraction', None))}",
                    f"top_uncertainty_mean: {self._format_component_value(getattr(confidence_row, 'top_uncertainty_mean', None))}",
                    f"largest_uncertain_region_fraction: {self._format_component_value(getattr(confidence_row, 'largest_uncertain_region_fraction', None))}",
                    f"mean_center_confidence: {self._format_component_value(getattr(confidence_row, 'mean_center_confidence', None))}",
                    f"mean_local_confidence: {self._format_component_value(getattr(confidence_row, 'mean_local_confidence', None))}",
                    f"mean_point_probability: {self._format_component_value(getattr(confidence_row, 'mean_point_probability', None))}",
                    f"mean_point_contrast: {self._format_component_value(getattr(confidence_row, 'mean_point_contrast', None))}",
                    f"point_count: {self._format_component_value(getattr(confidence_row, 'point_count', None))}",
                ]
            if family == 'model_uncertain_fraction' and confidence_row is not None:
                return [
                    f"model: {model_name}",
                    f"uncertain_fraction: {self._format_component_value(getattr(confidence_row, 'uncertain_fraction', None))}",
                    f"mean_boundary_uncertainty: {self._format_component_value(getattr(confidence_row, 'mean_boundary_uncertainty', None))}",
                    f"mean_transition_width: {self._format_component_value(getattr(confidence_row, 'mean_transition_width', None))}",
                    f"mean_core_confidence: {self._format_component_value(getattr(confidence_row, 'mean_core_confidence', None))}",
                ]
            if family == 'model_point_contrast' and confidence_row is not None:
                return [
                    f"model: {model_name}",
                    f"mean_point_contrast: {self._format_component_value(getattr(confidence_row, 'mean_point_contrast', None))}",
                    f"mean_local_confidence: {self._format_component_value(getattr(confidence_row, 'mean_local_confidence', None))}",
                    f"mean_center_confidence: {self._format_component_value(getattr(confidence_row, 'mean_center_confidence', None))}",
                    f"point_count: {self._format_component_value(getattr(confidence_row, 'point_count', None))}",
                ]
        if metric_key in {"model_labeled_score", "labeled_best_quality", "labeled_mean_quality"}:
            lines = []
            for spec in state.build_result.model_specs:
                metrics = summary.model_metrics.get(spec.model_id)
                if metrics is None:
                    continue
                score = getattr(metrics, "quality_score", None)
                if score is None:
                    continue
                lines.append(f"{spec.display_name}: {float(score):.4f}")
                if summary.frame_type == "polygon" and hasattr(metrics, "soft_dice"):
                    lines.append(f"  active soft_dice: {self._format_component_value(getattr(metrics, 'soft_dice', None))}")
                    lines.append(f"  active soft_iou: {self._format_component_value(getattr(metrics, 'soft_iou', None))}")
                    lines.append(f"  active ssim: {self._format_component_value(getattr(metrics, 'ssim', None))}")
                    lines.append(f"  active dice: {self._format_component_value(getattr(metrics, 'dice', None))}")
                    lines.append(f"  active iou: {self._format_component_value(getattr(metrics, 'iou', None))}")
                    lines.append(f"  active hausdorff_distance: {self._format_component_value(getattr(metrics, 'hausdorff_distance', None))}")
                    lines.append(f"  active centroid_distance: {self._format_component_value(getattr(metrics, 'centroid_distance', None))}")
                    lines.append(f"  auxiliary mae: {self._format_component_value(getattr(metrics, 'mae', None))}")
                    lines.append(f"  auxiliary rmse: {self._format_component_value(getattr(metrics, 'rmse', None))}")
                    lines.append(f"  auxiliary precision: {self._format_component_value(getattr(metrics, 'precision', None))}")
                    lines.append(f"  auxiliary recall: {self._format_component_value(getattr(metrics, 'recall', None))}")
                    lines.append(f"  auxiliary count_error: {self._format_component_value(getattr(metrics, 'count_error', None))}")
                    lines.append(f"  auxiliary cc_error: {self._format_component_value(getattr(metrics, 'connected_component_error', None))}")
                else:
                    lines.append(f"  active precision: {self._format_component_value(getattr(metrics, 'precision_at_radius', None))}")
                    lines.append(f"  active recall: {self._format_component_value(getattr(metrics, 'recall_at_radius', None))}")
                    lines.append(f"  active f1_at_r: {self._format_component_value(getattr(metrics, 'f1_at_radius', None))}")
                    lines.append(f"  active mean_localization_error: {self._format_component_value(getattr(metrics, 'mean_localization_error', None))}")
                    lines.append(f"  active localization_score: {self._format_component_value(getattr(metrics, 'localization_score', None))}")
                    lines.append(f"  active count_error: {self._format_component_value(getattr(metrics, 'count_error', None))}")
                    lines.append(f"  auxiliary chamfer_score: {self._format_component_value(getattr(metrics, 'chamfer_score', None))}")
                    lines.append(f"  auxiliary hausdorff_score: {self._format_component_value(getattr(metrics, 'hausdorff_score', None))}")
            return lines
        return []
