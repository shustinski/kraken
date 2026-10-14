"""details_dialog: selected comparison score info."""
from __future__ import annotations


class SelectedComparisonScoreInfoMixin:
    def _selected_comparison_score_info(self) -> tuple[str, float | None, list[str]]:
        model_ids = self._comparison_model_ids()
        group_key = self._current_comparison_group()
        preferred_confidence_model = self._preferred_confidence_model_id()
        if preferred_confidence_model is None and self._selected_result_kind() == "confidence":
            preferred_confidence_model = self._confidence_model_id()
        if preferred_confidence_model is not None:
            confidence_row = self._confidence_metrics_for_model(preferred_confidence_model)
            if confidence_row is not None:
                if hasattr(confidence_row, 'mean_object_confidence'):
                    lines = [
                        f"frame_uncertainty_score: {float(getattr(confidence_row, 'frame_uncertainty_score', 0.0)):.4f}",
                        f"mean_uncertainty: {float(getattr(confidence_row, 'mean_uncertainty', 0.0)):.4f}",
                        f"low_conf_fraction: {float(getattr(confidence_row, 'low_conf_fraction', 0.0)):.4f}",
                        f"worst_tail_uncertainty: {float(getattr(confidence_row, 'worst_tail_uncertainty', 0.0)):.4f}",
                        f"largest_low_conf_component: {float(getattr(confidence_row, 'largest_low_conf_component', 0.0)):.4f}",
                        f"mean_frame_confidence: {float(getattr(confidence_row, 'mean_object_confidence', 0.0)):.4f}",
                        f"mean_frame_probability: {float(getattr(confidence_row, 'mean_object_probability', 0.0)):.4f}",
                        f"uncertain_fraction: {float(getattr(confidence_row, 'uncertain_fraction', 0.0)):.4f}",
                        f"uncertain_support_fraction: {float(getattr(confidence_row, 'uncertain_support_fraction', 0.0)):.4f}",
                        f"top_uncertainty_mean: {float(getattr(confidence_row, 'top_uncertainty_mean', 0.0)):.4f}",
                        f"largest_uncertain_region_fraction: {float(getattr(confidence_row, 'largest_uncertain_region_fraction', 0.0)):.4f}",
                        f"focus_area_fraction: {float(getattr(confidence_row, 'object_area_fraction', 0.0)):.4f}",
                    ]
                    return (
                        self._t("details.frame_score"),
                        float(getattr(confidence_row, 'frame_uncertainty_score', 0.0)),
                        lines,
                    )
                lines = [
                    f"frame_uncertainty_score: {float(getattr(confidence_row, 'frame_uncertainty_score', 0.0)):.4f}",
                    f"mean_uncertainty: {float(getattr(confidence_row, 'mean_uncertainty', 0.0)):.4f}",
                    f"low_conf_fraction: {float(getattr(confidence_row, 'low_conf_fraction', 0.0)):.4f}",
                    f"worst_tail_uncertainty: {float(getattr(confidence_row, 'worst_tail_uncertainty', 0.0)):.4f}",
                    f"largest_low_conf_component: {float(getattr(confidence_row, 'largest_low_conf_component', 0.0)):.4f}",
                    f"mean_point_confidence: {float(getattr(confidence_row, 'mean_point_confidence', 0.0)):.4f}",
                    f"mean_center_confidence: {float(getattr(confidence_row, 'mean_center_confidence', 0.0)):.4f}",
                    f"mean_local_confidence: {float(getattr(confidence_row, 'mean_local_confidence', 0.0)):.4f}",
                    f"mean_point_probability: {float(getattr(confidence_row, 'mean_point_probability', 0.0)):.4f}",
                    f"uncertain_support_fraction: {float(getattr(confidence_row, 'uncertain_support_fraction', 0.0)):.4f}",
                    f"top_uncertainty_mean: {float(getattr(confidence_row, 'top_uncertainty_mean', 0.0)):.4f}",
                    f"largest_uncertain_region_fraction: {float(getattr(confidence_row, 'largest_uncertain_region_fraction', 0.0)):.4f}",
                    f"mean_point_contrast: {float(getattr(confidence_row, 'mean_point_contrast', 0.0)):.4f}",
                    f"point_count: {int(getattr(confidence_row, 'point_count', 0))}",
                ]
                return (
                    self._t("details.frame_score"),
                    float(getattr(confidence_row, 'frame_uncertainty_score', 0.0)),
                    lines,
                )
            quick = self._quick_confidence_score(preferred_confidence_model)
            if quick is not None:
                if self._is_point_geometry():
                    lines = [
                        f"frame_uncertainty_score: {quick['frame_uncertainty_score']:.4f}",
                        f"mean_uncertainty: {quick['mean_uncertainty']:.4f}",
                        f"low_conf_fraction: {quick['low_conf_fraction']:.4f}",
                        f"worst_tail_uncertainty: {quick['worst_tail_uncertainty']:.4f}",
                        f"largest_low_conf_component: {quick['largest_low_conf_component']:.4f}",
                        f"mean_point_confidence: {quick['mean_confidence']:.4f}",
                        f"mean_point_probability: {quick['mean_probability']:.4f}",
                        f"uncertain_fraction: {quick['uncertain_fraction']:.4f}",
                        f"uncertain_support_fraction: {quick['uncertain_support_fraction']:.4f}",
                        f"top_uncertainty_mean: {quick['top_uncertainty_mean']:.4f}",
                        f"largest_uncertain_region_fraction: {quick['largest_uncertain_region_fraction']:.4f}",
                        f"object_area_fraction: {quick['focus_fraction']:.4f}",
                    ]
                    return self._t("details.frame_score"), float(quick['frame_uncertainty_score']), lines
                lines = [
                    f"frame_uncertainty_score: {quick['frame_uncertainty_score']:.4f}",
                    f"mean_uncertainty: {quick['mean_uncertainty']:.4f}",
                    f"low_conf_fraction: {quick['low_conf_fraction']:.4f}",
                    f"worst_tail_uncertainty: {quick['worst_tail_uncertainty']:.4f}",
                    f"largest_low_conf_component: {quick['largest_low_conf_component']:.4f}",
                    f"mean_frame_confidence: {quick['mean_confidence']:.4f}",
                    f"mean_frame_probability: {quick['mean_probability']:.4f}",
                    f"uncertain_fraction: {quick['uncertain_fraction']:.4f}",
                    f"uncertain_support_fraction: {quick['uncertain_support_fraction']:.4f}",
                    f"top_uncertainty_mean: {quick['top_uncertainty_mean']:.4f}",
                    f"largest_uncertain_region_fraction: {quick['largest_uncertain_region_fraction']:.4f}",
                    f"focus_area_fraction: {quick['focus_fraction']:.4f}",
                ]
                return self._t("details.frame_score"), float(quick['frame_uncertainty_score']), lines
        if group_key == "model_model":
            if len(model_ids) >= 2:
                pairwise = self._pairwise_comparison_entry(model_ids[0], model_ids[1])
                if pairwise is not None:
                    score = float(pairwise.get("agreement_score", 0.0))
                    if self._is_point_geometry():
                        return (
                            self._t("details.frame_score"),
                            score,
                            [
                                f"active precision: {float(pairwise.get('precision', 0.0)):.4f}",
                                f"active recall: {float(pairwise.get('recall', 0.0)):.4f}",
                                f"active f1_at_r: {float(pairwise.get('f1', 0.0)):.4f}",
                                f"active mean_localization_error: {float(pairwise.get('mean_localization_error', 0.0)):.4f}",
                                f"active localization_agreement: {float(pairwise.get('localization_agreement', 0.0)):.4f}",
                                f"active count_agreement: {float(pairwise.get('count_agreement', 0.0)):.4f}",
                            ],
                        )
                    return (
                        self._t("details.frame_score"),
                        score,
                        [
                            f"active soft_dice: {float(pairwise.get('soft_dice', 0.0)):.4f}",
                            f"active soft_iou: {float(pairwise.get('soft_iou', 0.0)):.4f}",
                            f"active ssim: {float(pairwise.get('ssim', 0.0)):.4f}",
                            f"active dice: {float(pairwise.get('dice', 0.0)):.4f}",
                            f"active iou: {float(pairwise.get('iou', 0.0)):.4f}",
                            f"active hausdorff_distance: {float(pairwise.get('hausdorff_distance', 0.0)):.4f}",
                            f"active centroid_distance: {float(pairwise.get('centroid_distance', 0.0)):.4f}",
                            f"auxiliary mae: {float(pairwise.get('mae', 0.0)):.4f}",
                            f"auxiliary rmse: {float(pairwise.get('rmse', 0.0)):.4f}",
                            f"auxiliary count_agreement: {float(pairwise.get('count_agreement', 0.0)):.4f}",
                        ],
                    )
            return self._t("details.frame_score"), None, [self._t("details.score_unavailable")]
        if group_key == "model_labeled":
            model_id = model_ids[0] if model_ids else None
            model_metrics = self._payload.get("model_metrics") or {}
            selected = model_metrics.get(model_id) if model_id is not None else None
            if selected is not None:
                score = float(selected.quality_score)
                if hasattr(selected, "f1_at_radius"):
                    return (
                        self._t("details.frame_score"),
                        score,
                        [
                            f"active precision: {float(selected.precision_at_radius):.4f}",
                            f"active recall: {float(selected.recall_at_radius):.4f}",
                            f"active f1_at_r: {float(selected.f1_at_radius):.4f}",
                            f"active mean_localization_error: {float(selected.mean_localization_error):.4f}",
                            f"active localization_score: {float(selected.localization_score):.4f}",
                            f"active count_error: {float(selected.count_error):.4f}",
                            f"auxiliary chamfer_score: {float(selected.chamfer_score):.4f}",
                            f"auxiliary hausdorff_score: {float(selected.hausdorff_score):.4f}",
                        ],
                    )
                return (
                    self._t("details.frame_score"),
                    score,
                    [
                        f"active soft_dice: {float(selected.soft_dice):.4f}",
                        f"active soft_iou: {float(selected.soft_iou):.4f}",
                        f"active ssim: {float(selected.ssim):.4f}",
                        f"active dice: {float(selected.dice):.4f}",
                        f"active iou: {float(selected.iou):.4f}",
                        f"active hausdorff_distance: {float(selected.hausdorff_distance):.4f}",
                        f"active centroid_distance: {float(selected.centroid_distance):.4f}",
                        f"auxiliary mae: {float(selected.mae):.4f}",
                        f"auxiliary rmse: {float(selected.rmse):.4f}",
                        f"auxiliary precision: {float(selected.precision):.4f}",
                        f"auxiliary recall: {float(selected.recall):.4f}",
                        f"auxiliary count_error: {float(selected.count_error):.4f}",
                        f"auxiliary connected_component_error: {float(selected.connected_component_error):.4f}",
                    ],
                )
            return self._t("details.frame_score"), None, [self._t("details.score_unavailable")]
        return self._t("details.frame_score"), None, []
