"""details_dialog: result kind hint."""
from __future__ import annotations


class ResultKindHintMixin:
    def _component_value_text(self, value: str) -> str:
        values_en = {
            'labeled frame': 'labeled frame',
            'unlabeled frame': 'unlabeled frame',
            'supervised error map': 'supervised error map',
            'variance/entropy risk map': 'variance / entropy risk map',
            'mean entropy of consensus probability': 'mean entropy of consensus probability',
            'mean variance over model probability maps': 'mean variance over model probability maps',
            'acquisition_score': 'acquisition_score',
        }
        values_ru = {
            'labeled frame': 'размеченный кадр',
            'unlabeled frame': 'неразмеченный кадр',
            'supervised error map': 'supervised-карта ошибки',
            'variance/entropy risk map': 'карта риска по variance / entropy',
            'mean entropy of consensus probability': 'средняя энтропия consensus probability',
            'mean variance over model probability maps': 'средняя дисперсия probability maps моделей',
            'acquisition_score': 'приоритет на разметку',
            'weighted': 'взвешенная',
            'core': 'ядро',
        }
        values = values_ru if getattr(self._translator, 'language', 'en') == 'ru' else values_en
        return values.get(value, value.replace(' vs ', ' против ') if getattr(self._translator, 'language', 'en') == 'ru' else value)

    def _localize_metric_lines(self, lines: list[str]) -> list[str]:
        localized: list[str] = []
        for line in lines:
            stripped = line.lstrip()
            indent = line[:len(line) - len(stripped)]
            status_prefix = ''
            for status in ("active", "auxiliary", "legacy"):
                prefix = f"{status} "
                if stripped.startswith(prefix):
                    status_prefix = f"{self._status_label(status)} "
                    stripped = stripped[len(prefix):]
                    break
            if ':' in stripped:
                name, value = stripped.split(':', 1)
                name = self._component_name_label(name.strip())
                value = self._component_value_text(value.strip())
                stripped = f"{status_prefix}{name}: {value}"
            else:
                stripped = status_prefix + self._component_value_text(stripped)
            localized.append(indent + stripped)
        return localized

    def _frame_type_label(self) -> str:
        return self._t('frame_type.point') if self._is_point_geometry() else self._t('frame_type.polygon')

    def _result_kind_hint(self, result_kind: str) -> str | None:
        if self._is_point_geometry() and result_kind == "point_matches":
            return self._t("hint.point_matches")
        if self._is_point_geometry() and result_kind == "confidence":
            return self._t("hint.point_confidence")
        if (not self._is_point_geometry()) and result_kind == "boundary":
            return self._t("hint.boundary_difference")
        if (not self._is_point_geometry()) and result_kind == "confidence":
            return self._t("hint.polygon_confidence")
        if result_kind == "confidence_bad_areas":
            return self._t("hint.confidence_bad_areas")
        if result_kind == "confidence_mix":
            return self._t("hint.confidence_mix")
        if result_kind == 'confidence_low_mask':
            return self._t('hint.confidence_low_mask')
        if result_kind == 'confidence_high_mask':
            return self._t('hint.confidence_high_mask')
        if result_kind == 'confidence_final_mask':
            return self._t('hint.confidence_final_mask')
        if result_kind == 'confidence_object_contours':
            return self._t('hint.confidence_object_contours')
        if result_kind == 'confidence_branch_debug':
            return self._t('hint.confidence_branch_debug')
        if result_kind == 'confidence_preprocessed':
            return self._t('hint.confidence_preprocessed_probability')
        if result_kind == 'confidence_boundary_cues':
            return 'Raw boundary cues before thinning'
        if result_kind == 'confidence_thin_barrier':
            return 'Thin barrier map used to stop object growth'
        if result_kind == 'confidence_core_seeds':
            return 'High-confidence object cores used as instance seeds'
        if result_kind == 'confidence_candidate_region':
            return 'Candidate region allowed for object expansion'
        if result_kind == 'confidence_bridge_cuts':
            return 'Weak bridge cuts inserted between conflicting objects'
        if result_kind == 'confidence_barrier_stops':
            return 'Pixels blocked from growth by barriers and bridge cuts'
        return None

    def _selected_score_hint(self) -> str | None:
        preferred_confidence_model = self._preferred_confidence_model_id()
        if preferred_confidence_model is not None:
            return self._t('hint.intra_model_point') if self._is_point_geometry() else self._t('hint.confidence_polygon')
        group_key = self._current_comparison_group()
        if group_key == 'model_model':
            return self._t('hint.model_model_point') if self._is_point_geometry() else self._t('hint.model_model_polygon')
        if group_key == 'model_labeled':
            return self._t('hint.model_labeled_point') if self._is_point_geometry() else self._t('hint.model_labeled_polygon')
        return None

    def _refresh_tooltips(self, score_hint: str | None, overlay_hint: str | None, result_kind: str) -> None:
        score_tooltip = score_hint or ''
        overlay_tooltip = overlay_hint or ''
        self.comparison_score_card.setToolTip(score_tooltip)
        self.comparison_score_card.title_label.setToolTip(score_tooltip)
        self.comparison_score_card.value_button.setToolTip(score_tooltip)
        self.comparison_score_card.details_label.setToolTip(score_tooltip)
        if getattr(self, "subpixel_score_card", None) is not None:
            self.subpixel_score_card.setToolTip(score_tooltip)
            self.subpixel_score_card.title_label.setToolTip(score_tooltip)
            self.subpixel_score_card.value_button.setToolTip(score_tooltip)
            self.subpixel_score_card.details_label.setToolTip(score_tooltip)
        self.result_kind_combo.setToolTip(overlay_tooltip)
        self.result_layer_title.setToolTip(overlay_tooltip)
        self.result_item.setToolTip(overlay_tooltip) if hasattr(self.result_item, 'setToolTip') else None
        if result_kind == 'diff':
            self.grayscale_diff_checkbox.setToolTip(self._t('details.grayscale_difference'))
            self.operation_label.setToolTip(self._t('details.grayscale_difference'))
        else:
            self.grayscale_diff_checkbox.setToolTip('')
            self.operation_label.setToolTip('')

    def _preferred_group_from_metric(self) -> str:
        metric_key = str(self._preferred_metric_key or "")
        if metric_key in {"model_labeled_score", "labeled_best_quality", "labeled_mean_quality"} and self._has_ground_truth():
            return "model_labeled"
        if len(self._build_result.model_specs) >= 2:
            return "model_model"
        if self._has_ground_truth():
            return "model_labeled"
        return "fallback"

    def _auto_model_model_tuple(self) -> tuple[str, str, str]:
        pairwise = self._payload.get("pairwise_model_comparisons") or ()
        if pairwise:
            top = pairwise[0]
            model_a = str(top.get("model_a", ""))
            model_b = str(top.get("model_b", ""))
            if model_a and model_b:
                return f"model_vs_model::{model_a}::{model_b}", f"model:{model_a}", f"model:{model_b}"
        model_ids = [spec.model_id for spec in self._build_result.model_specs]
        if len(model_ids) >= 2:
            return f"model_vs_model::{model_ids[0]}::{model_ids[1]}", f"model:{model_ids[0]}", f"model:{model_ids[1]}"
        if model_ids:
            return f"model_vs_model::{model_ids[0]}::{model_ids[0]}", f"model:{model_ids[0]}", f"model:{model_ids[0]}"
        return "none", "gt", "gt"

    def _auto_model_labeled_tuple(self) -> tuple[str, str, str]:
        model_metrics = self._payload.get("model_metrics") or {}
        if model_metrics:
            best_model_id = max(model_metrics.items(), key=lambda item: float(getattr(item[1], 'quality_score', 0.0)))[0]
            return f"gt_vs_model::{best_model_id}", "gt", f"model:{best_model_id}"
        model_ids = [spec.model_id for spec in self._build_result.model_specs]
        if model_ids:
            return f"gt_vs_model::{model_ids[0]}", "gt", f"model:{model_ids[0]}"
        return "none", "gt", "gt"

    def _available_comparison_groups(self) -> list[tuple[str, str]]:
        groups: list[tuple[str, str]] = []
        if len(self._build_result.model_specs) >= 2:
            groups.append((self._t("metric.group.model_model"), "model_model"))
        if self._has_ground_truth():
            groups.append((self._t("metric.group.model_labeled"), "model_labeled"))
        if not groups:
            groups.append((self._t("details.comparisons"), "fallback"))
        return groups

    def _current_comparison_group(self) -> str:
        return self._preferred_group_from_metric()

    def _refresh_comparison_groups(self, previous_group: str | None) -> None:
        groups = self._available_comparison_groups()
        self.comparison_group_combo.blockSignals(True)
        self.comparison_group_combo.clear()
        for label, key in groups:
            self.comparison_group_combo.addItem(label, key)
        target_group = previous_group
        if target_group is None and groups:
            target_group = groups[0][1]
        selected_index = 0
        for index in range(self.comparison_group_combo.count()):
            if self.comparison_group_combo.itemData(index) == target_group:
                selected_index = index
                break
        if self.comparison_group_combo.count() > 0:
            self.comparison_group_combo.setCurrentIndex(selected_index)
        self.comparison_group_combo.blockSignals(False)
