"""presenter: update matrix preview."""
from __future__ import annotations
from ...core.domain import FrameRecord
from ...core.repository import (
    metric_value_for_record,
)
from ...ui.ui_constants import (
    DEFAULT_MATRIX_METRIC_KEY,
)
from neuralimage.Validation_gradient_widget_lite.app.state import ExtendMatrixTabState


class UpdateMatrixPreviewMixin:
    def _update_matrix_preview(self, state: ExtendMatrixTabState, record: FrameRecord | None = None) -> None:
        selected = record or state.matrix_view.current_record()
        preview = state.preview
        if preview is None:
            return
        if selected is None:
            preview.frame_value.setText("-")
            if preview.subpixel_group is not None:
                preview.subpixel_group.hide()
            if preview.subpixel_value is not None:
                preview.subpixel_value.setText("-")
            if getattr(preview, "subpixel_score_card", None) is not None:
                preview.subpixel_score_card.hide()
            for card in preview.score_cards.values():
                card.set_payload("-", self._metric_score_style(None, state.metric_key), "", visible=False, percentile_text="", percentile_style=self._percentile_style(None))
            if preview.overall_group is not None:
                preview.overall_group.hide()
            if preview.component_group is not None:
                preview.component_group.hide()
            return
        preview.frame_value.setText(selected.display_name)
        summary = selected.summary
        selected_subpixel = None
        if hasattr(state.matrix_view, "selected_tile_selection"):
            try:
                selected_subpixel = state.matrix_view.selected_tile_selection()
            except Exception:
                selected_subpixel = None
        if (
            preview.subpixel_group is not None
            and preview.subpixel_value is not None
            and selected_subpixel is not None
            and getattr(selected_subpixel, "record", None) is not None
            and getattr(selected_subpixel.record, "key", None) == selected.key
        ):
            parent_row = int(getattr(selected_subpixel, "matrix_row", getattr(selected_subpixel, "row", 0))) + 1
            parent_column = int(getattr(selected_subpixel, "matrix_column", getattr(selected_subpixel, "column", 0))) + 1
            sub_row = int(getattr(selected_subpixel, "sub_row", 0)) + 1
            sub_column = int(getattr(selected_subpixel, "sub_column", 0)) + 1
            spec = getattr(selected_subpixel, "spec", None)
            rows = int(getattr(spec, "rows", 0) or 0)
            columns = int(getattr(spec, "columns", 0) or 0)
            parent_value = float(getattr(selected_subpixel, "parent_value", selected.score if bool(getattr(selected, "score_ready", False)) else 0.0))
            subpixel_value = float(getattr(selected_subpixel, "subpixel_value", parent_value))
            summary = selected.summary
            frame_type = str(getattr(summary, "frame_type", "") or self._t("details.frame_type"))
            preview.subpixel_value.setText(
                self._t(
                    "details.subpixel_selection_value",
                    parent_row=parent_row,
                    parent_column=parent_column,
                    sub_row=sub_row,
                    sub_column=sub_column,
                    rows=rows,
                    columns=columns,
                    parent_value=parent_value,
                )
            )
            preview.subpixel_group.setTitle(self._t("details.subpixel_selection"))
            preview.subpixel_value.show()
            preview.subpixel_group.show()
            if getattr(preview, "subpixel_score_card", None) is not None:
                subpixel_metric_key = str(state.metric_key or state.build_result.selected_metric_key or DEFAULT_MATRIX_METRIC_KEY)
                subpixel_title = self._metric_text_for_key(subpixel_metric_key, state.build_result)
                subpixel_details = "\n".join(
                    [
                        f"{self._t('details.frame_type')}: {frame_type}",
                        f"{self._t('details.selected_comparison_score')}: {self._metric_score_text(subpixel_value, subpixel_metric_key)}",
                        f"{self._t('details.subpixel_selection')}: {subpixel_value:.4f}",
                        f"{self._t('details.parent_score')}: {float(getattr(selected_subpixel, 'parent_value', selected.score if bool(getattr(selected, 'score_ready', False)) else 0.0)):.4f}",
                    ]
                )
                preview.subpixel_score_card.set_payload(
                    self._metric_score_text(subpixel_value, subpixel_metric_key),
                    self._metric_score_style(subpixel_value, subpixel_metric_key),
                    subpixel_details,
                    visible=True,
                    percentile_text="",
                    percentile_style=self._percentile_style(None),
                    tooltip=self._metric_hint(subpixel_metric_key, selected.summary) if selected.summary is not None else self._metric_hint_fallback(subpixel_metric_key, state.build_result),
                )
                try:
                    preview.subpixel_score_card.title_label.setText(subpixel_title)
                except Exception:
                    pass
        elif preview.subpixel_group is not None:
            preview.subpixel_group.hide()
            if preview.subpixel_value is not None:
                preview.subpixel_value.setText("-")
            if getattr(preview, "subpixel_score_card", None) is not None:
                preview.subpixel_score_card.hide()
        percentile_cache: dict[str, dict[str, float]] = {}
        visible_metric_keys = set(self._display_metric_keys_for_state(state, state.build_result))
        overall_visible = False
        component_visible = False
        for metric_key, card in preview.score_cards.items():
            value = metric_value_for_record(selected, metric_key) if summary is not None else None
            visible = metric_key in visible_metric_keys and value is not None
            details = "\n".join(self._decorate_metric_lines(metric_key, summary, self._metric_component_lines(state, selected, metric_key))) if visible else ""
            percentile_map = percentile_cache.setdefault(metric_key, self._percentile_map_for_metric(state, metric_key)) if visible else {}
            percentile_value = percentile_map.get(selected.key) if visible else None
            tooltip = self._metric_hint(metric_key, summary) if summary is not None else self._metric_hint_fallback(metric_key, state.build_result)
            card.set_payload(
                self._metric_score_text(value, metric_key),
                self._metric_score_style(value, metric_key),
                details,
                visible=visible,
                percentile_text=self._percentile_text(percentile_value) if visible else "",
                percentile_style=self._percentile_style(percentile_value),
                tooltip=tooltip or "",
            )
            if visible:
                if str(metric_key).startswith("overall_"):
                    overall_visible = True
                else:
                    component_visible = True
        if preview.overall_group is not None:
            preview.overall_group.setVisible(overall_visible)
        if preview.component_group is not None:
            preview.component_group.setVisible(component_visible)
