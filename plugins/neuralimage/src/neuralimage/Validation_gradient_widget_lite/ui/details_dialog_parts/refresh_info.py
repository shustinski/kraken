"""details_dialog: refresh info."""
from __future__ import annotations


class RefreshInfoMixin:
    def _refresh_info(self) -> None:
        if self._payload_loading and not self._payload:
            self.comparison_score_card.set_payload(
                self._t("details.selected_comparison_score"),
                "Loading...",
                self._comparison_score_style(None),
                "Loading frame details...",
            )
            return
        if self._subpixel_selection is not None:
            selection = self._subpixel_selection
            self.comparison_score_card.hide()
            self.subpixel_score_card.show()
            spec = selection.spec or self._default_subpixel_spec()
            self.subpixel_selection_value.setText(
                self._t(
                    "details.subpixel_selection_value",
                    parent_row=int(selection.parent_row) + 1,
                    parent_column=int(selection.parent_column) + 1,
                    sub_row=int(selection.sub_row) + 1,
                    sub_column=int(selection.sub_column) + 1,
                    rows=int(spec.rows),
                    columns=int(spec.columns),
                    parent_value=float(selection.parent_value),
                )
            )
            self.subpixel_selection_value.show()
        else:
            self.comparison_score_card.show()
            self.subpixel_selection_value.hide()
            self.subpixel_score_card.hide()
        _preset_key, first_key, second_key = self._current_comparison_tuple()
        result_kind = self._selected_result_kind()

        score_title, score_value, score_lines = self._selected_comparison_score_info()
        score_metric_key = str(self._comparison_score_metric_key() or "overall_frame_score")

        comparison_lines: list[str] = []
        comparison_lines.append(f"{self._t('details.frame_type')}: {self._frame_type_label()}")
        if score_value is not None:
            comparison_lines.append(f"{score_title}: {float(score_value):.4f}")
        else:
            comparison_lines.append(f"{score_title}: {self._t('details.score_unavailable')}")
        comparison_lines.extend(self._localize_metric_lines(score_lines))
        if self._loading_confidence_model_id is not None and self._result_kind_requires_confidence(result_kind):
            comparison_lines.append("Loading confidence visualization...")
        self.comparison_score_card.set_payload(
            score_title,
            self._comparison_score_text(score_value, score_metric_key),
            self._comparison_score_style(score_value, score_metric_key),
            "\n".join(comparison_lines),
            self._selected_score_hint() or "",
        )
        if self._subpixel_selection is not None:
            selection = self._subpixel_selection
            subpixel_metric_key = str(selection.metric_key or score_metric_key)
            subpixel_title, subpixel_value, subpixel_lines = self._selected_subpixel_score_info()
            self.subpixel_score_card.set_payload(
                subpixel_title,
                self._comparison_score_text(subpixel_value, subpixel_metric_key),
                self._comparison_score_style(subpixel_value, subpixel_metric_key),
                "\n".join(self._localize_metric_lines(subpixel_lines)),
                self._selected_score_hint() or "",
            )
        else:
            self.subpixel_score_card.hide()
