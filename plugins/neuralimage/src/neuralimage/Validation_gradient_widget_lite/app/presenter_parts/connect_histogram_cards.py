"""presenter: connect histogram cards."""
from __future__ import annotations
from neuralimage.Validation_gradient_widget_lite.app.state import ExtendMatrixTabState


class ConnectHistogramCardsMixin:
    def _connect_histogram_cards(self, state: ExtendMatrixTabState) -> None:
        preview = state.preview
        if preview is None:
            return
        for metric_key, card in preview.histogram_cards.items():
            if hasattr(card, 'binClicked'):
                card.binClicked.connect(lambda clicked_metric_key, bin_index, s=state: self._on_histogram_bin_clicked(s, str(clicked_metric_key), int(bin_index)))
        if state.repeated_bad_column is not None:
            state.repeated_bad_column.columnClicked.connect(lambda band, s=state: self._on_correlation_column_clicked(s, str(band)))
        if state.repeated_good_column is not None:
            state.repeated_good_column.columnClicked.connect(lambda band, s=state: self._on_correlation_column_clicked(s, str(band)))

    def _on_correlation_column_clicked(self, state: ExtendMatrixTabState, band: str) -> None:
        state.percentile_filter_metric_key = None
        state.percentile_filter_bin_index = None
        state.correlation_filter_band = None if state.correlation_filter_band == band else str(band)
        self._apply_tab_visual_settings(state, reset_view=False)
        if state.content_tabs is not None:
            state.content_tabs.setCurrentIndex(0)

    def _on_histogram_bin_clicked(self, state: ExtendMatrixTabState, metric_key: str, bin_index: int) -> None:
        same_filter = state.percentile_filter_metric_key == metric_key and state.percentile_filter_bin_index == int(bin_index)
        state.correlation_filter_band = None
        if same_filter:
            state.percentile_filter_metric_key = None
            state.percentile_filter_bin_index = None
        else:
            state.percentile_filter_metric_key = str(metric_key)
            state.percentile_filter_bin_index = int(bin_index)
        self._apply_tab_visual_settings(state, reset_view=False)

    def _model_display_name(self, state: ExtendMatrixTabState, model_id: str) -> str:
        for spec in state.build_result.model_specs:
            if spec.model_id == model_id:
                return spec.display_name
        return model_id

    def _format_component_value(self, value) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
