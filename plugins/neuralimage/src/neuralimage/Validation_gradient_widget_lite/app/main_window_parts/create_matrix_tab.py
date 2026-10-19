"""main_window: create matrix tab."""
from __future__ import annotations
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from ...ui.matrix_view import MatrixListWidget, MatrixMiniMapWidget
from ...ui.ui_constants import (
    DEFAULT_ANALYSIS_MODE,
    DEFAULT_MATRIX_SCORE_VIEW_MODE,
    DEFAULT_MATRIX_METRIC_KEY,
    OVERVIEW_PANEL_MAX_WIDTH,
)
from neuralimage.Validation_gradient_widget_lite.app.state import ExtendMatrixTabState


class CreateMatrixTabMixin:
    def _create_matrix_tab(self, build_result: BuildResult, snapshot: dict[str, object]) -> ExtendMatrixTabState:
        host = QWidget(self.matrix_tabs)
        matrix_layout = QHBoxLayout(host)
        matrix_layout.setContentsMargins(0, 0, 0, 0)

        content_tabs = QTabWidget(host)
        matrix_page = QWidget(content_tabs)
        matrix_page_layout = QVBoxLayout(matrix_page)
        matrix_page_layout.setContentsMargins(0, 0, 0, 0)
        matrix_view = MatrixListWidget(matrix_page)
        matrix_page_layout.addWidget(matrix_view, stretch=1)
        charts_page, histogram_cards, repeated_bad_column, repeated_good_column = self._build_histograms_panel(content_tabs, tuple(build_result.available_metric_keys), build_result)
        content_tabs.addTab(matrix_page, self._t("tab.matrix"))
        content_tabs.addTab(charts_page, self._t("tab.percentiles"))
        matrix_layout.addWidget(content_tabs, stretch=1)

        overview_host = QWidget(host)
        overview_layout = QVBoxLayout(overview_host)
        overview_layout.setContentsMargins(0, 0, 0, 0)
        overview_layout.setSpacing(8)
        mini_map = MatrixMiniMapWidget(overview_host)
        overview_layout.addWidget(mini_map)
        preview = self._build_preview_panel(overview_host, tuple(build_result.available_metric_keys), build_result)
        overview_layout.addWidget(preview.group)
        overview_layout.addStretch(1)
        overview_host.setMinimumWidth(max(300, OVERVIEW_PANEL_MAX_WIDTH - 40))
        overview_host.setMaximumWidth(OVERVIEW_PANEL_MAX_WIDTH)
        overview_host.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        matrix_layout.addWidget(overview_host)

        preview.histogram_cards = histogram_cards
        state = ExtendMatrixTabState(
            widget=host,
            matrix_view=matrix_view,
            mini_map=mini_map,
            build_result=build_result,
            content_tabs=content_tabs,
            cell_size=int(snapshot["cell_size"]),
            layout_config=snapshot["layout_config"],
            matrix_score_view_mode=str(snapshot.get("matrix_score_view_mode") or DEFAULT_MATRIX_SCORE_VIEW_MODE),
            metric_key=str(snapshot.get("metric_key") or DEFAULT_MATRIX_METRIC_KEY),
            metric_scope=str(snapshot.get("metric_scope") or ""),
            analysis_mode=str(snapshot.get("analysis_mode") or DEFAULT_ANALYSIS_MODE),
            object_type=str(snapshot.get("object_type") or "polygon"),
            confidence_model_id=str(snapshot.get("confidence_model_id") or snapshot.get("metric_scope") or "") or None,
            frame_type_filter=str(snapshot.get("frame_type_filter") or snapshot.get("object_type") or "all"),
            preview=preview,
            repeated_bad_column=repeated_bad_column,
            repeated_good_column=repeated_good_column,
        )
        matrix_view.recordSelected.connect(lambda record, s=state: self._presenter._on_record_selected(s, record))
        if hasattr(matrix_view, "tileSelected"):
            matrix_view.tileSelected.connect(lambda selection, s=state: self._presenter._on_tile_selected(s, selection))
        matrix_view.recordActivated.connect(lambda record, s=state: self._presenter._open_record_details(record, s))
        if hasattr(matrix_view, "tileActivated"):
            matrix_view.tileActivated.connect(lambda selection, s=state: self._presenter._open_record_details(selection.record, s, tile_selection=selection))
        matrix_view.overviewChanged.connect(lambda image, visible_rect, selected_position, selected_blink_on, processing_positions, reference_position, s=state: self._presenter._on_matrix_overview_changed(s, image, visible_rect, selected_position, selected_blink_on, processing_positions, reference_position))
        return state

if TYPE_CHECKING:
    from neuralimage.Validation_gradient_widget_lite.core.domain import BuildResult
