"""main_window: connect signals."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMenu,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from neuralimage.Validation_gradient_widget_lite.app.main_window_parts.correlation_column_widget import _CorrelationColumnWidget

from neuralimage.Validation_gradient_widget_lite.app.main_window_parts.percentile_histogram_widget import _PercentileHistogramCard


class ConnectSignalsMixin:
    def _build_histograms_panel(self, parent: QWidget, metric_keys: tuple[str, ...], build_result: BuildResult) -> tuple[QWidget, dict[str, QWidget], QWidget, QWidget]:
        panel = QWidget(parent)
        panel_layout = QHBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(10)

        charts_scroll = QScrollArea(panel)
        charts_scroll.setWidgetResizable(True)
        charts_host = QWidget(charts_scroll)
        charts_layout = QVBoxLayout(charts_host)
        charts_layout.setContentsMargins(8, 8, 8, 8)
        charts_layout.setSpacing(8)
        histogram_cards: dict[str, QWidget] = {}
        seen_keys: set[str] = set()
        for metric_key in metric_keys:
            if metric_key in seen_keys:
                continue
            seen_keys.add(metric_key)
            card = _PercentileHistogramCard(self._metric_text_for_key(metric_key, build_result), metric_key, charts_host)
            card.hide()
            charts_layout.addWidget(card)
            histogram_cards[str(metric_key)] = card
        charts_layout.addStretch(1)
        charts_scroll.setWidget(charts_host)
        panel_layout.addWidget(charts_scroll, stretch=3)

        columns_host = QWidget(panel)
        columns_layout = QHBoxLayout(columns_host)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.setSpacing(8)
        repeated_bad_column = _CorrelationColumnWidget(self._t("correlation.bad"), 'bad', columns_host)
        repeated_good_column = _CorrelationColumnWidget(self._t("correlation.good"), 'good', columns_host)
        columns_layout.addWidget(repeated_bad_column, 0, Qt.AlignmentFlag.AlignTop)
        columns_layout.addWidget(repeated_good_column, 0, Qt.AlignmentFlag.AlignTop)
        columns_layout.addStretch(1)
        panel_layout.addWidget(columns_host, stretch=2)

        return panel, histogram_cards, repeated_bad_column, repeated_good_column

    def _add_menu_widget(self, menu: QMenu, widget: QWidget) -> None:
        action = QWidgetAction(menu)
        action.setDefaultWidget(widget)
        menu.addAction(action)

    def _connect_signals(self) -> None:
        self.btn_add_folder.clicked.connect(self._presenter._add_folder)
        self.btn_clear_folders.clicked.connect(self._presenter._clear_folders)
        self.btn_set_original.clicked.connect(self._presenter._set_original_folder)
        self.btn_clear_original.clicked.connect(self._presenter._clear_original_folder)
        self.btn_set_gt.clicked.connect(self._presenter._set_gt_folder)
        self.btn_clear_gt.clicked.connect(self._presenter._clear_gt_folder)
        self.btn_build.clicked.connect(self._presenter._start_build)
        self.btn_compute.clicked.connect(self._presenter._on_compute_requested)
        self.btn_cancel.clicked.connect(self._presenter._request_cancel_build)

        self.layout_mode_combo.currentIndexChanged.connect(self._presenter._on_matrix_visual_parameter_changed)
        self.matrix_score_view_combo.currentIndexChanged.connect(self._presenter._on_matrix_score_view_changed)
        self.thumbnail_size_spin.valueChanged.connect(self._presenter._on_matrix_visual_parameter_changed)
        self.analysis_mode_combo.currentIndexChanged.connect(self._presenter._on_analysis_mode_changed)
        self.geometry_mode_combo.currentIndexChanged.connect(self._presenter._on_object_type_changed)
        self.polygon_compare_profile_combo.currentIndexChanged.connect(self._presenter._on_polygon_compare_profile_changed)
        self.confidence_uncertainty_profile_combo.currentIndexChanged.connect(self._presenter._sync_action_buttons)
        self.polygon_confidence_summary_combo.currentIndexChanged.connect(self._presenter._sync_action_buttons)
        self.point_match_radius_spin.valueChanged.connect(self._presenter._sync_action_buttons)
        self.point_confidence_radius_spin.valueChanged.connect(self._presenter._sync_action_buttons)
        self.point_extraction_mode_combo.currentIndexChanged.connect(self._presenter._sync_action_buttons)
        self.subpixel_view_mode_combo.currentIndexChanged.connect(self._presenter._on_subpixel_view_mode_changed)
        self.subpixel_rows_spin.valueChanged.connect(self._presenter._on_subpixel_grid_parameter_changed)
        self.subpixel_columns_spin.valueChanged.connect(self._presenter._on_subpixel_grid_parameter_changed)
        self.subpixel_aggregation_combo.currentIndexChanged.connect(self._presenter._on_subpixel_grid_parameter_changed)
        self.tile_width_spin.valueChanged.connect(self._presenter._on_tile_grid_parameter_changed)
        self.tile_height_spin.valueChanged.connect(self._presenter._on_tile_grid_parameter_changed)
        self.tile_overlap_spin.valueChanged.connect(self._presenter._on_tile_grid_parameter_changed)
        self.total_frames_spin.valueChanged.connect(self._presenter._on_matrix_visual_parameter_changed)
        self.frames_per_row_spin.valueChanged.connect(self._presenter._on_matrix_visual_parameter_changed)
        self.matrix_rows_spin.valueChanged.connect(self._presenter._on_matrix_visual_parameter_changed)
        self.matrix_columns_spin.valueChanged.connect(self._presenter._on_matrix_visual_parameter_changed)
        self.metric_scope_combo.currentIndexChanged.connect(self._presenter._on_metric_scope_changed)
        self.metric_combo.currentIndexChanged.connect(self._presenter._on_metric_changed)
        self.frame_type_filter_combo.currentIndexChanged.connect(self._presenter._on_frame_type_filter_changed)
        self.language_toggle_button.clicked.connect(self._toggle_language)
        self.matrix_tabs.currentChanged.connect(self._presenter._on_current_tab_changed)
        self.matrix_tabs.tabCloseRequested.connect(self._presenter._close_matrix_tab)
