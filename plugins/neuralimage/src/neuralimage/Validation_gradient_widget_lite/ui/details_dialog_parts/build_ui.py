"""details_dialog: build ui."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from ...core.domain import ComparisonMode

from neuralimage.Validation_gradient_widget_lite.ui.details_dialog_parts.expandable_detail_score_card import _ExpandableDetailScoreCard

from neuralimage.Validation_gradient_widget_lite.ui.details_dialog_parts.overlay_graphics_view import _OverlayGraphicsView


class BuildUiMixin:
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        root.addWidget(splitter)

        viewer_widget = QWidget(splitter)
        viewer_layout = QVBoxLayout(viewer_widget)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_toolbar = QHBoxLayout()
        viewer_toolbar.setContentsMargins(0, 0, 0, 0)
        self.zoom_hint_label = QLabel(self._t("details.zoom_hint"), viewer_widget)
        self.zoom_hint_label.setWordWrap(True)
        self.zoom_hint_label.setMinimumWidth(0)
        self.zoom_hint_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.zoom_hint_label.hide()
        self.confidence_loading_label = QLabel("Loading confidence...", viewer_widget)
        self.confidence_loading_label.setVisible(False)
        self.confidence_loading_bar = QProgressBar(viewer_widget)
        self.confidence_loading_bar.setRange(0, 0)
        self.confidence_loading_bar.setFixedWidth(180)
        self.confidence_loading_bar.setTextVisible(False)
        self.confidence_loading_bar.setVisible(False)
        viewer_toolbar.addWidget(self.zoom_hint_label, stretch=1)
        viewer_toolbar.addWidget(self.confidence_loading_label)
        viewer_toolbar.addWidget(self.confidence_loading_bar)
        viewer_layout.addLayout(viewer_toolbar)
        self.overlay_view = _OverlayGraphicsView(viewer_widget)
        self.overlay_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        viewer_layout.addWidget(self.overlay_view, stretch=1)

        controls_scroll = QScrollArea(splitter)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(420)
        controls_host = QWidget(controls_scroll)
        controls_layout = QVBoxLayout(controls_host)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)
        controls_scroll.setWidget(controls_host)

        frame_summary_group = QGroupBox(self._t("details.frame_info"), controls_host)
        frame_summary_form = QFormLayout(frame_summary_group)
        frame_summary_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.frame_name_value = QLabel(self._record.display_name, frame_summary_group)
        self.frame_name_value.setWordWrap(True)
        self.frame_id_value.show()
        self.subpixel_selection_value = QLabel("-", frame_summary_group)
        self.subpixel_selection_value.setWordWrap(True)
        self.subpixel_selection_value.hide()
        frame_summary_form.addRow(self._t("matrix.preview.frame"), self.frame_name_value)
        frame_summary_form.addRow(self._t("details.frame_id"), self.frame_id_value)
        frame_summary_form.addRow(self._t("details.subpixel_selection"), self.subpixel_selection_value)
        controls_layout.addWidget(frame_summary_group)

        config_group = QGroupBox(self._t("details.display"), controls_host)
        config_form = QFormLayout(config_group)
        config_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.model_a_combo = QComboBox(config_group)
        self.model_a_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.model_b_combo = QComboBox(config_group)
        self.model_b_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.primary_model_title = QLabel(self._t("details.selected_source"), config_group)
        self.secondary_model_title = QLabel(self._t("details.comparison_source"), config_group)
        self.model_a_combo.hide()
        self.model_b_combo.hide()
        self.primary_model_title.hide()
        self.secondary_model_title.hide()
        self.comparison_preset_combo = QComboBox(config_group)
        self.comparison_preset_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.comparison_preset_combo.hide()
        self.comparison_group_combo = QComboBox(config_group)
        self.comparison_group_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.comparison_group_combo.hide()
        self.result_kind_combo = QComboBox(config_group)
        self.result_kind_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.layer_view_combo = QComboBox(config_group)
        self.layer_view_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.layer_view_combo.addItem(self._t("details.layer_view.binary"), "binary")
        self.layer_view_combo.addItem(self._t("details.layer_view.source"), "source")
        self.operation_combo = QComboBox(config_group)
        self.operation_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.operation_combo.addItem(self._t("details.difference"), ComparisonMode.DISAGREEMENT)
        self.operation_combo.addItem(self._t("details.grayscale_difference"), ComparisonMode.GRAYSCALE_DIFF)
        self.operation_combo.setCurrentIndex(self.operation_combo.findData(ComparisonMode.DISAGREEMENT))
        self.operation_combo.hide()
        self.operation_label = QLabel(self._t("details.grayscale_difference"), config_group)
        self.grayscale_diff_checkbox = QCheckBox(config_group)
        config_form.addRow(self._t("details.layer_view"), self.layer_view_combo)
        config_form.addRow(self._t("details.result_overlay"), self.result_kind_combo)
        config_form.addRow(self.operation_label, self.grayscale_diff_checkbox)
        controls_layout.addWidget(config_group)

        layers_group = QGroupBox(self._t("details.layers"), controls_host)
        layers_form = QFormLayout(layers_group)
        layers_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.original_visible = QCheckBox(layers_group)
        self.original_visible.setChecked(True)
        self.original_opacity = self._opacity_slider(100)
        self.first_source_visible = QCheckBox(layers_group)
        self.first_source_visible.setChecked(True)
        self.first_source_opacity = self._opacity_slider(42)
        self.second_source_visible = QCheckBox(layers_group)
        self.second_source_visible.setChecked(True)
        self.second_source_opacity = self._opacity_slider(55)
        self.result_visible = QCheckBox(layers_group)
        self.result_visible.setChecked(True)
        self.result_opacity = self._opacity_slider(78)
        self.original_layer_title = QLabel(self._t("details.original"), layers_group)
        self.first_source_layer_title = QLabel(self._t("details.selected_source"), layers_group)
        self.second_source_layer_title = QLabel(self._t("details.comparison_source"), layers_group)
        self.result_layer_title = QLabel(self._t("details.result_overlay"), layers_group)
        self.first_mask_color_button = self._color_button()
        self.second_mask_color_button = self._color_button()
        self.result_mask_color_button = self._color_button()
        layers_form.addRow(self.original_layer_title, self._layer_row(self.original_visible, self.original_opacity))
        layers_form.addRow(self.first_source_layer_title, self._layer_row(self.first_source_visible, self.first_source_opacity, self.first_mask_color_button))
        layers_form.addRow(self.second_source_layer_title, self._layer_row(self.second_source_visible, self.second_source_opacity, self.second_mask_color_button))
        layers_form.addRow(self.result_layer_title, self._layer_row(self.result_visible, self.result_opacity, self.result_mask_color_button))
        controls_layout.addWidget(layers_group)

        comparison_score_group = QGroupBox(self._t("details.selected_comparison_score"), controls_host)
        comparison_score_layout = QVBoxLayout(comparison_score_group)
        self.comparison_score_card = _ExpandableDetailScoreCard(self._t("details.frame_score"), comparison_score_group)
        comparison_score_layout.addWidget(self.comparison_score_card)
        self.subpixel_score_card = _ExpandableDetailScoreCard(self._t("details.subpixel_score"), comparison_score_group)
        self.subpixel_score_card.hide()
        comparison_score_layout.addWidget(self.subpixel_score_card)
        controls_layout.addWidget(comparison_score_group)
        controls_layout.addStretch(1)

        splitter.addWidget(viewer_widget)
        splitter.addWidget(controls_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([980, 420])
