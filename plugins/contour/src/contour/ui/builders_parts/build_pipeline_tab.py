"""builders: build pipeline tab."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)
from contour.ui.pipeline_list import PipelineListWidget

def build_pipeline_tab(self) -> QWidget:
    tab = QWidget()
    layout = QVBoxLayout(tab)

    self.available_filters_group = QGroupBox("Available filters")
    available_layout = QVBoxLayout(self.available_filters_group)
    self.operation_tree = QTreeWidget()
    self.operation_tree.setHeaderHidden(True)
    self.operation_tree.setRootIsDecorated(True)
    self.operation_tree.setUniformRowHeights(True)
    self.operation_tree.currentItemChanged.connect(self._on_available_operation_selected)
    self.operation_tree.itemDoubleClicked.connect(self._on_available_operation_activated)
    self.operation_tree.setMinimumHeight(180)
    available_layout.addWidget(self.operation_tree, 1)

    self.parameters_group = QGroupBox("Step parameters")
    parameters_scroll = QScrollArea()
    parameters_scroll.setWidgetResizable(True)
    parameters_scroll.setMinimumHeight(170)
    parameters_widget = QWidget()
    self.parameters_form = QFormLayout(parameters_widget)
    self._configure_compact_form(self.parameters_form)
    parameters_scroll.setWidget(parameters_widget)
    group_layout = QVBoxLayout(self.parameters_group)
    group_layout.addWidget(parameters_scroll)

    self.pipeline_steps_group = QGroupBox("Applied filters")
    steps_layout = QVBoxLayout(self.pipeline_steps_group)

    self.pipeline_list = PipelineListWidget()
    self.pipeline_list.currentRowChanged.connect(self._on_pipeline_step_selected)
    self.pipeline_list.itemChanged.connect(self._on_pipeline_item_changed)
    self.pipeline_list.deletePressed.connect(self._remove_pipeline_step)
    self.pipeline_list.orderChanged.connect(self._sync_pipeline_order_from_list)
    self.pipeline_list.setMinimumHeight(180)
    steps_layout.addWidget(self.pipeline_list, 1)

    apply_row = QWidget()
    apply_layout = QGridLayout(apply_row)
    apply_layout.setContentsMargins(0, 0, 0, 0)
    self.auto_apply_checkbox = QCheckBox("Auto apply")
    self.auto_apply_checkbox.setChecked(True)
    self.auto_apply_checkbox.hide()
    self.save_pipeline_button = QPushButton("Save JSON")
    self.save_pipeline_button.clicked.connect(self._save_pipeline_json)
    self.load_pipeline_button = QPushButton("Load JSON")
    self.load_pipeline_button.clicked.connect(self._load_pipeline_json)
    self.show_applied_filters_checkbox = QCheckBox("Show applied filters")
    self.show_applied_filters_checkbox.setChecked(True)
    self.show_applied_filters_checkbox.toggled.connect(self._on_show_applied_filters_changed)
    apply_layout.addWidget(self.save_pipeline_button, 0, 0)
    apply_layout.addWidget(self.load_pipeline_button, 0, 1)
    apply_layout.addWidget(self.show_applied_filters_checkbox, 1, 0, 1, 2)
    apply_layout.setColumnStretch(0, 1)
    apply_layout.setColumnStretch(1, 1)
    steps_layout.addWidget(apply_row)

    self.pipeline_help_group = QGroupBox("Filter help")
    help_layout = QVBoxLayout(self.pipeline_help_group)
    self.pipeline_help_title = QLabel()
    self.pipeline_help_title.setWordWrap(True)
    self.pipeline_help_title.setFixedHeight(28)
    self.pipeline_help_summary = QLabel()
    self.pipeline_help_summary.setWordWrap(True)
    self.pipeline_help_summary.setFixedHeight(58)
    self.pipeline_help_use = QLabel()
    self.pipeline_help_use.setWordWrap(True)
    self.pipeline_help_use.setFixedHeight(74)
    preview_row = QWidget()
    preview_layout = QHBoxLayout(preview_row)
    preview_layout.setContentsMargins(0, 0, 0, 0)
    before_column = QVBoxLayout()
    self.pipeline_help_before_title = QLabel("Before")
    self.pipeline_help_before_image = QLabel()
    self.pipeline_help_before_image.setFixedSize(190, 132)
    self.pipeline_help_before_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
    before_column.addWidget(self.pipeline_help_before_title)
    before_column.addWidget(self.pipeline_help_before_image)
    after_column = QVBoxLayout()
    self.pipeline_help_after_title = QLabel("After")
    self.pipeline_help_after_image = QLabel()
    self.pipeline_help_after_image.setFixedSize(190, 132)
    self.pipeline_help_after_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
    after_column.addWidget(self.pipeline_help_after_title)
    after_column.addWidget(self.pipeline_help_after_image)
    preview_layout.addLayout(before_column)
    preview_layout.addLayout(after_column)
    help_layout.addWidget(self.pipeline_help_title)
    help_layout.addWidget(self.pipeline_help_summary)
    help_layout.addWidget(self.pipeline_help_use)
    help_layout.addWidget(preview_row)

    layout.addWidget(self.available_filters_group)
    layout.addWidget(self.parameters_group)
    layout.addWidget(self.pipeline_steps_group)
    layout.addWidget(self.pipeline_help_group)
    layout.addStretch(1)
    return tab
