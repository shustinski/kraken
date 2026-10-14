"""main_window: setup ui 2."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDockWidget,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from neuralimage.UI import ClickableLabel
from neuralimage.lib.ui_texts import get_ui_section
from neuralimage.view.metrics_panel import TrainingMetricsDock
from neuralimage.view.settings_panel import create_spinbox

from neuralimage.view.main_window_parts.module_state import PREVIEW_MINIMUM_SIZE

from neuralimage.view.main_window_parts.responsive_preview_label import (
    ResponsivePreviewLabel,
    TaskQueueListWidget,
)


class SetupUi2Mixin:
    def _setup_ui(self):

        t = get_ui_section("main_window")
        self._texts = t
        self._central_content = QWidget(self)
        self.main_grid = QGridLayout(self._central_content)

        row = 0
        self.main_grid.setColumnStretch(0, 1)
        self.main_grid.setColumnStretch(1, 10)
        self.sample_count_top_label = QLabel(t.get("samples_count", "Кадров в выборке: 0"))
        self.sample_count_top_label.hide()

        self.work_mode_group = QGroupBox(t["mode"])
        sample_type_layout = QHBoxLayout(self.work_mode_group)

        self.rb_train_and_recognition = QRadioButton(t["mode_train_and_rec"])
        self.rb_further_train_model = QRadioButton(t["mode_ft_and_rec"])
        self.rb_recognition = QRadioButton(t["mode_rec"])
        self.rb_train_only = QRadioButton(t["mode_train"])
        self.rb_continue_training = QRadioButton(t.get("mode_continue_training", "Continue training"))

        sample_type_layout.addWidget(self.rb_train_and_recognition)
        sample_type_layout.addWidget(self.rb_recognition)
        sample_type_layout.addWidget(self.rb_train_only)
        sample_type_layout.addWidget(self.rb_continue_training)
        sample_type_layout.addWidget(self.rb_further_train_model)
        self.main_grid.addWidget(self.work_mode_group, row, 0, 1, 2)

        row += 1
        self.source_title_label = QLabel(t["source"])
        self.main_grid.addWidget(self.source_title_label, row, 0)
        self.source_path_row = QWidget()
        source_path_layout = QHBoxLayout(self.source_path_row)
        source_path_layout.setContentsMargins(0, 0, 0, 0)
        source_path_layout.setSpacing(10)
        self.lbl_source = ClickableLabel()
        source_path_layout.addWidget(self.lbl_source, 1)
        self.recursive_file_search_check_box = QCheckBox(str(t.get("recursive_file_search", "Search in subfolders")))
        self.recursive_file_search_check_box.setToolTip(
            str(t.get("recursive_file_search_tip", "Find source files recursively under the selected folder."))
        )
        source_path_layout.addWidget(self.recursive_file_search_check_box, 0)
        self.main_grid.addWidget(self.source_path_row, row, 1)

        row += 1
        self.result_title_label = QLabel(t["result"])
        self.main_grid.addWidget(self.result_title_label, row, 0)
        self.lbl_result = ClickableLabel()
        self.main_grid.addWidget(self.lbl_result, row, 1)

        row += 1
        self.sample_path_group = QGroupBox(t["sample"])
        self.main_grid.addWidget(self.sample_path_group, row, 0, 1, 2)
        sample_path_form = QFormLayout(self.sample_path_group)

        self.sample_path = ClickableLabel()
        self.sample_path.setToolTip(t["sample_tip"])
        sample_path_form.addRow(t["sample_src"], self.sample_path)
        self.sample_src_title_label = sample_path_form.labelForField(self.sample_path)

        self.label_path = ClickableLabel()
        self.label_path.setToolTip(t["label_tip"])
        sample_path_form.addRow(t["labels"], self.label_path)
        self.label_path_title_label = sample_path_form.labelForField(self.label_path)

        row += 1
        self.model_title_label = QLabel(t["model"])
        self.main_grid.addWidget(self.model_title_label, row, 0)
        self.model_path = ClickableLabel()
        self.main_grid.addWidget(self.model_path, row, 1)

        self.epochs_title_label = QLabel(t["epochs"])
        self.epochs_title_label.hide()
        self.le_epochs = create_spinbox((0, 1000), 1, 40)
        self.le_epochs.hide()

        row += 1
        self.buttons_row = QWidget()
        buttons_layout = QHBoxLayout(self.buttons_row)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_start = QPushButton(t["start"])
        self.btn_start.setEnabled(True)
        self.btn_start.setStyleSheet(
            """background-color: #C62828; color: white; padding: 8px 20px;
               border: none; border-radius: 5px;"""
        )
        self.btn_stop = QPushButton(t["stop"])
        self.btn_stop.setVisible(False)
        buttons_layout.addWidget(self.btn_start)
        self.main_grid.addWidget(self.buttons_row, row, 0, 1, 2)

        row += 1
        self.simple_workflows_group = QGroupBox(t.get("simple_workflows_group", "Simple workflows"))
        simple_workflows_root_layout = QVBoxLayout(self.simple_workflows_group)
        simple_workflows_layout = QHBoxLayout()
        self.btn_simple_conductors = QPushButton(
            t.get("simple_workflow_conductors", "Conductor recognition")
        )
        self.btn_simple_contacts = QPushButton(
            t.get("simple_workflow_contacts", "Contact recognition")
        )
        self.btn_simple_memory = QPushButton(
            t.get("simple_workflow_memory", "Memory recognition")
        )
        for button in (
            self.btn_simple_conductors,
            self.btn_simple_contacts,
            self.btn_simple_memory,
        ):
            button.setCheckable(True)
            button.setProperty("selectionRole", "mode")
        simple_workflows_layout.addWidget(self.btn_simple_conductors)
        simple_workflows_layout.addWidget(self.btn_simple_contacts)
        simple_workflows_layout.addWidget(self.btn_simple_memory)
        simple_workflows_root_layout.addLayout(simple_workflows_layout)
        self.simple_workflow_label = QLabel()
        simple_workflows_root_layout.addWidget(self.simple_workflow_label)
        self.main_grid.addWidget(self.simple_workflows_group, row, 0, 1, 2)
        self.simple_workflows_group.setVisible(False)
        self._update_simple_workflow_label()

        row += 1
        self.queue_group = QGroupBox(t["queue"])
        queue_layout = QVBoxLayout(self.queue_group)
        self.queue_list = TaskQueueListWidget()
        self.queue_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.queue_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.queue_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.queue_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.queue_list.setStyleSheet('QListWidget::item:selected { background: transparent; }')
        queue_layout.addWidget(self.queue_list)

        self.progress_group = QGroupBox(t["progress_group"], self._central_content)
        progress_layout = QVBoxLayout(self.progress_group)
        progress_layout.setContentsMargins(8, 8, 8, 8)
        progress_layout.setSpacing(6)
        self.training_progress_widget = QWidget(self.progress_group)
        training_progress_layout = QHBoxLayout(self.training_progress_widget)
        training_progress_layout.setContentsMargins(0, 0, 0, 0)
        training_progress_layout.setSpacing(10)
        self.epoch_progress_bar = QProgressBar()
        self.batch_progress_bar = QProgressBar()
        self.recognition_progress_bar = QProgressBar()
        for progress_bar in (self.epoch_progress_bar, self.batch_progress_bar, self.recognition_progress_bar):
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            progress_bar.setFormat("%p%")
            progress_bar.setMinimumHeight(24)
        self.progress_epochs_title_label = QLabel(t["progress_epochs"])
        self.progress_batches_title_label = QLabel(t["progress_batches"])
        for title in (self.progress_epochs_title_label, self.progress_batches_title_label):
            title.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        training_progress_layout.addWidget(self.progress_epochs_title_label)
        self.progress_epochs_title_label.hide()
        self.epoch_progress_bar.hide()
        training_progress_layout.addWidget(self.progress_batches_title_label)
        training_progress_layout.addWidget(self.batch_progress_bar, 1)
        progress_layout.addWidget(self.training_progress_widget)

        self.recognition_progress_widget = QWidget(self.progress_group)
        recognition_progress_layout = QHBoxLayout(self.recognition_progress_widget)
        recognition_progress_layout.setContentsMargins(0, 0, 0, 0)
        recognition_progress_layout.setSpacing(10)
        self.progress_recognition_title_label = QLabel(t["progress_recognition"])
        self.progress_recognition_title_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        recognition_progress_layout.addWidget(self.progress_recognition_title_label)
        recognition_progress_layout.addWidget(self.recognition_progress_bar, 1)
        progress_layout.addWidget(self.recognition_progress_widget)
        recognition_speed_default = (
            "Recognition speed: —" if self._ui_language == "en" else "Скорость распознавания: —"
        )
        self.recognition_speed_label = QLabel(t.get("recognition_speed_default", recognition_speed_default))
        self.recognition_speed_label.hide()
        self.memory_usage_label = QLabel(t["memory_label_default"])
        self.validation_quality_label = QLabel(t["validation_quality_default"])
        self.performance_label = QLabel(t["performance_label_default"])
        self._show_progress_mode("idle")
        self.progress_group.hide()

        row += 1
        self.preview_group = QGroupBox(t["preview_group"])
        preview_layout = QVBoxLayout(self.preview_group)
        self.preview_frame_name_label = QLabel(t.get("preview_current_frame_default", "Кадр: —"))
        self.preview_frame_name_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        preview_layout.addWidget(self.preview_frame_name_label)

        preview_row = QHBoxLayout()
        preview_row.setContentsMargins(0, 0, 0, 0)
        preview_row.setSpacing(8)

        self.preview_image_title_label = QLabel(t["preview_image"])
        self.preview_label_title_label = QLabel(t["preview_label"])
        self.preview_output_title_label = QLabel(t["preview_output"])
        self.preview_image_label = ResponsivePreviewLabel()
        self.preview_label_label = ResponsivePreviewLabel()
        self.preview_output_label = ResponsivePreviewLabel()
        for preview in (self.preview_image_label, self.preview_label_label, self.preview_output_label):
            preview.setStyleSheet("border: 1px solid #666; background: #111;")
        for title in (self.preview_image_title_label, self.preview_label_title_label, self.preview_output_title_label):
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        for attr_prefix, title, preview in (
            ("preview_image", self.preview_image_title_label, self.preview_image_label),
            ("preview_label", self.preview_label_title_label, self.preview_label_label),
            ("preview_output", self.preview_output_title_label, self.preview_output_label),
        ):
            column_widget = QWidget()
            column_widget.setMinimumWidth(PREVIEW_MINIMUM_SIZE)
            column_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            column_layout = QVBoxLayout(column_widget)
            column_layout.setContentsMargins(0, 0, 0, 0)
            column_layout.setSpacing(4)
            column_layout.addWidget(title, 0, Qt.AlignmentFlag.AlignHCenter)
            column_layout.addWidget(preview, 1)
            setattr(self, f"{attr_prefix}_column_widget", column_widget)
            preview_row.addWidget(column_widget, 1)
        preview_layout.addLayout(preview_row)
        self.main_grid.addWidget(self.preview_group, row, 0, 1, 2)

        row += 1
        self.log_scroll = QScrollArea()
        self.log_scroll.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.log_scroll.setWidgetResizable(True)

        self.log_container = QWidget()
        self.log_layout = QVBoxLayout(self.log_container)
        self.log_layout.setContentsMargins(5, 5, 5, 5)
        self.log_layout.setSpacing(2)
        self.log_scroll.setWidget(self.log_container)
        self.log_dock = QDockWidget(t.get("log_dock_title", "Лог"), self)
        self.log_dock.setObjectName("logDock")
        self.log_dock.setWidget(self.log_scroll)
        self.log_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        self.main_grid.setRowStretch(row, 10)

        self._central_scroll = QScrollArea(self)
        self._central_scroll.setWidgetResizable(True)
        self._central_scroll.setWidget(self._central_content)
        self.setCentralWidget(self._central_scroll)
        self.runtime_status_widget = QWidget(self)
        runtime_status_layout = QHBoxLayout(self.runtime_status_widget)
        runtime_status_layout.setContentsMargins(6, 0, 6, 0)
        runtime_status_layout.setSpacing(14)
        for runtime_label in (
            self.memory_usage_label,
            self.validation_quality_label,
            self.performance_label,
        ):
            runtime_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            runtime_status_layout.addWidget(runtime_label)
        runtime_status_layout.addStretch(1)
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().addPermanentWidget(self.runtime_status_widget, 1)

        self.queue_dock = QDockWidget(t["queue"], self)
        self.queue_dock.setObjectName('taskQueueDock')
        self.queue_dock.setWidget(self.queue_group)
        self.queue_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.queue_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.queue_dock.setMinimumHeight(110)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.queue_dock)

        self.metrics_panel = TrainingMetricsDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.metrics_panel)
        self.metrics_panel.setMinimumHeight(220)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.log_dock)
        self.tabifyDockWidget(self.metrics_panel, self.log_dock)

        if self.settings_dock is not None:
            self.settings_dock.setWindowTitle(t["settings_dock_title"])
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.settings_dock)
            self.tabifyDockWidget(self.metrics_panel, self.settings_dock)
            self.tabifyDockWidget(self.log_dock, self.settings_dock)
            self.settings_dock.show()
            self.settings_dock.raise_()

        self._create_menubar(t)
        self._apply_theme(self._theme)
        self._restore_window_layout()
        for dock in self._managed_docks():
            dock.installEventFilter(self)
            dock.dockLocationChanged.connect(self._schedule_window_layout_save)
            dock.topLevelChanged.connect(self._schedule_window_layout_save)
            dock.visibilityChanged.connect(self._schedule_window_layout_save)
