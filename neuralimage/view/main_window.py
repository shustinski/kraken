
import os
import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from UI import ClickableLabel
import numpy as np
from lib.data_interfaces import WorkMode
from lib.ui_texts import get_ui_section
from lib.version import get_app_title
from view.changelog_dialog import show_changelog_dialog
from view.help_dialog import show_help_dialog
from view.metrics_panel import TrainingMetricsDock
from view.settings_panel import create_spinbox
from view.tic_tac_toe_dialog import TicTacToeDialog
from view.window_dataclasses import MainWindowState


def load_qss_from_resource(qss_path: str):
    if not qss_path:
        return ""
    if not os.path.exists(qss_path):
        return ""
    with open(qss_path, "r", encoding="utf-8") as file:
        return file.read()


class MainView(QMainWindow):
    sample_type_changed: pyqtSignal = pyqtSignal(str)

    source_path_requested: pyqtSignal = pyqtSignal()
    result_path_requested: pyqtSignal = pyqtSignal()

    label_path_requested: pyqtSignal = pyqtSignal()
    jpg_path_requested: pyqtSignal = pyqtSignal()

    model_path_requested: pyqtSignal = pyqtSignal()

    start_requested: pyqtSignal = pyqtSignal()
    stop_requested: pyqtSignal = pyqtSignal()
    queue_remove_requested: pyqtSignal = pyqtSignal()
    queue_pause_toggle_requested: pyqtSignal = pyqtSignal()

    epochs_changed: pyqtSignal = pyqtSignal()
    request_close: pyqtSignal = pyqtSignal()

    log_message: pyqtSignal = pyqtSignal(object)
    log_message_with_delete_last: pyqtSignal = pyqtSignal(object)
    metrics_message: pyqtSignal = pyqtSignal(object)
    enable_start: pyqtSignal = pyqtSignal(bool)
    show_info: pyqtSignal = pyqtSignal(str)
    show_warning: pyqtSignal = pyqtSignal(str)
    toggle_start_stop: pyqtSignal = pyqtSignal(bool)
    batch_preview_visibility_changed: pyqtSignal = pyqtSignal(bool)
    release_memory_requested: pyqtSignal = pyqtSignal()
    open_tic_tac_toe_requested: pyqtSignal = pyqtSignal()

    def __init__(self, side_panel: QWidget | None = None):
        super().__init__()
        self.setWindowTitle(get_app_title())
        self.setWindowIcon(QIcon("_internal/icon.png"))
        self.setGeometry(200, 200, 1200, 740)
        

        self.settings_dock = side_panel if isinstance(side_panel, QDockWidget) else None
        self._close_allowed = False
        self.log_scroll: QScrollArea | None = None

        self._batch_points_by_epoch: dict[int, list[tuple[float, float]]] = {}
        self._tic_tac_toe_dialog: TicTacToeDialog | None = None
        self._ram_mb: float | None = None
        self._vram_alloc_mb: float | None = None
        self._vram_reserved_mb: float | None = None
        self._train_speed_batches_per_sec: float | None = None

        self._setup_ui()

    def _setup_ui(self):

        t = get_ui_section("main_window")
        self._texts = t
        central = QWidget(self)
        self.main_grid = QGridLayout(central)

        row = 0
        self.main_grid.setColumnStretch(0, 1)
        self.main_grid.setColumnStretch(1, 10)
        sample_type_group = QGroupBox(t["mode"])
        sample_type_layout = QHBoxLayout(sample_type_group)

        self.rb_train_and_recognition = QRadioButton(t["mode_train_and_rec"])
        self.rb_further_train_model = QRadioButton(t["mode_ft_and_rec"])
        self.rb_recognition = QRadioButton(t["mode_rec"])
        self.rb_train_only = QRadioButton(t["mode_train"])

        sample_type_layout.addWidget(self.rb_train_and_recognition)
        sample_type_layout.addWidget(self.rb_further_train_model)
        sample_type_layout.addWidget(self.rb_recognition)
        sample_type_layout.addWidget(self.rb_train_only)
        self.main_grid.addWidget(sample_type_group, row, 0, 1, 2)

        row += 1
        self.main_grid.addWidget(QLabel(t["source"]), row, 0)
        self.lbl_source = ClickableLabel()
        self.main_grid.addWidget(self.lbl_source, row, 1)

        row += 1
        self.main_grid.addWidget(QLabel(t["result"]), row, 0)
        self.lbl_result = ClickableLabel()
        self.main_grid.addWidget(self.lbl_result, row, 1)

        row += 1
        self.sample_path_group = QGroupBox(t["sample"])
        self.main_grid.addWidget(self.sample_path_group, row, 0, 1, 2)
        sample_path_form = QFormLayout(self.sample_path_group)

        self.sample_path = ClickableLabel()
        self.sample_path.setToolTip(t["sample_tip"])
        sample_path_form.addRow(t["sample_src"], self.sample_path)

        self.label_path = ClickableLabel()
        self.label_path.setToolTip(t["label_tip"])
        sample_path_form.addRow(t["labels"], self.label_path)

        row += 1
        self.main_grid.addWidget(QLabel(t["model"]), row, 0)
        self.model_path = ClickableLabel()
        self.main_grid.addWidget(self.model_path, row, 1)

        row += 1
        self.main_grid.addWidget(QLabel(t["epochs"]), row, 0)
        self.le_epochs = create_spinbox((0, 1000), 1, 40)
        self.main_grid.addWidget(self.le_epochs, row, 1)

        row += 1
        self.buttons_row = QWidget()
        buttons_layout = QHBoxLayout(self.buttons_row)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_start = QPushButton(t["start"])
        self.btn_start.setEnabled(False)
        self.btn_stop = QPushButton(t["stop"])
        self.btn_stop.setVisible(False)
        buttons_layout.addWidget(self.btn_start)
        buttons_layout.addWidget(self.btn_stop)
        self.main_grid.addWidget(self.buttons_row, row, 0, 1, 2)

        row += 1
        self.queue_group = QGroupBox(t["queue"])
        queue_layout = QVBoxLayout(self.queue_group)
        self.queue_list = QListWidget()
        queue_layout.addWidget(self.queue_list)
        queue_buttons_layout = QHBoxLayout()
        self.btn_queue_remove = QPushButton(t["queue_remove"])
        self.btn_queue_pause_toggle = QPushButton(t["queue_pause"])
        queue_buttons_layout.addWidget(self.btn_queue_remove)
        queue_buttons_layout.addWidget(self.btn_queue_pause_toggle)
        queue_layout.addLayout(queue_buttons_layout)
        self.main_grid.addWidget(self.queue_group, row, 0, 1, 2)

        row += 1
        self.progress_group = QGroupBox(t["progress_group"])
        progress_layout = QFormLayout(self.progress_group)
        self.epoch_progress_bar = QProgressBar()
        self.batch_progress_bar = QProgressBar()
        self.recognition_progress_bar = QProgressBar()
        for progress_bar in (self.epoch_progress_bar, self.batch_progress_bar, self.recognition_progress_bar):
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            progress_bar.setFormat("%p%")
        progress_layout.addRow(t["progress_epochs"], self.epoch_progress_bar)
        progress_layout.addRow(t["progress_batches"], self.batch_progress_bar)
        progress_layout.addRow(t["progress_recognition"], self.recognition_progress_bar)
        self.memory_usage_label = QLabel(t["memory_label_default"])
        progress_layout.addRow(self.memory_usage_label)
        self.validation_quality_label = QLabel(t["validation_quality_default"])
        progress_layout.addRow(self.validation_quality_label)
        self.performance_label = QLabel(t["performance_label_default"])
        progress_layout.addRow(self.performance_label)
        self.main_grid.addWidget(self.progress_group, row, 0, 1, 2)

        row += 1
        self.preview_group = QGroupBox(t["preview_group"])
        preview_layout = QHBoxLayout(self.preview_group)
        self.preview_image_label = QLabel(t["preview_image"])
        self.preview_label_label = QLabel(t["preview_label"])
        self.preview_output_label = QLabel(t["preview_output"])
        for preview in (self.preview_image_label, self.preview_label_label, self.preview_output_label):
            preview.setFixedSize(220, 220)
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview.setStyleSheet("border: 1px solid #666; background: #111;")
        preview_layout.addWidget(self.preview_image_label)
        preview_layout.addWidget(self.preview_label_label)
        preview_layout.addWidget(self.preview_output_label)
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

        self.setCentralWidget(central)

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

    def _create_menubar(self, t: dict[str, str]):
        menubar = self.menuBar()
        if menubar is None:
            return
        settings_menu = menubar.addMenu(t["menu_settings"])
        view_menu = menubar.addMenu(t["menu_view"])
        info_menu = menubar.addMenu(t["menu_help"])
        if settings_menu is None or view_menu is None or info_menu is None:
            return

        settings_menu.addAction(QAction(QIcon("./assets/new.png"), t["menu_sample"], self))
        settings_menu.addAction(QAction(QIcon("./assets/new.png"), t["menu_train"], self))
        settings_menu.addAction(QAction(QIcon("./assets/new.png"), t["menu_pred"], self))
        metrics_action = self.metrics_panel.toggleViewAction()
        if metrics_action is not None:
            metrics_action.setText(t["menu_metrics"])
            view_menu.addAction(metrics_action)
        log_action = self.log_dock.toggleViewAction()
        if log_action is not None:
            log_action.setText(t.get("menu_log_panel", "Панель лога"))
            view_menu.addAction(log_action)
        if self.settings_dock is not None:
            settings_action = self.settings_dock.toggleViewAction()
            if settings_action is not None:
                settings_action.setText(t["menu_settings_panel"])
                view_menu.addAction(settings_action)
        self.batch_preview_action = QAction(t["menu_batch_preview"], self)
        self.batch_preview_action.setCheckable(True)
        self.batch_preview_action.setChecked(True)
        view_menu.addAction(self.batch_preview_action)
        self.release_memory_action = QAction(t["menu_release_memory"], self)
        view_menu.addAction(self.release_memory_action)
        self.open_tic_tac_toe_action = QAction(
            t.get("menu_open_tic_tac_toe", "Крестики-нолики (нейросеть)"),
            self,
        )
        view_menu.addAction(self.open_tic_tac_toe_action)
        help_action = QAction(t["menu_open_help"], self)
        help_action.triggered.connect(lambda: show_help_dialog(self))
        info_menu.addAction(help_action)
        changelog_action = QAction(t.get("menu_open_changelog", "Список изменений"), self)
        changelog_action.triggered.connect(lambda: show_changelog_dialog(self))
        info_menu.addAction(changelog_action)
        menu_action = info_menu.menuAction()
        if menu_action is not None:
            menu_action.setVisible(True)

    def set_batch_preview_enabled(self, enabled: bool) -> None:
        action = getattr(self, "batch_preview_action", None)
        if action is not None:
            action.setChecked(enabled)
        preview_group = getattr(self, "preview_group", None)
        if preview_group is not None:
            preview_group.setVisible(bool(enabled))

    def is_batch_preview_enabled(self) -> bool:
        action = getattr(self, "batch_preview_action", None)
        if action is None:
            return True
        return bool(action.isChecked())

    def _current_side_panel_width(self) -> int:
        if self.settings_dock is None or not self.settings_dock.isVisible():
            return 0
        return self.settings_dock.width()

    def _compute_target_side_panel_width(self) -> int:
        if self.settings_dock is None:
            return 0
        return self.settings_dock.width()

    def _show_side_panel_fully(self):
        if self.settings_dock is None:
            return
        self.settings_dock.show()
        self.settings_dock.raise_()

    def _hide_side_panel(self):
        if self.settings_dock is None:
            return
        self.settings_dock.hide()

    def _toogle_button_clicked(self):
        if self.settings_dock is None:
            return
        panel_visible = self.settings_dock.isVisible()
        if panel_visible:
            self._hide_side_panel()
        else:
            self._show_side_panel_fully()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        return

    def connect_internal_signals(self):
        t = self._texts if hasattr(self, "_texts") else get_ui_section("main_window")
        self.rb_train_and_recognition.clicked.connect(
            lambda _: self.sample_type_changed.emit(WorkMode.train_and_recognition.value)
        )
        self.rb_recognition.clicked.connect(lambda _: self.sample_type_changed.emit(WorkMode.recognition_only.value))
        self.rb_further_train_model.clicked.connect(
            lambda _: self.sample_type_changed.emit(WorkMode.further_training.value)
        )
        self.rb_train_only.clicked.connect(lambda _: self.sample_type_changed.emit(WorkMode.train_only.value))

        self.lbl_source.clicked.connect(lambda: self.source_path_requested.emit())
        self.lbl_result.clicked.connect(lambda: self.result_path_requested.emit())

        self.label_path.clicked.connect(lambda: self.label_path_requested.emit())
        self.sample_path.clicked.connect(lambda: self.jpg_path_requested.emit())

        self.model_path.clicked.connect(lambda: self.model_path_requested.emit())

        self.le_epochs.valueChanged.connect(lambda _: self.epochs_changed.emit())

        self.btn_start.clicked.connect(lambda: self.start_requested.emit())
        self.btn_stop.clicked.connect(lambda: self.stop_requested.emit())
        self.btn_queue_remove.clicked.connect(lambda: self.queue_remove_requested.emit())
        self.btn_queue_pause_toggle.clicked.connect(lambda: self.queue_pause_toggle_requested.emit())
        if hasattr(self, "batch_preview_action"):
            self.batch_preview_action.toggled.connect(self.batch_preview_visibility_changed.emit)
            self.batch_preview_action.toggled.connect(self.set_batch_preview_enabled)
        if hasattr(self, "release_memory_action"):
            self.release_memory_action.triggered.connect(self.release_memory_requested.emit)
        if hasattr(self, "open_tic_tac_toe_action"):
            self.open_tic_tac_toe_action.triggered.connect(self.open_tic_tac_toe_requested.emit)
        self.open_tic_tac_toe_requested.connect(self._open_tic_tac_toe_dialog)

        self.log_message.connect(self._append_log)
        self.log_message_with_delete_last.connect(self.append_with_delete_previous)
        self.metrics_message.connect(self._append_metrics)
        self.enable_start.connect(self._set_start_enabled)
        self.show_info.connect(lambda txt: QMessageBox.information(self, t["info"], txt))
        self.show_warning.connect(lambda txt: QMessageBox.warning(self, t["warning"], txt))
        self.toggle_start_stop.connect(self._switch_start_stop)

    def _append_log(self, data):
        layout: QVBoxLayout = self.log_layout
        new_label = QLabel(data)
        layout.addWidget(new_label)

        if isinstance(self.log_scroll, QScrollArea):
            vbar = self.log_scroll.verticalScrollBar()
            if vbar is not None:
                vbar.setValue(vbar.maximum())

    def append_with_delete_previous(self, data: str) -> None:
        layout: QVBoxLayout = self.log_layout
        count: int = layout.count()

        if count:
            item = layout.itemAt(count - 1)
            old_widget = item.widget() if item is not None else None
            if isinstance(old_widget, QWidget):
                old_widget.deleteLater()
                layout.removeWidget(old_widget)

        self._append_log(data)

    def _append_metrics(self, data):
        if not isinstance(data, dict):
            return

        metric_type = data.get("type")
        if metric_type == "train_epoch":
            self.metrics_panel.add_train_epoch_point(int(data.get("epoch", 0)), float(data.get("loss", 0.0)))
            return

        if metric_type == "val_epoch":
            self.metrics_panel.add_val_epoch_point(int(data.get("epoch", 0)), float(data.get("loss", 0.0)))
            iou = data.get("iou")
            dice = data.get("dice")
            f1 = data.get("f1")
            if iou is not None and dice is not None and f1 is not None:
                self.validation_quality_label.setText(
                    f"IoU: {float(iou):.2%} | Dice: {float(dice):.2%} | F1: {float(f1):.2%}"
                )
            return

        if metric_type == "train_batch":
            epoch = int(data.get("epoch", 0))
            batch_index = float(data.get("batch_index", 0.0))
            loss = float(data.get("loss", 0.0))
            epoch_points = self._batch_points_by_epoch.setdefault(epoch, [])
            epoch_points.append((batch_index, loss))
            self.metrics_panel.set_batch_points(epoch, self._sparsify_batch_points(epoch_points))
            return

        if metric_type == "train_epoch_progress":
            self._set_progress_bar(self.epoch_progress_bar, int(data.get("current", 0)), int(data.get("total", 0)))
            return

        if metric_type == "train_batch_progress":
            self._set_progress_bar(self.batch_progress_bar, int(data.get("current", 0)), int(data.get("total", 0)))
            return

        if metric_type == "recognition_progress":
            self._set_progress_bar(
                self.recognition_progress_bar,
                int(data.get("current", 0)),
                int(data.get("total", 0)),
            )
            return

        if metric_type == "train_batch_preview":
            image = data.get("image")
            label = data.get("label")
            outputs = data.get("outputs", data.get("output"))
            self._set_preview_image(self.preview_image_label, image)
            self._set_preview_image(self.preview_label_label, label)
            self._set_preview_image(self.preview_output_label, outputs)
            return

        if metric_type == "system_memory":
            ram_mb = data.get("ram_mb")
            vram_alloc_mb = data.get("vram_allocated_mb")
            vram_reserved_mb = data.get("vram_reserved_mb")
            self._ram_mb = float(ram_mb) if ram_mb is not None else None
            self._vram_alloc_mb = float(vram_alloc_mb) if vram_alloc_mb is not None else None
            self._vram_reserved_mb = float(vram_reserved_mb) if vram_reserved_mb is not None else None
            self._update_memory_runtime_label()
            return

        if metric_type in ("train_perf", "train_perf_epoch"):
            data_wait_ms = float(data.get("data_wait_ms", 0.0))
            forward_ms = float(data.get("forward_ms", 0.0))
            backward_ms = float(data.get("backward_ms", 0.0))
            optimizer_ms = float(data.get("optimizer_ms", 0.0))
            total_ms = float(data.get("total_ms", 0.0))
            self.performance_label.setText(
                f"Batch ms | data: {data_wait_ms:.1f} | fwd: {forward_ms:.1f} | bwd: {backward_ms:.1f} | opt: {optimizer_ms:.1f} | total: {total_ms:.1f}"
            )
            if math.isfinite(total_ms) and total_ms > 0.0:
                self._train_speed_batches_per_sec = 1000.0 / total_ms
            else:
                self._train_speed_batches_per_sec = None
            self._update_memory_runtime_label()
            return

    def _update_memory_runtime_label(self) -> None:
        no_runtime_data = (
            self._ram_mb is None
            and self._vram_alloc_mb is None
            and self._vram_reserved_mb is None
            and self._train_speed_batches_per_sec is None
        )
        if no_runtime_data:
            self.memory_usage_label.setText(get_ui_section("main_window")["memory_label_default"])
            return

        ram_text = f"RAM: {self._ram_mb:.0f} МБ" if self._ram_mb is not None else "RAM: —"
        if self._vram_alloc_mb is None:
            vram_text = "VRAM: —"
        else:
            reserved_text = f"/{self._vram_reserved_mb:.0f}" if self._vram_reserved_mb is not None else ""
            vram_text = f"VRAM: {self._vram_alloc_mb:.0f}{reserved_text} МБ"
        speed_text = (
            f"Скорость: {self._train_speed_batches_per_sec:.2f} batch/s"
            if self._train_speed_batches_per_sec is not None
            else "Скорость: — batch/s"
        )
        self.memory_usage_label.setText(f"{ram_text} | {vram_text} | {speed_text}")

    def _reset_runtime_metrics(self) -> None:
        self._ram_mb = None
        self._vram_alloc_mb = None
        self._vram_reserved_mb = None
        self._train_speed_batches_per_sec = None

    @staticmethod
    def _sparsify_batch_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(points) > 1000:
            return points[::2]
        return points

    @staticmethod
    def _set_progress_bar(progress_bar: QProgressBar, current: int, total: int):
        if total <= 0:
            progress_bar.setValue(0)
            progress_bar.setFormat("0%")
            return
        value = max(0, min(100, int((current / total) * 100)))
        progress_bar.setValue(value)
        progress_bar.setFormat(f"{value}% ({current}/{total})")

    @staticmethod
    def _set_preview_image(widget: QLabel, image_data):
        if not isinstance(image_data, np.ndarray) or image_data.size == 0:
            return
        arr = image_data
        if arr.ndim == 2:
            qimg = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format.Format_Grayscale8).copy()
        elif arr.ndim == 3 and arr.shape[2] == 3:
            qimg = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format.Format_RGB888).copy()
        else:
            return
        pix = QPixmap.fromImage(qimg).scaled(
            widget.width(),
            widget.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        widget.setPixmap(pix)

    def _set_start_enabled(self, ok: bool):
        self.btn_start.setEnabled(ok)
        self.btn_start.setStyleSheet(
            """background-color: #4CAF50; color: white; padding: 8px 20px;
               border: none; border-radius: 5px;"""
            if ok
            else """background-color: rgb(217,217,217);"""
        )

    def _switch_start_stop(self, show_stop: bool):
        self.btn_start.setVisible(True)
        self.btn_stop.setVisible(show_stop)
        if show_stop:
            self._reset_runtime_metrics()
            self.metrics_panel.clear()
            self._batch_points_by_epoch.clear()
            self._set_progress_bar(self.epoch_progress_bar, 0, 0)
            self._set_progress_bar(self.batch_progress_bar, 0, 0)
            self._set_progress_bar(self.recognition_progress_bar, 0, 0)
            self.preview_image_label.clear()
            self.preview_label_label.clear()
            self.preview_output_label.clear()
            self.preview_image_label.setText(get_ui_section("main_window")["preview_image"])
            self.preview_label_label.setText(get_ui_section("main_window")["preview_label"])
            self.preview_output_label.setText(get_ui_section("main_window")["preview_output"])
            self._update_memory_runtime_label()
            self.validation_quality_label.setText(get_ui_section("main_window")["validation_quality_default"])
            self.performance_label.setText(get_ui_section("main_window")["performance_label_default"])

    def _open_tic_tac_toe_dialog(self):
        if self._tic_tac_toe_dialog is None:
            self._tic_tac_toe_dialog = TicTacToeDialog(self)
        self._tic_tac_toe_dialog.show()
        self._tic_tac_toe_dialog.raise_()
        self._tic_tac_toe_dialog.activateWindow()

    def get_selected_queue_row(self) -> int:
        return self.queue_list.currentRow()

    def set_task_queue_items(self, items: list[str], selected_row: int = -1):
        self.queue_list.clear()
        self.queue_list.addItems(items)
        if 0 <= selected_row < len(items):
            self.queue_list.setCurrentRow(selected_row)

    def set_stylesheet(self, style):
        self.setStyleSheet(style)

    def set_source_path(self, path: str):
        self.lbl_source.setText(path)

    def set_result_path(self, path: str):
        self.lbl_result.setText(path)

    def set_label_path(self, path: str):
        self.label_path.setText(path)

    def set_jpg_path(self, path: str):
        self.sample_path.setText(path)

    def restore_from_dataclass(self, state: MainWindowState):
        self.set_source_path(state.source_folder)
        self.set_result_path(state.result_folder)
        self.label_path.setText(state.label_folder)
        self.sample_path.setText(state.sample_folder)
        self.model_path.setText(state.model_path)
        self.le_epochs.setValue(state.epochs)

    def closeEvent(self, event):
        if self._close_allowed:
            event.accept()
            return
        self.request_close.emit()
        event.ignore()

    def allow_close(self):
        self._close_allowed = True
        self.close()

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = MainView(QWidget())
    window.show()
    sys.exit(app.exec())



