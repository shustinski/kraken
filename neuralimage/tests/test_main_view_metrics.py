import pytest
from pathlib import Path
import numpy as np

pytest.importorskip('PyQt6')

from PyQt6.QtWidgets import QApplication, QSizePolicy, QScrollArea, QWidget

from UI.clickable_label import ClickableLabel
from lib.logging_policy import MAX_LOG_MESSAGES
from view.main_window import MainView


@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_view_metrics_are_collected_without_capping(qapp):
    view = MainView(QWidget())
    view.connect_internal_signals()

    view.metrics_message.emit({'type': 'train_epoch', 'epoch': 1, 'loss': 0.6})
    view.metrics_message.emit({'type': 'val_epoch', 'epoch': 1, 'loss': 0.7, 'iou': 0.8, 'dice': 0.85, 'f1': 0.85})
    view.metrics_message.emit({'type': 'train_epoch_progress', 'current': 2, 'total': 10})
    view.metrics_message.emit({'type': 'train_batch_progress', 'current': 50, 'total': 200})
    view.metrics_message.emit({'type': 'recognition_progress', 'current': 3, 'total': 12})
    view.metrics_message.emit(
        {
            'type': 'train_perf',
            'epoch': 1,
            'batch_index': 10,
            'data_wait_ms': 5.0,
            'forward_ms': 12.0,
            'backward_ms': 10.0,
            'optimizer_ms': 3.0,
            'total_ms': 30.0,
        }
    )
    view.metrics_message.emit(
        {
            'type': 'system_memory',
            'ram_mb': 256.0,
            'vram_allocated_mb': 128.0,
            'vram_reserved_mb': 192.0,
        }
    )
    view.metrics_message.emit(
        {
            'type': 'train_batch_preview',
            'sample_name': 'frame_001.png',
            'image': np.full((32, 32), 128, dtype=np.uint8),
            'label': np.full((32, 32), 255, dtype=np.uint8),
        }
    )

    for i in range(250):
        view.metrics_message.emit(
            {'type': 'train_batch', 'epoch': 1, 'batch_index': i + 1, 'loss': 1.0 - i / 1000.0}
        )

    assert len(view.metrics_panel._train_epoch_points) == 1
    assert len(view.metrics_panel._val_epoch_points) == 1
    assert len(view.metrics_panel._val_iou_points) == 1
    assert len(view.metrics_panel._val_dice_points) == 1
    assert len(view._batch_points_by_epoch[1]) == 250
    assert view.epoch_progress_bar.value() == 20
    assert view.batch_progress_bar.value() == 25
    assert view.recognition_progress_bar.value() == 25
    assert "IoU: 80.00%" in view.validation_quality_label.text()
    assert "total: 30.0" in view.performance_label.text()
    assert "RAM: 256 МБ" in view.memory_usage_label.text()
    assert "VRAM: 128/192 МБ" in view.memory_usage_label.text()
    assert "33.33 batch/s" in view.memory_usage_label.text()
    assert view.preview_image_label.pixmap() is not None
    assert view.preview_label_label.pixmap() is not None
    assert "frame_001.png" in view.preview_frame_name_label.text()
    assert view.preview_image_title_label.text()
    assert view.preview_label_title_label.text()
    assert view.preview_output_title_label.text()


def test_main_view_recognition_preview_uses_two_columns(qapp):
    view = MainView(QWidget())
    view.connect_internal_signals()

    view.metrics_message.emit(
        {
            'type': 'recognition_preview',
            'sample_name': 'recognized_frame_007.png',
            'image': np.full((40, 40), 96, dtype=np.uint8),
            'outputs': np.full((40, 40), 255, dtype=np.uint8),
        }
    )

    assert view.preview_image_label.pixmap() is not None
    assert view.preview_output_label.pixmap() is not None
    assert "recognized_frame_007.png" in view.preview_frame_name_label.text()
    assert view.preview_label_column_widget.isHidden()

    view.metrics_message.emit(
        {
            'type': 'train_batch_preview',
            'sample_name': 'train_frame_001.png',
            'image': np.full((40, 40), 32, dtype=np.uint8),
            'label': np.full((40, 40), 128, dtype=np.uint8),
            'outputs': np.full((40, 40), 224, dtype=np.uint8),
        }
    )

    assert not view.preview_label_column_widget.isHidden()
    assert view.preview_label_label.pixmap() is not None


def test_main_view_keeps_sample_count_label_hidden(qapp):
    view = MainView(QWidget())

    view.set_samples_count_loading()
    assert view.sample_count_top_label.text().strip()
    assert view.sample_count_top_label.isHidden() is True

    view.set_samples_count(42)
    assert "42" in view.sample_count_top_label.text()
    assert view.sample_count_top_label.isHidden() is True

    view.apply_ui_language('en')
    assert "42" in view.sample_count_top_label.text()


def test_main_view_wraps_central_content_in_scroll_area(qapp):
    view = MainView(QWidget())

    central_widget = view.centralWidget()

    assert isinstance(central_widget, QScrollArea)
    assert central_widget.widget() is view._central_content


def test_clickable_label_does_not_force_window_width(qapp):
    label = ClickableLabel()
    full_path = r'D:\very\long\folder\structure\with\many\nested\directories\and\file_name_that_should_not_define_window_minimum_width.ext'
    label.setText(full_path)

    assert label.text() == full_path
    assert label.minimumSizeHint().width() == 0
    assert label.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored


def test_main_view_batch_points_are_sparsified_after_1000(qapp):
    view = MainView(QWidget())
    view.connect_internal_signals()

    captured: dict[str, object] = {}

    def _capture(epoch: int, points):
        captured["epoch"] = epoch
        captured["points"] = list(points)

    view.metrics_panel.set_batch_points = _capture

    for i in range(1001):
        view.metrics_message.emit(
            {'type': 'train_batch', 'epoch': 1, 'batch_index': i + 1, 'loss': 1.0 - i / 5000.0}
        )

    assert len(view._batch_points_by_epoch[1]) == 1001
    assert captured["epoch"] == 1
    points = captured["points"]
    assert isinstance(points, list)
    assert len(points) == 501


def test_main_view_log_history_is_capped(qapp):
    view = MainView(QWidget())
    view.connect_internal_signals()

    for i in range(MAX_LOG_MESSAGES + 25):
        view.log_message.emit(f"log message {i}")

    assert view.log_layout.count() == MAX_LOG_MESSAGES
    first_item = view.log_layout.itemAt(0)
    first_widget = first_item.widget() if first_item is not None else None
    assert first_widget is not None
    assert first_widget.text() == "log message 25"


def test_main_view_status_bar_shows_last_log_message(qapp):
    view = MainView(QWidget())
    view.connect_internal_signals()

    view.log_message.emit("background indexing started")

    assert view.statusBar().currentMessage() == "background indexing started"


def test_main_view_recognition_speed_label_updates_and_resets(qapp, monkeypatch):
    view = MainView(QWidget())
    view.connect_internal_signals()

    timestamps = iter([100.0, 102.0])
    monkeypatch.setattr("view.main_window.time.perf_counter", lambda: next(timestamps))

    view.metrics_message.emit({'type': 'recognition_progress', 'current': 0, 'total': 12})
    assert "—" in view.recognition_speed_label.text()

    view.metrics_message.emit({'type': 'recognition_progress', 'current': 6, 'total': 12})
    assert "3.00" in view.recognition_speed_label.text()
    assert "изобр./с" in view.recognition_speed_label.text()

    view._switch_start_stop(True)
    assert "—" in view.recognition_speed_label.text()


def test_metrics_panel_can_be_restored_from_view_menu(qapp):
    view = MainView(QWidget())
    view.show()
    qapp.processEvents()

    view.metrics_panel.close()
    qapp.processEvents()
    assert view.metrics_panel.isHidden()

    menubar = view.menuBar()
    assert menubar is not None
    view_menu = next((action.menu() for action in menubar.actions() if action.text() == "Вид"), None)
    assert view_menu is not None

    metrics_action = next((action for action in view_menu.actions() if action.text() == "Панель графиков"), None)
    assert metrics_action is not None

    metrics_action.trigger()
    qapp.processEvents()
    assert not view.metrics_panel.isHidden()


def test_main_view_file_menu_exposes_open_action(qapp):
    view = MainView(QWidget())
    view.connect_internal_signals()

    captured = []
    view.open_config_requested.connect(lambda: captured.append(True))

    menubar = view.menuBar()
    assert menubar is not None
    file_menu = next((action.menu() for action in menubar.actions() if action.text() == "Файл"), None)
    assert file_menu is not None

    open_action = next((action for action in file_menu.actions() if action.text() == "Открыть..."), None)
    assert open_action is not None

    open_action.trigger()
    assert captured == [True]


def test_main_view_queue_widget_and_start_stop_visibility(qapp):
    view = MainView(QWidget())
    view.show()
    qapp.processEvents()
    view.set_task_queue_items(['#1 | train_only | в очереди', '#2 | recognition | на паузе'], selected_row=1)

    assert view.queue_list.count() == 2
    assert view.get_selected_queue_row() == 1

    view._switch_start_stop(True)
    assert view.btn_start.isHidden() is False
    assert view.btn_stop.isHidden() is False


def test_main_view_queue_context_menu_emits_properties_signal(qapp, monkeypatch):
    view = MainView(QWidget())
    view.connect_internal_signals()
    view.show()
    qapp.processEvents()
    view.set_task_queue_items(['#1 | train_only | queued'])

    captured_rows: list[int] = []
    view.queue_properties_requested.connect(captured_rows.append)

    monkeypatch.setattr(
        'view.main_window.QMenu.exec',
        lambda menu, *_args, **_kwargs: menu.actions()[1],
    )

    item = view.queue_list.item(0)
    assert item is not None
    position = view.queue_list.visualItemRect(item).center()
    view._show_queue_context_menu(position)

    assert captured_rows == [0]


def test_main_view_simple_mode_hides_docks_and_shows_presets(qapp):
    view = MainView(QWidget())
    view.connect_internal_signals()
    view.show()
    qapp.processEvents()

    view.apply_ui_mode('simple')
    qapp.processEvents()

    assert view.current_ui_mode() == 'simple'
    assert view.simple_workflows_group.isVisible()
    assert view.metrics_panel.isHidden()
    assert view.log_dock.isHidden()
    assert not view.model_path.isHidden()
    assert view.le_epochs.isHidden()
    assert view.simple_workflow_label.text()

    view.btn_simple_contacts.click()
    assert view.btn_simple_contacts.text() in view.simple_workflow_label.text()
    assert view.btn_simple_contacts.isChecked() is True
    assert view.btn_simple_conductors.isChecked() is False

    view.apply_ui_mode('advanced')
    qapp.processEvents()

    assert view.current_ui_mode() == 'advanced'
    assert not view.simple_workflows_group.isVisible()
    assert not view.metrics_panel.isHidden()
    assert not view.log_dock.isHidden()
    assert not view.model_path.isHidden()
    assert view.le_epochs.isHidden()


def test_main_view_work_mode_visibility_tracks_model_and_epochs(qapp):
    view = MainView(QWidget())
    view.show()
    qapp.processEvents()
    view.apply_ui_mode('advanced')
    qapp.processEvents()

    view.apply_work_mode('train_only')
    qapp.processEvents()
    assert view.model_path.isHidden()
    assert view.le_epochs.isHidden()
    assert not view.sample_path_group.isHidden()

    view.apply_work_mode('recognition_only')
    qapp.processEvents()
    assert not view.model_path.isHidden()
    assert view.le_epochs.isHidden()
    assert view.sample_path_group.isHidden()

    view.apply_work_mode('further_training')
    qapp.processEvents()
    assert not view.model_path.isHidden()
    assert view.le_epochs.isHidden()
    assert not view.sample_path_group.isHidden()


def test_main_view_places_recursive_search_checkbox_next_to_source_path(qapp):
    view = MainView(QWidget())
    view.show()
    qapp.processEvents()

    assert view.source_path_row.layout().indexOf(view.lbl_source) >= 0
    assert view.source_path_row.layout().indexOf(view.recursive_file_search_check_box) >= 0
    assert view.recursive_file_search_check_box.text() == 'Искать в дочерних каталогах'

    view.set_recursive_file_search(True)

    assert view.is_recursive_file_search_enabled() is True


def test_main_view_restores_ui_mode_from_persisted_settings(qapp, monkeypatch):
    settings_dir = Path('d:/PyCharm/neuralimage-feature-no_cut_dataset/.test_runtime/ui_mode_persist')
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_file = settings_dir / 'NeuralImage_MainWindow.ini'
    if settings_file.exists():
        settings_file.unlink()
    monkeypatch.setenv('NEURALIMAGE_SETTINGS_DIR', str(settings_dir))

    first_view = MainView(QWidget())
    first_view.apply_ui_mode('advanced')
    assert first_view.current_ui_mode() == 'advanced'
    first_view.close()

    second_view = MainView(QWidget())
    assert second_view.current_ui_mode() == 'advanced'
    second_view.close()
