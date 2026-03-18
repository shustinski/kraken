import pytest
import numpy as np

pytest.importorskip('PyQt6')

from PyQt6.QtWidgets import QApplication, QWidget

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
