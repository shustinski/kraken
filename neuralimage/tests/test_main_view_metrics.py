import pytest
import numpy as np

pytest.importorskip('PyQt6')

from PyQt6.QtWidgets import QApplication, QWidget

from view.main_window import MainView


@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_view_metrics_are_collected_and_capped(qapp):
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
    assert len(view._batch_points_by_epoch[1]) == 200
    assert view.epoch_progress_bar.value() == 20
    assert view.batch_progress_bar.value() == 25
    assert view.recognition_progress_bar.value() == 25
    assert "IoU: 80.00%" in view.validation_quality_label.text()
    assert "total: 30.0" in view.performance_label.text()
    assert view.preview_image_label.pixmap() is not None
    assert view.preview_label_label.pixmap() is not None


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
