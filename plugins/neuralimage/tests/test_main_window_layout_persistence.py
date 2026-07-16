import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from neuralimage.view.main_window import MainView
from neuralimage.view.settings_panel import SettingsPanel


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_main_window_and_dock_layout_roundtrip(qapp, monkeypatch, tmp_path):
    monkeypatch.setenv("NEURALIMAGE_SETTINGS_DIR", str(tmp_path))

    first = MainView(SettingsPanel())
    first.resize(980, 720)
    first.move(70, 80)
    first.show()
    qapp.processEvents()

    assert {dock.objectName() for dock in first._managed_docks()} == {
        "taskQueueDock",
        "trainingMetricsDock",
        "logDock",
        "settingsDock",
    }

    first.log_dock.setFloating(True)
    first.log_dock.setGeometry(140, 150, 330, 260)
    first.queue_dock.hide()
    qapp.processEvents()
    saved_window_size = first.size()
    saved_log_size = first.log_dock.size()
    first._save_window_layout()
    first.hide()

    restored = MainView(SettingsPanel())
    restored.show()
    qapp.processEvents()

    assert restored.size() == saved_window_size
    assert restored.log_dock.isFloating()
    assert abs(restored.log_dock.width() - saved_log_size.width()) <= 2
    assert abs(restored.log_dock.height() - saved_log_size.height()) <= 2
    assert restored.queue_dock.isHidden()
    assert restored.dockWidgetArea(restored.settings_dock) == Qt.DockWidgetArea.RightDockWidgetArea

    restored.hide()


def test_main_window_size_is_saved_after_live_resize(qapp, monkeypatch, tmp_path):
    monkeypatch.setenv("NEURALIMAGE_SETTINGS_DIR", str(tmp_path))

    first = MainView(SettingsPanel())
    first.show()
    qapp.processEvents()
    first.resize(1080, 760)
    qapp.processEvents()
    saved_size = first.size()

    QTest.qWait(350)
    first.hide()

    restored = MainView(SettingsPanel())
    restored.show()
    qapp.processEvents()

    assert restored.size() == saved_size
    restored.hide()
