from __future__ import annotations

from PyQt6.QtCore import QSettings

from csliser.infrastructure.settings_store import WindowSettings, WindowSettingsStore
from csliser.presentation.qt.window import CSliserWindow


def test_window_keeps_original_csliser_layout_labels(qtbot, tmp_path) -> None:
    settings_path = tmp_path / "csliser.ini"
    store = WindowSettingsStore(lambda: QSettings(str(settings_path), QSettings.Format.IniFormat))
    window = CSliserWindow(settings_store=store)
    qtbot.addWidget(window)

    assert window.windowTitle() == "CSlicer"
    assert window.width() == 800
    assert window.height() == 300
    assert window.source_select_label.text() == "Копировать:"
    assert window.destination_folder_label.text() == "Поместить в:"
    assert window.first_frame_label.text() == "Кадры:"
    assert window.frames_in_row_label.text() == "Кадров в строке:"
    assert window.copy_groupbox.title() == "Режим копирования"
    assert window.copy_button.text() == "Копировать"
    assert window.move_button.text() == "Переместить"
    assert window.delete_button.text() == "Удалить"
    assert window.add_extension_l.text() == "Добавить расширение"


def test_window_restores_font_size_from_settings(qtbot, tmp_path) -> None:
    settings_path = tmp_path / "csliser.ini"
    store = WindowSettingsStore(lambda: QSettings(str(settings_path), QSettings.Format.IniFormat))
    store.save(WindowSettings(font_size=22))

    window = CSliserWindow(settings_store=store)
    qtbot.addWidget(window)

    assert window.font.pixelSize() == 22
    assert "font-size: 22px" in window.styleSheet()
