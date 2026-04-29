from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings

from csliser.domain.models import FileOperation
from csliser.infrastructure.settings_store import CSliserPreset, WindowSettings, WindowSettingsStore
from csliser.presentation.qt.window import (
    CSliserWindow,
    duplicate_source_folder_names,
    legacy_source_extension_state,
)


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


def test_window_config_keeps_multiple_source_folders(qtbot, tmp_path) -> None:
    settings_path = tmp_path / "csliser.ini"
    store = WindowSettingsStore(lambda: QSettings(str(settings_path), QSettings.Format.IniFormat))
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    destination = tmp_path / "out"
    source_a.mkdir()
    source_b.mkdir()

    window = CSliserWindow(settings_store=store)
    qtbot.addWidget(window)
    window._source_folders[source_a] = {"all_extensions": (".jpg",), "active_extensions": {".jpg"}}
    window._source_folders[source_b] = {"all_extensions": (".cif",), "active_extensions": {".cif"}}
    window._source_ui()
    window.destination_folder_lineedit.setText(str(destination))
    window.first_frame_lineedit.setText("1")
    window.frames_in_row_lineedit.setText("135")

    config = window._config(FileOperation.COPY)

    assert [source.path for source in config.sources] == [source_a, source_b]
    assert [source.extensions for source in config.sources] == [(".jpg",), (".cif",)]


def test_window_editing_one_source_path_does_not_drop_other_sources(qtbot, tmp_path) -> None:
    settings_path = tmp_path / "csliser.ini"
    store = WindowSettingsStore(lambda: QSettings(str(settings_path), QSettings.Format.IniFormat))
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    source_c = tmp_path / "c"
    for source in (source_a, source_b, source_c):
        source.mkdir()

    window = CSliserWindow(settings_store=store)
    qtbot.addWidget(window)
    window._source_folders[source_a] = {"all_extensions": (".jpg",), "active_extensions": {".jpg"}}
    window._source_folders[source_b] = {"all_extensions": (".cif",), "active_extensions": {".cif"}}
    window._source_ui()

    window._source_line_edits[0].setText(str(source_c))
    config = window._config(FileOperation.DELETE)

    assert [source.path for source in config.sources] == [source_c, source_b]


def test_window_keeps_legacy_button_colors_and_folder_delete_size(qtbot, tmp_path) -> None:
    settings_path = tmp_path / "csliser.ini"
    store = WindowSettingsStore(lambda: QSettings(str(settings_path), QSettings.Format.IniFormat))
    source = tmp_path / "source"
    source.mkdir()

    window = CSliserWindow(settings_store=store)
    qtbot.addWidget(window)
    window._source_folders[source] = {"all_extensions": (".jpg",), "active_extensions": {".jpg"}}
    window._source_ui()
    window.destination_folder_lineedit.setText(str(tmp_path / "out"))
    window.first_frame_lineedit.setText("1")
    window.frames_in_row_lineedit.setText("135")
    window._refresh_state()

    delete_folder_button = window.copy_groupbox_layout.itemAtPosition(0, 2).widget()

    assert "153,255,204" in window.plus_dirs_button.styleSheet()
    assert "153,255,204" in window.copy_button.styleSheet()
    assert "204, 204, 204" in window.move_button.styleSheet()
    assert "255, 102, 102" in window.delete_button.styleSheet()
    assert delete_folder_button is not None
    assert delete_folder_button.minimumWidth() >= 48
    assert "255, 102, 102" in delete_folder_button.styleSheet()


def test_window_keeps_plus_button_after_rebuilding_source_list(qtbot, tmp_path) -> None:
    settings_path = tmp_path / "csliser.ini"
    store = WindowSettingsStore(lambda: QSettings(str(settings_path), QSettings.Format.IniFormat))
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    source_a.mkdir()
    source_b.mkdir()

    window = CSliserWindow(settings_store=store)
    qtbot.addWidget(window)
    window._add_source_path(source_a)
    window._source_ui()
    window._add_source_path(source_b)
    window._source_ui()
    qtbot.wait(0)

    assert window.copy_groupbox_layout.indexOf(window.plus_dirs_button) != -1
    assert window.plus_dirs_button.isVisibleTo(window.copy_dir_groupbox)
    assert "153,255,204" in window.plus_dirs_button.styleSheet()


def test_window_directory_dialog_uses_qfiledialog_not_native_explorer(qtbot, tmp_path) -> None:
    settings_path = tmp_path / "csliser.ini"
    store = WindowSettingsStore(lambda: QSettings(str(settings_path), QSettings.Format.IniFormat))
    window = CSliserWindow(settings_store=store)
    qtbot.addWidget(window)

    dialog = window._create_directory_dialog("folders", initial=tmp_path, multi_select=True)

    assert dialog.testOption(dialog.Option.DontUseNativeDialog)
    assert dialog.testOption(dialog.Option.ShowDirsOnly)
    assert dialog.fileMode() == dialog.FileMode.Directory


def test_legacy_source_extension_defaults_match_static_drive_rules() -> None:
    extensions, active = legacy_source_extension_state(Path("X:/project/source"), dynamic_extensions=False)
    assert extensions == (".jpg", ".bmp", ".cif")
    assert active == {".jpg"}

    extensions, active = legacy_source_extension_state(Path("Z:/project/source"), dynamic_extensions=False)
    assert extensions == (".jpg", ".bmp", ".cif")
    assert active == {".cif"}

    extensions, active = legacy_source_extension_state(Path("D:/project/source"), dynamic_extensions=False)
    assert extensions == (".jpg", ".bmp", ".cif")
    assert active == set()


def test_legacy_source_extension_defaults_match_dynamic_rules(tmp_path) -> None:
    (tmp_path / "chip_000001.jpg").write_text("jpg", encoding="utf-8")
    (tmp_path / "chip_000001.cif").write_text("cif", encoding="utf-8")
    (tmp_path / "chip_000001.txt").write_text("txt", encoding="utf-8")

    extensions, active = legacy_source_extension_state(tmp_path, dynamic_extensions=True)

    assert extensions == (".cif", ".jpg", ".txt")
    assert active == {".cif", ".jpg"}


def test_duplicate_source_folder_names_detects_legacy_collision() -> None:
    duplicate = duplicate_source_folder_names([Path("X:/a/source"), Path("Z:/b/source")])

    assert duplicate == (Path("X:/a/source"), Path("Z:/b/source"))


def test_window_preset_roundtrip_restores_full_legacy_payload(qtbot, tmp_path) -> None:
    settings_path = tmp_path / "csliser.ini"
    store = WindowSettingsStore(lambda: QSettings(str(settings_path), QSettings.Format.IniFormat))
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    source_a.mkdir()
    source_b.mkdir()
    store.save_preset(
        "full",
        CSliserPreset(
            sources=(str(source_a), str(source_b)),
            destination=str(tmp_path / "out"),
            frames="1-10;20",
            row_frames="135",
        ),
        include_sources=True,
    )

    window = CSliserWindow(settings_store=store)
    qtbot.addWidget(window)
    window._apply_preset(store.load_presets(include_sources=True)["full"], include_sources=True)

    assert list(window._source_folders) == [source_a, source_b]
    assert window.destination_folder_lineedit.text() == str(tmp_path / "out")
    assert window.first_frame_lineedit.text() == "1-10;20"
    assert window.frames_in_row_lineedit.text() == "135"
