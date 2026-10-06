"""An update folder the installer wrote into settings.ini survives Qt, and a missing one is reported."""
from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QWidget


def _ini(tmp_path: Path, line: bytes) -> Path:
    path = tmp_path / "settings.ini"
    path.write_bytes(b"[General]\r\nselected_channel=beta\r\n" + line + b"\r\n")
    return path


def _qt_root(path: Path) -> str:
    return str(QSettings(str(path), QSettings.Format.IniFormat).value("update_root", "", type=str))


@pytest.fixture
def settings_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KARAKAL_SETTINGS_DIR", str(tmp_path))
    monkeypatch.delenv("KARAKAL_UPDATE_ROOT", raising=False)
    return tmp_path


def test_qt_alone_damages_an_installer_path(settings_dir: Path) -> None:
    # The bug this guards against: Qt reads the backslash as an escape.
    assert _qt_root(_ini(settings_dir, rb"update_root=G:\ProgramStore")) == "G:rogramStore"


def test_installer_backslash_path_is_kept(settings_dir: Path) -> None:
    from karakal.updater import load_karakal_update_root

    path = _ini(settings_dir, rb"update_root=G:\ProgramStore")
    assert load_karakal_update_root() == "G:/ProgramStore"
    # Saved through Qt, so later Qt saves keep it.
    assert _qt_root(path) == "G:/ProgramStore"
    assert "selected_channel=beta" in path.read_text(encoding="utf-8")


def test_network_path_in_windows_code_page_is_kept(settings_dir: Path) -> None:
    from karakal.updater import load_karakal_update_root

    _ini(settings_dir, "update_root=\\\\ОФИС-ПК\\Karakal".encode("cp1251"))
    assert load_karakal_update_root() == "//ОФИС-ПК/Karakal"


def test_paths_written_by_qt_are_left_alone(settings_dir: Path) -> None:
    from karakal.updater import load_karakal_update_root, save_karakal_update_root

    save_karakal_update_root(r"D:\SVZH\ProgramStore")
    before = (settings_dir / "settings.ini").read_bytes()
    assert load_karakal_update_root() == r"D:\SVZH\ProgramStore"
    assert (settings_dir / "settings.ini").read_bytes() == before

    _ini(settings_dir, b"update_root=G:/ProgramStore")
    assert load_karakal_update_root() == "G:/ProgramStore"


def test_app_settings_open_repairs_first(settings_dir: Path) -> None:
    from karakal.infra.services import default_settings

    _ini(settings_dir, rb"update_root=G:\ProgramStore")
    settings = default_settings()
    settings.setValue("x", 1)
    settings.sync()
    assert _qt_root(settings_dir / "settings.ini") == "G:/ProgramStore"


def test_manual_check_reports_a_missing_folder(settings_dir: Path, qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    from PyQt6.QtWidgets import QMessageBox

    from karakal import updater

    shown: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, text, *args: shown.append(text))
    monkeypatch.setattr(QMessageBox, "information", lambda _parent, _title, text, *args: shown.append(text))
    missing = settings_dir / "no-such-share"
    updater.save_karakal_update_root(str(missing).replace("\\", "/"))
    window = QWidget()
    qtbot.addWidget(window)
    controller = updater.create_karakal_update_controller(window)

    # tests/conftest.py stubs check_for_updates, so ask the folder resolver directly.
    assert controller._resolve_manifest_url("beta", True) is None
    assert len(shown) == 1
    assert shown[0].startswith("Папка обновлений недоступна")
    assert "no-such-share" in shown[0]

    # The start-up check stays quiet.
    assert controller._resolve_manifest_url("beta", False) is None
    assert len(shown) == 1
