"""Update button (grey / blue / downloading), the offer at start and the Help menu."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QMessageBox

import karakal.app.update_flow as update_flow
from karakal.app.update_flow import STATE_AVAILABLE, STATE_IDLE, check_update_now, notes_since
from karakal.updater import save_karakal_update_channel

NEWER = "9.9.9001-beta0"
OLDER = "0.0.1-beta0"


def _share(tmp_path: Path, version: str) -> Path:
    share = tmp_path / "share"
    beta = share / "beta"
    beta.mkdir(parents=True, exist_ok=True)
    installer = beta / f"Karakal-setup-{version}.exe"
    installer.write_bytes(b"installer" * 1000)
    releases = [
        {"version": version, "download_url": installer.name, "channel": "beta", "notes": "### Новое\n- синяя кнопка"},
        {"version": OLDER, "download_url": "", "channel": "beta", "notes": "старое"},
    ]
    (beta / "version.json").write_text(
        json.dumps({"version": version, "channel": "beta", "releases": releases}), encoding="utf-8"
    )
    return share


@pytest.fixture
def update_folder(tmp_path, monkeypatch):
    def make(version: str) -> Path:
        share = _share(tmp_path, version)
        monkeypatch.setenv("KARAKAL_UPDATE_ROOT", str(share))
        return share

    monkeypatch.setenv("KARAKAL_SETTINGS_DIR", str(tmp_path / "settings_home"))
    save_karakal_update_channel("beta")
    return make


def _widget(tmp_path, qtbot):
    from karakal.app.main_window import KarakalWidget

    widget = KarakalWidget(settings=QSettings(str(tmp_path / "k.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    return widget


def test_button_turns_blue_when_a_newer_version_waits(tmp_path, qtbot, update_folder) -> None:
    update_folder(NEWER)
    widget = _widget(tmp_path, qtbot)
    button = widget.update_tool_button
    assert button.menu() is None
    manager = widget._update_manager
    assert manager.state == STATE_IDLE and button.styleSheet() == ""
    manager.check(manual=False)
    qtbot.waitUntil(lambda: manager.state == STATE_AVAILABLE, timeout=10000)
    assert NEWER in button.text()
    assert "#0e639c" in button.styleSheet()


def test_button_stays_grey_without_a_newer_version(tmp_path, qtbot, update_folder) -> None:
    update_folder(OLDER)
    widget = _widget(tmp_path, qtbot)
    manager = widget._update_manager
    manager.check(manual=False)
    qtbot.waitUntil(lambda: manager._check_thread is None, timeout=10000)
    assert manager.state == STATE_IDLE
    assert widget.update_tool_button.styleSheet() == ""


def test_start_check_finds_only_newer_versions(tmp_path, qtbot, update_folder) -> None:
    update_folder(NEWER)
    info = check_update_now(timeout_ms=5000)
    assert info is not None and info.version == NEWER
    assert "синяя кнопка" in notes_since(info, "0.3.9409-beta0")
    assert "старое" not in notes_since(info, "0.3.9409-beta0")
    update_folder(OLDER)
    assert check_update_now(timeout_ms=5000) is None


def test_blue_button_copies_installer_runs_it_and_closes(tmp_path, qtbot, update_folder, monkeypatch) -> None:
    update_folder(NEWER)
    monkeypatch.setattr(update_flow, "get_update_staging_dir", lambda _app_id: tmp_path / "staging")
    launched: list[str] = []
    monkeypatch.setattr(update_flow, "launch_update_installer", lambda path: launched.append(str(path)))
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes)
    widget = _widget(tmp_path, qtbot)
    manager = widget._update_manager
    closed: list[bool] = []
    manager._close_app = lambda: closed.append(True)
    manager.check(manual=False)
    qtbot.waitUntil(lambda: manager.state == STATE_AVAILABLE, timeout=10000)
    widget.update_tool_button.click()
    qtbot.waitUntil(lambda: bool(closed), timeout=10000)
    assert launched and Path(launched[0]).parent == tmp_path / "staging"
    assert Path(launched[0]).read_bytes() == b"installer" * 1000


def test_install_waits_for_a_running_analysis(tmp_path, qtbot, update_folder, monkeypatch) -> None:
    update_folder(NEWER)
    widget = _widget(tmp_path, qtbot)
    busy = [True]
    manager = update_flow.KarakalUpdateManager(widget, widget._t, is_busy=lambda: busy[0], close_app=lambda: None)
    started: list[str] = []
    monkeypatch.setattr(manager, "_copy", lambda release: started.append(release.version))

    def choose_after(box):
        box.clickedButton = lambda: box.buttons()[0]
        return 0

    monkeypatch.setattr(QMessageBox, "exec", choose_after)
    manager.install(update_flow.ReleaseInfo(NEWER, "x.exe"))
    qtbot.wait(1500)
    assert started == []
    busy[0] = False
    qtbot.waitUntil(lambda: started == [NEWER], timeout=5000)


def test_help_menu_holds_channel_whats_new_versions_check_and_about(tmp_path, qtbot, update_folder) -> None:
    update_folder(OLDER)
    widget = _widget(tmp_path, qtbot)
    help_menu = widget._help_menu
    texts = [action.text() for action in help_menu.actions() if not action.isSeparator()]
    t = widget._t
    assert texts == [
        t("update.channel_menu"),
        t("update.whats_new"),
        t("update.choose_version"),
        t("update.check"),
        t("menu.help.about"),
    ]
    channels = [action.data() for action in widget._update_channel_menu.actions()]
    assert set(channels) >= {"stable", "beta"}
