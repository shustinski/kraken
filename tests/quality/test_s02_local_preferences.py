"""S02: local Hub preferences and workspace settings through the GUI."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QComboBox, QDialogButtonBox, QLineEdit, QMessageBox

from kraken_core.settings import QSettings as HubSettings
from kraken_core.settings import configure_settings_directory
from kraken_hub.composition.catalog import build_workspace_service
from kraken_hub.presentation.qt import manager_app
from kraken_hub.presentation.qt import preferences_dialog
from kraken_hub.presentation.qt.parts import run_manager_gui as manager_gui
from tests.helpers.qt_harness import click_button, click_menu, launch_hub, modal_action


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _settings():
    return HubSettings("Kraken", "KrakenHub")


def test_s02_preferences_dialog_persists_server_url_and_theme(qapp, tmp_path: Path, monkeypatch):
    class Secrets:
        values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

        def delete(self, key):
            self.values.pop(key, None)

    secrets = Secrets()
    monkeypatch.setattr(preferences_dialog, "program_secret_store", lambda: secrets)
    with launch_hub(tmp_path / "hub") as hub:
        hub.controller._connect_program_account = lambda: None
        preferences_dialog.save_program_accounts(
            [preferences_dialog.ProgramAccount("account-1", "local", "operator", "Operator", "private-token", True)],
            _settings(),
            secrets,
        )

        def fill(dialog):
            assert dialog.objectName() == "programPreferences"
            assert dialog.findChild(preferences_dialog.QListWidget, "preferenceAccounts").count() == 1
            dialog.findChild(QLineEdit, "preferenceServerUrl").setText("https://example.test:8443/")
            theme = dialog.findChild(QComboBox, "preferenceTheme")
            theme.setCurrentIndex(theme.findData("dark"))
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()

        modal_action(lambda: click_menu(hub.window, "Вид → Настройки…"), fill)
        settings = _settings()
        assert settings.value("server/url") == "https://example.test:8443"
        assert settings.value("appearance/theme") == "dark"
        settings.sync()
        raw_settings = Path(settings.fileName()).read_text(encoding="utf-8")
        assert "https://example.test:8443" in raw_settings
        assert "private-token" not in raw_settings
        assert secrets.values["account-1"] == b"private-token"


def test_s02_matrix_visual_mode_persisted(qapp, tmp_path: Path):
    with launch_hub(tmp_path / "hub") as hub:
        click_menu(hub.window, "Вид → Рамка ячейки → Качество")
        click_menu(hub.window, "Вид → Заполнение ячейки → Статус")
        assert hub.window.visual_modes() == ("quality", "status")
        assert _settings().value("matrix/border-mode") == "quality"
        assert _settings().value("matrix/fill-mode") == "status"


def test_s02_update_source_dialog_stores_manifest_url(qapp, tmp_path: Path, monkeypatch):
    import kraken_core.settings as settings_module
    from tests.helpers.qt_harness import _HarnessAgentRuntime

    previous_directory = settings_module._directory
    configure_settings_directory(tmp_path / "settings")
    profile_ini = tmp_path / "kraken.ini"
    profile_ini.write_text("[profiling]\nenabled = false\nmain_window = true\n", encoding="utf-8")
    monkeypatch.setenv("KRAKEN_CONFIG", str(profile_ini))
    service = build_workspace_service(data_dir=tmp_path / "hub")
    service.create_initial_account("operator", "Operator", "secret")
    session = service.login("operator", "secret")
    monkeypatch.setattr(manager_gui, "_development_session", lambda _local: session)
    monkeypatch.setattr(manager_gui, "_connect_saved_server", lambda *_args: None)
    monkeypatch.setattr(manager_app, "LocalAgentRuntime", _HarnessAgentRuntime)
    errors = []

    def drive():
        try:
            window = next(
                window for window in qapp.topLevelWidgets() if window.windowTitle() == "Kraken" and window.isVisible()
            )

            def fill(dialog):
                assert dialog.windowTitle() == "Источник обновлений"
                dialog.findChild(QLineEdit).setText("file:///manifest.json")
                dialog.accept()

            modal_action(lambda: click_menu(window, "Вид → Источник обновлений…"), fill)
        except Exception as exc:
            errors.append(exc)
        finally:
            qapp.quit()

    QTimer.singleShot(0, drive)
    try:
        assert (
            manager_gui.run_manager_gui([], service=service, application_version="0", plugin_launcher=lambda *_: None)
            == 0
        )
        assert not errors, errors
        assert _settings().value("updates/url") == "file:///manifest.json"
        report = (tmp_path / "profiling.log").read_text(encoding="utf-8")
        assert "[main_window]" in report
        assert "ready=yes" in report
    finally:
        # A top-level quit leaves Qt's nested login loops stopped until a new
        # application loop starts; reset it before the next GUI test.
        QTimer.singleShot(0, qapp.quit)
        qapp.exec()
        settings_module._directory = previous_directory


def test_s02_configure_roots_writes_settings(qapp, tmp_path: Path, monkeypatch):
    source = tmp_path / "sources"
    derived = tmp_path / "derived"
    source.mkdir()
    derived.mkdir()
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes)
    with launch_hub(tmp_path / "hub") as hub:
        admin = hub.window.page("administration")

        def fill(dialog):
            assert dialog.objectName() == "workspaceRootsDialog"
            dialog.findChild(QLineEdit, "workspaceSourceRoot").setText(str(source))
            dialog.findChild(QLineEdit, "workspaceDerivedRoot").setText(str(derived))
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()

        modal_action(lambda: click_button(admin, "Настроить хранилища…"), fill)
        assert _settings().value("workspace/source-root") == str(source)
        assert _settings().value("workspace/derived-root") == str(derived)
