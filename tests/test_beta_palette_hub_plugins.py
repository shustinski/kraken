"""Палитра W3 — запуск процесса плагина из Hub (live UI). S13 (Hub)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtWidgets import QFrame, QLabel, QPushButton

from kraken_core.plugins import build_plugin_inventory, scan_plugin_directory
from kraken_hub.composition.app import launch_plugin
from kraken_hub.infrastructure.remote.client import RemoteServerProjectService
from kraken_hub.presentation.qt.parts.deletion_reason_dialog import _plugin_panel
from kraken_manager.presentation.qt import ProjectManagerShell
from tests.test_beta_palette_hub_ui_project_cycle import _isolate_settings, _ui_config


@pytest.fixture
def stand(monkeypatch, tmp_path, qtbot):
    config = _ui_config()
    _isolate_settings(monkeypatch, tmp_path)
    owner = RemoteServerProjectService.authenticate_local(
        config["server_url"],
        username=config["username"],
        password=config["password"],
        data_dir=tmp_path / "owner",
    )
    shell = ProjectManagerShell()
    qtbot.addWidget(shell)
    launched: list = []

    def launching(plugin, arguments=()):
        process = launch_plugin(plugin, arguments=arguments)
        launched.append(process)
        return process

    plugins = build_plugin_inventory(scan_plugin_directory(Path(__file__).resolve().parents[1] / "plugins"))
    shell.page("plugins").set_content(_plugin_panel(plugins, launch_plugin=launching))
    shell.show()
    try:
        yield type("Stand", (), {"owner": owner, "shell": shell, "launched": launched})()
    finally:
        for process in launched:
            if process.poll() is None:
                process.kill()
        owner.stop_sync()
        shell.close_guard = None
        shell.close()


def test_s13_plugins_dialog_opens_plugin_process(stand, qtbot):
    page = stand.shell.page("plugins")
    row = next(
        item for item in page.findChildren(QFrame, "pluginInventoryRow")
        if "Contour" in item.findChild(QLabel).text()
    )
    button = row.findChild(QPushButton, )
    assert button.text() == "Открыть" and button.isEnabled()
    before = {str(job.id) for job in stand.owner.plugin_jobs()}
    button.click()
    qtbot.wait(2000)
    assert stand.launched and stand.launched[0].poll() is None
    assert stand.launched[0].pid != __import__("os").getpid()
    assert {str(job.id) for job in stand.owner.plugin_jobs()} == before
    stand.shell.close()
    qtbot.waitUntil(lambda: stand.launched[0].poll() is not None, timeout=10000)
