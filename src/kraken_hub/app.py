from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib import parse as urlparse

from kraken_core.plugins import PluginMetadata, load_plugin_catalog
from kraken_core.qt import configure_application_identity
from kraken_core.runtime import current_platform, workspace_root
from kraken_core.styles import load_shared_stylesheet


def bundled_catalog_path() -> Path:
    return Path(__file__).resolve().parent / "resources" / "plugins.json"


def discover_catalog(explicit_path: str | None = None) -> str:
    if explicit_path:
        return explicit_path
    root = workspace_root()
    neuralimage_catalog = discover_neuralimage_channel_catalog(root)
    if neuralimage_catalog:
        return neuralimage_catalog
    local = root / "src" / "kraken_hub" / "resources" / "plugins.json"
    return str(local if local.exists() else bundled_catalog_path())


def discover_neuralimage_channel_catalog(root: Path) -> str:
    config_path = root / "plugins" / "neuralimage" / "resources" / "update_client.json"
    if not config_path.is_file():
        return ""
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return ""
    channels = payload.get("channels", {})
    default_channel = str(payload.get("default_channel", "stable") or "stable")
    manifest = ""
    if isinstance(channels, dict):
        manifest = str(channels.get(default_channel, "") or next(iter(channels.values()), "") or "")
    manifest = manifest or str(payload.get("manifest_url", "") or "")
    if not manifest:
        return ""
    if manifest.startswith(("http://", "https://")):
        return urlparse.urljoin(manifest, "plugins.json")
    return str((Path(manifest).expanduser().resolve().parent / "plugins.json"))


def build_launch_command(plugin: PluginMetadata, *, root: Path | None = None) -> list[str]:
    platform = current_platform()
    executable = plugin.executable_for(platform)
    if executable.path:
        return [str(Path(executable.path).expanduser())]
    plugin_root = (root or workspace_root()) / "plugins" / plugin.id
    if plugin_root.exists() and shutil.which("uv"):
        if (plugin_root / "__main__.py").is_file():
            return ["uv", "run", "python", "__main__.py"]
        return ["uv", "run", "python", "-m", plugin.id]
    if plugin_root.exists() and (plugin_root / "__main__.py").is_file():
        return [sys.executable, str(plugin_root / "__main__.py")]
    if executable.command:
        return [part.format(workspace=str(root or workspace_root()), plugin_id=plugin.id) for part in executable.command]
    return [sys.executable, "-m", plugin.id]


def launch_plugin(plugin: PluginMetadata) -> None:
    root = workspace_root()
    plugin_root = root / "plugins" / plugin.id
    cwd = plugin_root if plugin_root.exists() and not plugin.executable_for().path else root
    env = None
    if not plugin.executable_for().path:
        env = dict(**os.environ)
        python_paths = [str(root / "src"), str(root / "plugins" / plugin.id / "src")]
        env["PYTHONPATH"] = os.pathsep.join(python_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    subprocess.Popen(build_launch_command(plugin, root=root), cwd=str(cwd), env=env)


def print_catalog(plugins: list[PluginMetadata]) -> None:
    for plugin in plugins:
        state = "disabled" if not plugin.enabled else "enabled"
        print(f"{plugin.id}\t{plugin.version}\t{state}\t{plugin.display_name}")


def run_gui(plugins: list[PluginMetadata]) -> int:
    from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea, QVBoxLayout, QWidget

    app = QApplication(sys.argv)
    configure_application_identity(app, app_id="Kraken.Hub", icon_name="kraken")
    app.setStyleSheet(load_shared_stylesheet("dark_modern.qss"))
    window = QMainWindow()
    window.setWindowTitle("Kraken Hub")
    central = QWidget()
    layout = QVBoxLayout(central)
    for plugin in plugins:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.addWidget(QLabel(f"{plugin.display_name} ({plugin.version})"))
        status = "Disabled" if not plugin.enabled else "Ready"
        row_layout.addWidget(QLabel(status))
        button = QPushButton("Launch")
        button.setEnabled(plugin.enabled)
        button.clicked.connect(lambda _checked=False, item=plugin: launch_plugin(item))
        row_layout.addWidget(button)
        layout.addWidget(row)
    layout.addStretch(1)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(central)
    window.setCentralWidget(scroll)
    window.resize(920, 540)
    window.show()
    return app.exec()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kraken-hub")
    parser.add_argument("--catalog", help="Path to plugins.json catalog.")
    parser.add_argument("--list", action="store_true", help="Print plugin catalog and exit.")
    args = parser.parse_args(argv)
    catalog = discover_catalog(args.catalog)
    plugins = load_plugin_catalog(catalog)
    if not plugins and catalog != str(bundled_catalog_path()):
        plugins = load_plugin_catalog(bundled_catalog_path())
    if args.list:
        print_catalog(plugins)
        return
    raise SystemExit(run_gui(plugins))
