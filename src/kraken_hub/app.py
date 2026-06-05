from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib import parse as urlparse

from kraken_core.plugins import (
    PluginInventoryItem,
    PluginMetadata,
    build_plugin_inventory,
    load_plugin_catalog,
    merge_plugin_sources,
    scan_plugin_directory,
)
from kraken_core.qt import configure_application_identity
from kraken_core.runtime import current_platform, workspace_root
from kraken_core.styles import load_shared_stylesheet


def bundled_catalog_path() -> Path:
    return Path(__file__).resolve().parent / "resources" / "plugins.json"


def default_plugins_dir() -> Path:
    return Path(os.getenv("KRAKEN_PLUGINS_DIR", "")).expanduser().resolve() if os.getenv("KRAKEN_PLUGINS_DIR") else workspace_root() / "plugins"


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


def build_install_command(plugin: PluginMetadata) -> list[str]:
    if not plugin.source_dir:
        return []
    plugin_root = Path(plugin.source_dir)
    if not plugin_root.is_dir():
        return []
    return [sys.executable, "-m", "pip", "install", "-e", str(plugin_root)]


def install_plugin(plugin: PluginMetadata) -> subprocess.CompletedProcess[str]:
    command = build_install_command(plugin)
    if not command:
        raise ValueError(f"Plugin {plugin.display_name} has no local source directory.")
    return subprocess.run(command, cwd=str(workspace_root()), text=True, capture_output=True, check=False)


def open_plugin_folder(plugin: PluginMetadata) -> None:
    if not plugin.source_dir:
        return
    path = Path(plugin.source_dir)
    if not path.exists():
        return
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def print_catalog(plugins: list[PluginMetadata]) -> None:
    for plugin in plugins:
        state = "disabled" if not plugin.enabled else "enabled"
        print(f"{plugin.id}\t{plugin.version}\t{state}\t{plugin.display_name}")


def print_inventory(items: list[PluginInventoryItem]) -> None:
    for item in items:
        plugin = item.metadata
        enabled = "disabled" if not plugin.enabled else "enabled"
        installed = "installed" if item.installed else "not-installed"
        print(f"{plugin.id}\t{plugin.version}\t{enabled}\t{installed}\t{plugin.display_name}")


def run_gui(items: list[PluginInventoryItem]) -> int:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    app = QApplication(sys.argv)
    configure_application_identity(app, app_id="Kraken.Hub", icon_name="kraken")
    app.setStyleSheet(load_shared_stylesheet("dark_modern.qss"))
    window = QMainWindow()
    window.setWindowTitle("Kraken Hub")
    central = QWidget()
    layout = QVBoxLayout(central)
    layout.setSpacing(10)

    def refresh_after_install() -> None:
        QMessageBox.information(window, "Kraken Hub", "Plugin installation finished. Restart Kraken Hub to refresh status.")

    for item in items:
        plugin = item.metadata
        row = QFrame()
        row.setObjectName("pluginInventoryRow")
        row_layout = QVBoxLayout(row)
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel(f"{plugin.display_name} ({plugin.version})")
        title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header_layout.addWidget(title, 2)

        status_text = "Disabled" if not plugin.enabled else ("Installed" if item.installed else "Available")
        status = QLabel(status_text)
        status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(status, 1)

        open_button = QPushButton("Open")
        open_button.setEnabled(plugin.enabled and item.installed)
        open_button.clicked.connect(lambda _checked=False, selected=plugin: launch_plugin(selected))
        header_layout.addWidget(open_button)

        install_button = QPushButton("Install")
        install_button.setEnabled(plugin.enabled and not item.installed and bool(build_install_command(plugin)))

        def run_install(_checked: bool = False, selected: PluginMetadata = plugin) -> None:
            result = install_plugin(selected)
            if result.returncode == 0:
                refresh_after_install()
                return
            details = result.stderr.strip() or result.stdout.strip() or "Unknown installation error."
            QMessageBox.warning(window, "Kraken Hub", f"Failed to install {selected.display_name}.\n\n{details[:1200]}")

        install_button.clicked.connect(run_install)
        header_layout.addWidget(install_button)

        folder_button = QPushButton("Folder")
        folder_button.setEnabled(bool(plugin.source_dir))
        folder_button.clicked.connect(lambda _checked=False, selected=plugin: open_plugin_folder(selected))
        header_layout.addWidget(folder_button)
        row_layout.addWidget(header)

        description = QLabel(plugin.description or "No description provided.")
        description.setWordWrap(True)
        description.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row_layout.addWidget(description)

        history = QTextEdit()
        history.setReadOnly(True)
        history.setMinimumHeight(92)
        history.setMaximumHeight(150)
        history.setPlainText(format_version_history(plugin))
        row_layout.addWidget(history)
        layout.addWidget(row)
    layout.addStretch(1)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(central)
    window.setCentralWidget(scroll)
    window.resize(920, 540)
    window.show()
    return app.exec()


def format_version_history(plugin: PluginMetadata) -> str:
    if not plugin.version_history:
        return f"Current version: {plugin.version}"
    chunks: list[str] = []
    for entry in plugin.version_history:
        chunks.append(entry.version)
        if entry.notes:
            chunks.append(entry.notes)
        chunks.append("")
    return "\n".join(chunks).strip()


def load_plugins(catalog_path: str, plugins_dir: Path) -> list[PluginMetadata]:
    catalog_plugins = load_plugin_catalog(catalog_path)
    if not catalog_plugins and catalog_path != str(bundled_catalog_path()):
        catalog_plugins = load_plugin_catalog(bundled_catalog_path())
    scanned_plugins = scan_plugin_directory(plugins_dir)
    return merge_plugin_sources(scanned_plugins, catalog_plugins)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kraken-hub")
    parser.add_argument("--catalog", help="Path to plugins.json catalog.")
    parser.add_argument("--plugins-dir", help="Folder where Kraken plugins are stored.")
    parser.add_argument("--list", action="store_true", help="Print plugin catalog and exit.")
    args = parser.parse_args(argv)
    catalog = discover_catalog(args.catalog)
    plugins_dir = Path(args.plugins_dir).expanduser().resolve() if args.plugins_dir else default_plugins_dir()
    plugins = load_plugins(catalog, plugins_dir)
    inventory = build_plugin_inventory(plugins)
    if args.list:
        print_inventory(inventory)
        return
    raise SystemExit(run_gui(inventory))
