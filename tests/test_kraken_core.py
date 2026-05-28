from __future__ import annotations

import json

import kraken_hub.app as hub_app
from kraken_core.ipc import ActionRegistry, ActionRequest
from kraken_core.plugins import PluginMetadata, load_plugin_catalog, scan_plugin_directory
from kraken_core.qt import resolve_icon_path
from kraken_core.styles import plugin_icon_path, shared_icon_path
from kraken_core.updater import compare_versions, parse_update_payload, select_platform_release


def test_plugin_catalog_loads(tmp_path):
    catalog = tmp_path / "plugins.json"
    catalog.write_text(
        json.dumps(
            {
                "plugins": [
                    {
                        "id": "contour",
                        "display_name": "Contour",
                        "executables": {"linux": {"command": ["python", "-m", "contour"]}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    plugins = load_plugin_catalog(catalog)

    assert plugins[0].id == "contour"
    assert plugins[0].executable_for("linux").command == ("python", "-m", "contour")


def test_update_payload_selects_platform_release():
    update = parse_update_payload(
        {
            "version": "2.0.0",
            "releases": [
                {"version": "2.0.0", "platform": "windows", "download_url": "setup.exe"},
                {"version": "2.0.0", "platform": "linux", "download_url": "app.tar.gz"},
            ],
        }
    )

    assert update is not None
    assert compare_versions("2.0.0", "1.9.0") == 1
    assert select_platform_release(update, "linux").download_url == "app.tar.gz"


def test_action_registry_reports_invalid_action():
    registry = ActionRegistry()

    response = registry.dispatch(ActionRequest("missing", {}))

    assert response.ok is False
    assert "Unsupported action" in response.message


def test_plugin_icons_live_in_plugin_resources():
    contour_icon = plugin_icon_path("contour", suffix=".png")
    krona_icon = plugin_icon_path("krona", suffix=".png")

    assert contour_icon.exists()
    assert krona_icon.exists()
    assert "plugins" in contour_icon.parts
    assert "plugins" in krona_icon.parts
    assert not shared_icon_path("contour", suffix=".png").exists()
    assert not shared_icon_path("krona", suffix=".png").exists()
    assert resolve_icon_path("contour") == plugin_icon_path("contour", suffix=".ico")


def test_hub_prefers_root_plugin_launcher(tmp_path, monkeypatch):
    plugin_root = tmp_path / "plugins" / "contour"
    plugin_root.mkdir(parents=True)
    (plugin_root / "__main__.py").write_text("print('contour')\n", encoding="utf-8")
    monkeypatch.setattr(hub_app.shutil, "which", lambda name: "uv" if name == "uv" else None)

    command = hub_app.build_launch_command(PluginMetadata(id="contour", display_name="Contour"), root=tmp_path)

    assert command == ["uv", "run", "python", "__main__.py"]


def test_plugin_directory_scan_reads_manifest_and_changelog(tmp_path):
    plugin_root = tmp_path / "plugins" / "sample"
    resources = plugin_root / "resources"
    resources.mkdir(parents=True)
    (resources / "plugin.json").write_text(
        json.dumps(
            {
                "id": "sample",
                "display_name": "Sample",
                "description": "Sample plugin.",
                "version": "1.2.3",
            }
        ),
        encoding="utf-8",
    )
    (resources / "changelog.md").write_text(
        "# Changelog\n\n## 1.2.3\n\n- Added inventory support.\n\n## 1.0.0\n\n- Initial release.\n",
        encoding="utf-8",
    )

    plugins = scan_plugin_directory(tmp_path / "plugins")

    assert plugins[0].id == "sample"
    assert plugins[0].description == "Sample plugin."
    assert plugins[0].source_dir == str(plugin_root.resolve())
    assert plugins[0].version_history[0].version == "1.2.3"
    assert "inventory" in plugins[0].version_history[0].notes


def test_plugin_directory_scan_falls_back_to_pyproject(tmp_path):
    plugin_root = tmp_path / "plugins" / "sample"
    plugin_root.mkdir(parents=True)
    (plugin_root / "pyproject.toml").write_text(
        '[project]\nname = "sample"\nversion = "0.2.0"\ndescription = "From pyproject."\n',
        encoding="utf-8",
    )

    plugins = scan_plugin_directory(tmp_path / "plugins")

    assert plugins[0].id == "sample"
    assert plugins[0].version == "0.2.0"
    assert plugins[0].description == "From pyproject."


def test_hub_builds_editable_install_command(tmp_path):
    plugin_root = tmp_path / "plugins" / "sample"
    plugin_root.mkdir(parents=True)
    plugin = PluginMetadata(id="sample", display_name="Sample", source_dir=str(plugin_root))

    command = hub_app.build_install_command(plugin)

    assert command[:3] == [hub_app.sys.executable, "-m", "pip"]
    assert command[-2:] == ["-e", str(plugin_root)]
