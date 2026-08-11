from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import kraken_hub.app as hub_app
import kraken_core.plugins as plugin_core
from kraken_core.ipc import ActionRegistry, ActionRequest
from kraken_core.plugins import (
    PluginMetadata,
    PluginExecutable,
    build_plugin_inventory,
    load_plugin_catalog,
    scan_plugin_directory,
)
from kraken_core.qt import resolve_icon_path
from kraken_core.styles import load_shared_stylesheet, plugin_icon_path, shared_icon_path, shared_styles_root
from kraken_core.updater import (
    UpdateService,
    compare_versions,
    parse_update_payload,
    resolve_source_reference,
    select_platform_release,
)


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


def test_registered_standalone_plugin_resolves_relative_executable(tmp_path):
    plugin_root = tmp_path / "contour"
    plugin_root.mkdir()
    executable = plugin_root / "Contour.exe"
    executable.write_bytes(b"MZ")
    (plugin_root / "plugin.json").write_text(
        json.dumps(
            {
                "id": "contour",
                "display_name": "Contour",
                "executables": {"windows": {"path": "Contour.exe"}},
            }
        ),
        encoding="utf-8",
    )

    plugin = scan_plugin_directory(tmp_path)[0]

    assert plugin.executable_for("windows").path == str(executable.resolve())


def test_existing_standalone_executable_marks_plugin_installed(tmp_path, monkeypatch):
    executable = tmp_path / "Contour.exe"
    executable.write_bytes(b"MZ")
    plugin = PluginMetadata(
        id="contour",
        display_name="Contour",
        executables={"windows": PluginExecutable(path=str(executable))},
    )
    monkeypatch.setattr("kraken_core.plugins.current_platform", lambda: "windows")

    inventory = build_plugin_inventory([plugin])

    assert inventory[0].installed is True


def test_standalone_plugin_launches_from_its_install_directory(tmp_path, monkeypatch):
    executable = tmp_path / "installed" / "Contour.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"MZ")
    plugin = PluginMetadata(
        id="contour",
        display_name="Contour",
        executables={"windows": PluginExecutable(path=str(executable))},
    )
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]
        return object()

    monkeypatch.setattr("kraken_core.plugins.current_platform", lambda: "windows")
    monkeypatch.setattr(hub_app, "current_platform", lambda: "windows")
    monkeypatch.setattr(hub_app, "workspace_root", lambda: tmp_path / "workspace")
    monkeypatch.setattr(hub_app.subprocess, "Popen", fake_popen)

    hub_app.launch_plugin(plugin, arguments=("--example",))

    assert captured == {
        "command": [str(executable), "--example"],
        "cwd": str(executable.parent),
    }


def test_registered_plugin_overrides_catalog_executable(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "plugins": [
                    {
                        "id": "contour",
                        "display_name": "Contour",
                        "version": "1.0.0",
                        "capabilities": [{"operation": "frames.vectorize.v1"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    source_plugins = tmp_path / "source-plugins"
    source_plugins.mkdir()
    registered = tmp_path / "registered"
    plugin_root = registered / "contour"
    plugin_root.mkdir(parents=True)
    executable = plugin_root / "Contour.exe"
    executable.write_bytes(b"MZ")
    (plugin_root / "plugin.json").write_text(
        json.dumps(
            {
                "id": "contour",
                "display_name": "Contour",
                "executables": {"windows": {"path": "Contour.exe"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("KRAKEN_REGISTERED_PLUGINS_DIR", str(registered))
    monkeypatch.setattr(hub_app, "scan_windows_plugin_registry", lambda: [])

    plugin = hub_app.load_plugins(str(catalog), source_plugins)[0]

    assert plugin.executable_for("windows").path == str(executable.resolve())
    assert plugin.version == "1.0.0"
    assert plugin.capabilities[0].operation == "frames.vectorize.v1"


def test_windows_installer_registration_is_discovered(tmp_path, monkeypatch):
    executable = tmp_path / "Contour.exe"
    executable.write_bytes(b"MZ")

    class FakeKey:
        def __init__(self, *, children=None, values=None):
            self.children = children or {}
            self.values = values or {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    contour = FakeKey(
        values={
            "Executable": str(executable),
            "DisplayName": "Contour",
            "Version": "1.2.3",
            "ProtocolVersion": "1.0",
        }
    )
    machine = FakeKey(children={"contour": contour})
    user = FakeKey()

    def open_key(parent, name):
        if parent == "HKLM":
            return machine
        if parent == "HKCU":
            return user
        return parent.children[name]

    def enum_key(key, index):
        try:
            return list(key.children)[index]
        except IndexError as exc:
            raise OSError from exc

    fake_winreg = SimpleNamespace(
        HKEY_LOCAL_MACHINE="HKLM",
        HKEY_CURRENT_USER="HKCU",
        OpenKey=open_key,
        EnumKey=enum_key,
        QueryValueEx=lambda key, name: (key.values[name], 1),
    )
    monkeypatch.setattr(plugin_core.os, "name", "nt")
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg)

    plugins = plugin_core.scan_windows_plugin_registry()

    assert len(plugins) == 1
    assert plugins[0].id == "contour"
    assert plugins[0].version == "1.2.3"
    assert plugins[0].executable_for("windows").path == str(executable)


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


def test_update_service_dismissal_only_affects_current_session(tmp_path):
    manifest = tmp_path / "version.json"
    manifest.write_text(json.dumps({"version": "2.0.0", "download_url": "setup.exe"}), encoding="utf-8")
    service = UpdateService(str(manifest), current_version="1.0.0", app_id="test")

    update = service.check()
    assert update is not None
    service.dismiss_for_session(update)

    assert service.check() is None
    assert UpdateService(str(manifest), current_version="1.0.0", app_id="test").check() is not None


def test_relative_download_source_is_resolved_from_manifest(tmp_path):
    manifest = tmp_path / "updates" / "version.json"
    manifest.parent.mkdir()

    assert resolve_source_reference("install/setup.exe", str(manifest)) == str(
        (manifest.parent / "install" / "setup.exe").resolve()
    )
    assert resolve_source_reference("install/setup.exe", "https://updates.example/app/version.json") == (
        "https://updates.example/app/install/setup.exe"
    )


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


def test_shared_stylesheet_includes_tab_scroller_arrow_assets():
    stylesheet = load_shared_stylesheet("dark_modern.qss")

    expected_left = (shared_styles_root() / "icons" / "chevron_left_light.svg").resolve().as_posix()
    expected_right = (shared_styles_root() / "icons" / "chevron_right_light.svg").resolve().as_posix()
    assert f'url("{expected_left}")' in stylesheet
    assert f'url("{expected_right}")' in stylesheet
    assert "QTabBar QToolButton::left-arrow" in stylesheet
    assert "QTabBar QToolButton::right-arrow" in stylesheet


def test_hub_prefers_root_plugin_launcher(tmp_path, monkeypatch):
    plugin_root = tmp_path / "plugins" / "contour"
    plugin_root.mkdir(parents=True)
    (plugin_root / "__main__.py").write_text("print('contour')\n", encoding="utf-8")
    monkeypatch.setattr(hub_app.shutil, "which", lambda name: "uv" if name == "uv" else None)

    command = hub_app.build_launch_command(PluginMetadata(id="contour", display_name="Contour"), root=tmp_path)

    assert command == ["uv", "run", "python", "__main__.py"]


def test_bundled_plugins_have_standalone_launchers():
    plugins = load_plugin_catalog(hub_app.bundled_catalog_path())

    assert plugins
    for plugin in plugins:
        launcher = hub_app.workspace_root() / "plugins" / plugin.id / "__main__.py"
        assert launcher.is_file(), plugin.id
        assert plugin.executable_for("windows").command == ("python", "__main__.py")
        assert plugin.executable_for("linux").command == ("python", "__main__.py")
