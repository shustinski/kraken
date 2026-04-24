from __future__ import annotations

import json

from kraken_core.ipc import ActionRegistry, ActionRequest
from kraken_core.plugins import load_plugin_catalog
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
