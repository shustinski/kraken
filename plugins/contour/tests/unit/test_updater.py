from __future__ import annotations

import json
import sys
from pathlib import Path

from updater.client import fetch_update_info

from contour.updater import CONTOUR_UPDATE_APP_ID, contour_update_client_config_path, load_contour_update_client_config


def test_contour_update_manifest_points_at_program_store_folder() -> None:
    payload = json.loads(contour_update_client_config_path().read_text(encoding="utf-8"))
    manifest = Path(str(payload["manifest_url"]))
    assert manifest.as_posix() == "W:/ProgramStore/Contour"


def test_contour_update_app_id_matches_neuralimage_style() -> None:
    assert CONTOUR_UPDATE_APP_ID == "Contour"
    config = load_contour_update_client_config()
    assert config.get_manifest_url().replace("\\", "/") == "W:/ProgramStore/Contour"


def test_contour_update_config_uses_pyinstaller_onedir_resources(tmp_path: Path, monkeypatch) -> None:
    executable_dir = tmp_path / "Contour"
    resources_dir = executable_dir / "_internal" / "resources"
    resources_dir.mkdir(parents=True)
    config_path = resources_dir / "update_client.json"
    config_path.write_text('{"manifest_url": "W:/ProgramStore/Contour"}', encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable_dir / "Contour.exe"))

    assert contour_update_client_config_path() == config_path
    assert load_contour_update_client_config().get_manifest_url() == "W:/ProgramStore/Contour"


def test_contour_pyinstaller_spec_bundles_update_client_config() -> None:
    spec_path = Path(__file__).resolve().parents[2] / "packaging" / "Contour.spec"
    spec_source = spec_path.read_text(encoding="utf-8")

    assert '(str(PROJECT_ROOT / "resources" / "update_client.json"), "resources")' in spec_source


def test_fetch_update_info_reads_version_json_from_directory(tmp_path: Path) -> None:
    installer = tmp_path / "Contour-1.2.0.exe"
    installer.write_bytes(b"installer")
    (tmp_path / "version.json").write_text(
        json.dumps(
            {
                "version": "1.2.0",
                "channel": "stable",
                "download_url": "Contour-1.2.0.exe",
                "releases": [
                    {
                        "version": "1.2.0",
                        "download_url": "Contour-1.2.0.exe",
                        "notes": "Folder release.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    update_info = fetch_update_info(str(tmp_path), expected_channel="stable")

    assert update_info is not None
    assert update_info.version == "1.2.0"
    assert Path(update_info.download_url) == installer.resolve()
    assert Path(update_info.releases[0].download_url) == installer.resolve()


def test_fetch_update_info_scans_installers_when_manifest_missing(tmp_path: Path) -> None:
    older = tmp_path / "Contour-1.0.0.exe"
    newer = tmp_path / "Contour-1.1.0.exe"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")

    update_info = fetch_update_info(str(tmp_path))

    assert update_info is not None
    assert update_info.version == "1.1.0"
    assert Path(update_info.download_url) == newer.resolve()
    assert [release.version for release in update_info.releases] == ["1.1.0", "1.0.0"]
