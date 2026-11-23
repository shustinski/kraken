"""Public, read-only update files served from an operator-selected directory."""

import json
import hashlib

from fastapi.testclient import TestClient

from kraken_server.app import create_app


def test_public_update_manifest_and_installer(tmp_path, monkeypatch):
    folder = tmp_path / "contour" / "stable"
    folder.mkdir(parents=True)
    (folder / "version.json").write_text(
        json.dumps(
            {"version": "2.0.0", "download_url": "setup.exe", "sha256": hashlib.sha256(b"installer").hexdigest()}
        )
    )
    (folder / "setup.exe").write_bytes(b"installer")
    monkeypatch.setenv("KRAKEN_UPDATES_ROOT", str(tmp_path))

    with TestClient(create_app(development=True)) as client:
        manifest = client.get("/api/v1/updates/contour/stable/version.json")
        installer = client.get("/api/v1/updates/contour/stable/setup.exe")

    assert manifest.status_code == 200
    assert manifest.json()["version"] == "2.0.0"
    assert installer.status_code == 200
    assert installer.content == b"installer"


def test_update_files_are_unavailable_without_configuration(monkeypatch):
    monkeypatch.delenv("KRAKEN_UPDATES_ROOT", raising=False)
    with TestClient(create_app(development=True)) as client:
        assert client.get("/api/v1/updates/contour/stable/version.json").status_code == 404


def test_manifest_accepts_release_list(tmp_path, monkeypatch):
    folder = tmp_path / "neuralimage" / "beta"
    folder.mkdir(parents=True)
    (folder / "setup.exe").write_bytes(b"installer")
    (folder / "version.json").write_text(
        json.dumps(
            {
                "version": "2.0.0",
                "releases": [
                    {
                        "version": "2.0.0",
                        "platform": "windows",
                        "download_url": "setup.exe",
                        "sha256": hashlib.sha256(b"installer").hexdigest(),
                    }
                ],
            }
        )
    )
    monkeypatch.setenv("KRAKEN_UPDATES_ROOT", str(tmp_path))
    with TestClient(create_app(development=True)) as client:
        assert client.get("/api/v1/updates/neuralimage/beta/version.json").status_code == 200


def test_update_route_rejects_unknown_product_and_symlink(tmp_path, monkeypatch):
    folder = tmp_path / "contour" / "stable"
    folder.mkdir(parents=True)
    outside = tmp_path.parent / "outside-update.txt"
    outside.write_bytes(b"private")
    (folder / "escape.exe").symlink_to(outside)
    monkeypatch.setenv("KRAKEN_UPDATES_ROOT", str(tmp_path))

    with TestClient(create_app(development=True)) as client:
        assert client.get("/api/v1/updates/other/stable/version.json").status_code == 404
        assert client.get("/api/v1/updates/contour/stable/escape.exe").status_code == 404


def test_manifest_rejects_external_installer_and_missing_hash(tmp_path, monkeypatch):
    folder = tmp_path / "contour" / "stable"
    folder.mkdir(parents=True)
    manifest_path = folder / "version.json"
    monkeypatch.setenv("KRAKEN_UPDATES_ROOT", str(tmp_path))

    with TestClient(create_app(development=True)) as client:
        manifest_path.write_text(
            json.dumps({"version": "2.0.0", "download_url": "https://example.org/setup.exe", "sha256": "a" * 64})
        )
        assert client.get("/api/v1/updates/contour/stable/version.json").status_code == 404
        manifest_path.write_text(json.dumps({"version": "2.0.0", "download_url": "setup.exe"}))
        assert client.get("/api/v1/updates/contour/stable/version.json").status_code == 404


def test_channel_publishes_plugin_catalog_as_json(tmp_path, monkeypatch):
    # Hub reads <channel>/plugins.json next to the NeuralImage update manifest.
    folder = tmp_path / "neuralimage" / "stable"
    folder.mkdir(parents=True)
    catalog = {"plugins": [{"id": "neuralimage", "version": "6.3.0"}, {"id": "contour", "version": "0.9.0"}]}
    (folder / "plugins.json").write_text(json.dumps(catalog), encoding="utf-8")
    monkeypatch.setenv("KRAKEN_UPDATES_ROOT", str(tmp_path))

    with TestClient(create_app(development=True)) as client:
        response = client.get("/api/v1/updates/neuralimage/stable/plugins.json")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == catalog


def test_malformed_plugin_catalog_is_not_published(tmp_path, monkeypatch):
    folder = tmp_path / "neuralimage" / "stable"
    folder.mkdir(parents=True)
    catalog_path = folder / "plugins.json"
    monkeypatch.setenv("KRAKEN_UPDATES_ROOT", str(tmp_path))

    with TestClient(create_app(development=True)) as client:
        for payload in ("{not json", json.dumps({"plugins": "neuralimage"}), json.dumps({"plugins": [{"version": "1"}]})):
            catalog_path.write_text(payload, encoding="utf-8")
            assert client.get("/api/v1/updates/neuralimage/stable/plugins.json").status_code == 404
