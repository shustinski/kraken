import pytest

from kraken_core.updater import UpdateService
from updater.client import download_update_installer, load_update_client_config, parse_update_payload


def test_contour_and_neuralimage_use_kraken_server(monkeypatch):
    monkeypatch.setenv("KRAKEN_SERVER_URL", "https://kraken.example.test/base/")
    monkeypatch.delenv("CONTOUR_UPDATE_URL", raising=False)
    monkeypatch.delenv("NEURALIMAGE_UPDATE_URL", raising=False)
    for app, path in (("Contour", "contour"), ("NeuralImage", "neuralimage")):
        config = load_update_client_config(app_id=app, config_path="missing.json", env_prefix=app)
        assert (
            config.get_manifest_url("stable")
            == f"https://kraken.example.test/base/api/v1/updates/{path}/stable/version.json"
        )
        assert (
            config.get_manifest_url("beta")
            == f"https://kraken.example.test/base/api/v1/updates/{path}/beta/version.json"
        )


def test_explicit_update_url_still_wins(monkeypatch):
    monkeypatch.setenv("KRAKEN_SERVER_URL", "https://kraken.example.test")
    monkeypatch.setenv("CONTOUR_UPDATE_URL", "https://other.example.test/version.json")
    config = load_update_client_config(app_id="Contour", config_path="missing.json")
    assert config.get_manifest_url("stable") == "https://other.example.test/version.json"


def test_plugin_installer_reference_resolves_against_server_manifest():
    update = parse_update_payload(
        {"version": "2.0.0", "download_url": "setup.exe"},
        source="https://kraken.example.test/api/v1/updates/contour/stable/version.json",
    )
    assert update is not None
    assert update.releases[0].download_url == "https://kraken.example.test/api/v1/updates/contour/stable/setup.exe"


def test_plugin_download_verifies_published_digest(tmp_path, monkeypatch):
    installer = tmp_path / "setup.exe"
    installer.write_bytes(b"installer")
    update = parse_update_payload({"version": "2.0.0", "download_url": str(installer), "sha256": "0" * 64})
    monkeypatch.setattr("updater.client.get_update_staging_dir", lambda _app: tmp_path / "stage")
    with pytest.raises(ValueError, match="SHA-256"):
        download_update_installer(update.releases[0], app_id="Contour")


def test_hub_download_verifies_published_digest(tmp_path, monkeypatch):
    installer = tmp_path / "setup.exe"
    installer.write_bytes(b"installer")
    manifest = tmp_path / "version.json"
    manifest.write_text('{"version":"2.0.0","download_url":"setup.exe","sha256":"' + "0" * 64 + '"}')
    service = UpdateService(str(manifest), current_version="1.0.0", app_id="kraken-hub")
    monkeypatch.setattr("kraken_core.updater.tempfile.gettempdir", lambda: str(tmp_path))
    with pytest.raises(ValueError, match="SHA-256"):
        service.download(service.check())
