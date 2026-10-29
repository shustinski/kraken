"""Remote transfer failures must never delete an existing destination."""
from io import BytesIO
import urllib.error

import pytest

from kraken_hub.infrastructure.remote.client import RemoteHttpClient, RemoteServerError


@pytest.fixture(params=["server", "gateway"])
def download(request, tmp_path):
    client = RemoteHttpClient("https://example.test")
    target = tmp_path / "user-export.cif"

    def call():
        if request.param == "server":
            return client.download("/artifact", token="test", destination=target)
        return client.download_url("https://example.test/blob", token="test", destination=target)

    return target, call


def test_download_collision_preserves_existing_file(download, monkeypatch):
    target, call = download
    target.write_bytes(b"independent existing user file")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: BytesIO(b"download"))
    with pytest.raises(FileExistsError):
        call()
    assert target.read_bytes() == b"independent existing user file"


def test_http_error_before_create_preserves_existing_file(download, monkeypatch):
    target, call = download
    target.write_bytes(b"independent existing user file")

    def denied(*_args, **_kwargs):
        raise urllib.error.HTTPError("https://example.test", 403, "Denied", {}, BytesIO(b"denied"))

    monkeypatch.setattr("urllib.request.urlopen", denied)
    with pytest.raises((urllib.error.HTTPError, RemoteServerError)):
        call()
    assert target.read_bytes() == b"independent existing user file"


def test_stream_failure_cleans_only_own_partial_and_allows_retry(download, monkeypatch):
    target, call = download
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: BytesIO(b"download"))

    def broken(_source, output, **_kwargs):
        output.write(b"partial")
        raise OSError("transfer interrupted")

    with monkeypatch.context() as scoped:
        scoped.setattr("shutil.copyfileobj", broken)
        with pytest.raises(OSError, match="interrupted"):
            call()
    assert not target.exists()
    assert call() == target
    assert target.read_bytes() == b"download"


def test_transfer_cleanup_preserves_another_writers_replacement(download, monkeypatch):
    target, call = download
    replacement = target.with_suffix(".replacement")
    replacement.write_bytes(b"other writer completed")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: BytesIO(b"download"))

    def replaced(_source, output, **_kwargs):
        output.close()
        replacement.replace(target)
        raise OSError("interrupted after replacement")

    monkeypatch.setattr("shutil.copyfileobj", replaced)
    with pytest.raises(OSError, match="replacement"):
        call()
    assert target.read_bytes() == b"other writer completed"
