import hashlib
import io

import pytest

from kraken_core.asset_source import FileAssetSource, ServerAssetSource
from kraken_core.plugin_protocol import PluginFrameInput


def input_item(content):
    return PluginFrameInput(
        "frame", 1, 1, "version", hashlib.sha256(content).hexdigest(), "image/png", "inputs/1_1.png"
    )


def test_file_source_verifies_and_keeps_file_mode(tmp_path):
    content = b"local image"
    item = input_item(content)
    target = tmp_path / item.relative_path
    target.parent.mkdir()
    target.write_bytes(content)
    source = FileAssetSource(tmp_path, {item.relative_path: item})
    with source.open(item.relative_path) as stream:
        assert stream.read() == content
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        with source.open(item.relative_path):
            pytest.fail("Unverified stream exposed")


@pytest.mark.parametrize(
    "actual,limit,valid", [(b"server image", 100, True), (b"damaged", 100, False), (b"server image", 4, False)]
)
def test_server_source_verifies_seekable_memory_stream(monkeypatch, actual, limit, valid):
    item = input_item(b"server image")
    responses = []

    def open_response(request, *, timeout):
        assert request.full_url == "http://localhost/api/v1/agent/jobs/job/inputs/version"
        assert request.get_header("Authorization") == "Bearer secret"
        response = io.BytesIO(actual)
        responses.append(response)
        return response

    monkeypatch.setattr("urllib.request.urlopen", open_response)
    source = ServerAssetSource(
        "http://localhost", "job", {item.relative_path: item}, "secret", maximum_input_bytes=limit
    )
    assert "secret" not in repr(source)
    if valid:
        with source.open(item.relative_path) as stream:
            assert stream.read() == actual
            stream.seek(0)
            assert stream.read(6) == b"server"
        assert stream.closed
    else:
        with pytest.raises(ValueError):
            with source.open(item.relative_path):
                pytest.fail("Unverified input exposed")
    assert responses[0].closed
    with pytest.raises(KeyError):
        with source.open("../unknown.png"):
            pytest.fail("Unknown input opened")


def test_server_source_rejects_non_http_url():
    with pytest.raises(ValueError, match="HTTP"):
        ServerAssetSource("file:///project", "job", {}, "secret")
