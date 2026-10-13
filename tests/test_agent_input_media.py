from __future__ import annotations

import hashlib
import io
from email.message import Message
from uuid import uuid4

import pytest
from PIL import Image

from kraken_agent.runner import PluginRegistry
from kraken_agent.server_worker import ServerAgentWorker
from kraken_core.plugin_protocol import PluginFrameInput, PluginJobManifest


@pytest.mark.parametrize(
    "suffix,ticket_type,expected",
    [
        (".jpg", "image/jpeg", "image/jpeg"),
        (".jpg", None, "image/jpeg"),
        (".jpeg", "application/octet-stream", "image/jpeg"),
        (".png", None, "image/png"),
        (".tiff", None, "image/tiff"),
        (".zip", None, "application/zip"),
        (".bin", None, "application/octet-stream"),
        (".jpg", "application/pdf", "application/pdf"),
    ],
)
def test_blob_download_preserves_artifact_type_for_contour(tmp_path, monkeypatch, suffix, ticket_type, expected):
    from contour.kraken_bridge import ContourKrakenSession

    buffer = io.BytesIO()
    Image.new("RGB", (12, 16), "red").save(buffer, format={".png": "PNG", ".tiff": "TIFF"}.get(suffix, "JPEG"))
    content = buffer.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    worker = ServerAgentWorker(
        server_url="http://localhost", token="owner", data_dir=tmp_path / "worker", registry=PluginRegistry({})
    )
    ticket = {"mode": "gateway", "url": "http://localhost/blob", "token": "transfer"}
    if ticket_type is not None:
        ticket["media_type"] = ticket_type
    worker._request = lambda *_args: ticket

    def open_blob(*_args, **_kwargs):
        response = io.BytesIO(content)
        response.headers = Message()
        response.headers["Content-Type"] = "application/octet-stream"
        return response

    monkeypatch.setattr("urllib.request.urlopen", open_blob)
    root = tmp_path / "job"
    destination = root / "inputs" / f"1_1{suffix}"
    media_type = worker._download("/input", destination, expected_sha256=digest)
    assert destination.read_bytes() == content
    assert media_type == expected
    if expected.startswith("image/"):
        identity = str(uuid4())
        manifest = PluginJobManifest(
            job_id=identity,
            operation="frames.vectorize.v1",
            project_id=identity,
            layer_id=identity,
            actor_id=identity,
            target_representation_id=identity,
            inputs=(
                PluginFrameInput(
                    frame_id=identity,
                    x=1,
                    y=1,
                    artifact_version_id=identity,
                    sha256=digest,
                    media_type=media_type,
                    relative_path=f"inputs/1_1{suffix}",
                ),
            ),
        )
        manifest_path = root / "job.json"
        manifest_path.write_text(manifest.to_json(), encoding="utf-8")
        session = ContourKrakenSession.load(
            job_manifest=manifest_path, result_manifest=root / "result.json", staging_root=root
        )
        assert session.input_paths == (destination,)


def test_generic_blob_download_still_rejects_wrong_checksum(tmp_path, monkeypatch):
    worker = ServerAgentWorker(
        server_url="http://localhost", token="owner", data_dir=tmp_path / "worker", registry=PluginRegistry({})
    )
    worker._request = lambda *_args: {"mode": "proxy"}

    def open_blob(*_args, **_kwargs):
        response = io.BytesIO(b"damaged-image")
        response.headers = Message()
        response.headers["Content-Type"] = "application/octet-stream"
        return response

    monkeypatch.setattr("urllib.request.urlopen", open_blob)
    destination = tmp_path / "1_1.jpg"
    with pytest.raises(ValueError, match="hash mismatch"):
        worker._download("/input", destination, expected_sha256="0" * 64)
    assert not destination.exists()
