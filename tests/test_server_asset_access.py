from __future__ import annotations

import io
import hashlib
from types import SimpleNamespace

import pytest

from PIL import Image
from PyQt6.QtGui import QImage

from kraken_core.frame_matrix import MatrixAssetRef
from kraken_hub.presentation.qt.matrix_source import KrakenMatrixAssetSource
from kraken_hub.infrastructure.remote.client import RemoteAuth, RemoteServerProjectService
from kraken_manager.domain.identity import Principal


def test_remote_preview_never_opens_server_path(tmp_path, monkeypatch):
    content = io.BytesIO()
    Image.new("RGB", (20, 16), "red").save(content, "PNG")
    reads = []

    class Service:
        def is_remote_project(self, project_id):
            return project_id == "server-project"

        def read_project_blob(self, project_id, version_id):
            reads.append((project_id, version_id))
            return content.getvalue()

    def forbidden_file(*_args, **_kwargs):
        raise AssertionError("Hub attempted to open a server-side path")

    monkeypatch.setattr("kraken_hub.presentation.qt.matrix_source.open_regular_read", forbidden_file)
    reference = MatrixAssetRef(
        source_key="image-version",
        source_revision="1",
        media_type="image/png",
        metadata={"external_path": str(tmp_path / "server-only.png")},
    )
    result = KrakenMatrixAssetSource(Service(), project_id="server-project").load_asset(
        reference,
        width=12,
        height=8,
    )
    assert not QImage.fromData(result).isNull()
    assert reads == [("server-project", "image-version")]


def test_local_preview_preserves_file_access(tmp_path):
    path = tmp_path / "local.png"
    Image.new("RGB", (20, 16), "blue").save(path)

    class Service:
        def is_remote_project(self, _project_id):
            return False

        def read_project_blob(self, *_args):
            raise AssertionError("Local preview should use its local source")

    reference = MatrixAssetRef(
        source_key="local-image",
        source_revision="1",
        media_type="image/png",
        metadata={"external_path": str(path)},
    )
    result = KrakenMatrixAssetSource(Service(), project_id="local-project").load_asset(
        reference,
        width=12,
        height=8,
    )
    assert not QImage.fromData(result).isNull()


@pytest.mark.parametrize("chunks", [(b"server", b" image"), (b"damaged",), (b"server image extra",), ()])
def test_server_preview_stream_checks_integrity_without_disk_cache(tmp_path, monkeypatch, chunks):
    content = b"server image"
    reference = SimpleNamespace(
        id="version", blob=SimpleNamespace(sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content))
    )
    calls = []

    class Http:
        def iter_bytes(self, path, *, token):
            calls.append((path, token))
            yield from chunks

    service = RemoteServerProjectService(
        "http://localhost",
        auth=RemoteAuth("session", Principal.local(subject="audit", display_name="Audit")),
        data_dir=tmp_path,
        http=Http(),
    )
    monkeypatch.setattr(service, "artifact_version", lambda *_args: reference)
    monkeypatch.setattr(service, "managed_artifact_path", lambda *_args: pytest.fail("Persistent download used"))
    if b"".join(chunks) == content:
        assert service.read_project_blob("project", "version") == content
    else:
        with pytest.raises(ValueError):
            service.read_project_blob("project", "version")
    assert calls == [("/api/v1/projects/project/artifacts/versions/version", "session")]
    assert not (tmp_path / "remote-cache").exists()
