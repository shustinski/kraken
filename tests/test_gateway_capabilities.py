"""Reject an older data plane even when its health endpoint responds."""

import io
import json

import pytest

from kraken_server.blob_gateway import BlobGatewayManager


@pytest.mark.parametrize("missing", ["storage_backup_barrier", "filesystem_path_safety"])
def test_gateway_readiness_rejects_missing_safety_capability(tmp_path, monkeypatch, missing):
    health = {
        "service": "kraken-blob-gateway",
        "project_deletion_barrier": 1,
        "storage_backup_barrier": 1,
        "filesystem_path_safety": 1,
    }
    health.pop(missing)
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_kw: io.BytesIO(json.dumps(health).encode()))
    manager = BlobGatewayManager(
        public_url="http://127.0.0.1:9",
        bind="127.0.0.1:9",
        blob_root=tmp_path,
        executable=tmp_path / "gateway.exe",
        secret="a-test-secret-at-least-thirty-two-bytes",
    )
    with pytest.raises(RuntimeError, match="Blob Gateway"):
        manager._ready()
