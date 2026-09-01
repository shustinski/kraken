from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kraken_server.app import create_app
from kraken_server.blob_gateway import BlobTicketSigner, validate_public_gateway_url
from kraken_server.services import InMemoryServerServices


class _Gateway:
    def __init__(self) -> None:
        self.signer = BlobTicketSigner("test-secret-which-is-at-least-thirty-two-bytes")
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def ticket(self, operation, digest, size_bytes, *, context=None):
        issued = self.signer.issue(operation, digest, size_bytes, context=context)
        return {
            "mode": "gateway",
            "url": f"http://127.0.0.1:8081/v1/blobs/{digest}",
            "token": issued.token,
            "expires_at": issued.expires_at,
            "sha256": digest,
            "size_bytes": size_bytes,
        }


class _Services(InMemoryServerServices):
    def __init__(self, digest: str) -> None:
        super().__init__()
        self.digest = digest
        self.prepared = None
        self.registered = None

    def prepare_managed_artifact_upload(self, project_id, series_id, payload, context):
        self.prepared = (project_id, series_id, dict(payload), context)
        return None

    def register_managed_artifact_upload(self, project_id, series_id, payload, context):
        self.registered = (project_id, series_id, dict(payload), context)
        return {"artifact_version_id": "version-1", "sha256": self.digest}

    def get_artifact_version(self, project_id, version_id):
        return {
            "project_id": project_id,
            "artifact_version_id": version_id,
            "blob": {"sha256": self.digest, "size_bytes": 12},
        }


def test_ticket_round_trip_rejects_tampering(monkeypatch) -> None:
    monkeypatch.setattr("kraken_server.blob_gateway._unix_time", lambda: 100)
    signer = BlobTicketSigner("test-secret-which-is-at-least-thirty-two-bytes", lifetime_seconds=60)
    ticket = signer.issue("upload", "a" * 64, 123, context={"project": "one"})

    assert ticket.expires_at == 160
    assert signer.verify(ticket.token, operation="upload") == {
        "v": 1,
        "op": "upload",
        "digest": "a" * 64,
        "size": 123,
        "exp": 160,
        "ctx": {"project": "one"},
    }
    with pytest.raises(ValueError, match="Invalid"):
        signer.verify(ticket.token[:-1] + "A", operation="upload")


def test_remote_gateway_requires_https() -> None:
    assert validate_public_gateway_url("http://127.0.0.1:8081/") == "http://127.0.0.1:8081"
    with pytest.raises(ValueError, match="HTTPS"):
        validate_public_gateway_url("http://blob.example.test:8081")


def test_server_issues_and_commits_bound_upload_ticket() -> None:
    content = b"hello kraken"
    digest = hashlib.sha256(content).hexdigest()
    services = _Services(digest)
    gateway = _Gateway()
    app = create_app(services=services, development=True, blob_gateway=gateway)
    headers = {"Authorization": "Bearer developer", "Idempotency-Key": "upload-1", "If-Match": "0"}
    payload = {
        "filename": "frame.bin",
        "media_type": "application/octet-stream",
        "sha256": digest,
        "size_bytes": len(content),
    }

    with TestClient(app) as client:
        initiated = client.post(
            "/api/v1/projects/project-1/artifacts/series-1/uploads",
            headers=headers,
            json=payload,
        )
        assert initiated.status_code == 200
        transfer = initiated.json()
        assert transfer["mode"] == "gateway"
        completed = client.post(
            "/api/v1/projects/project-1/artifacts/series-1/uploads/complete",
            headers=headers,
            json={**payload, "upload_token": transfer["token"]},
        )
        assert completed.status_code == 201
        assert completed.json()["artifact_version_id"] == "version-1"

        changed = client.post(
            "/api/v1/projects/project-1/artifacts/series-1/uploads/complete",
            headers=headers,
            json={**payload, "filename": "other.bin", "upload_token": transfer["token"]},
        )
        assert changed.status_code == 401

        download = client.get(
            "/api/v1/projects/project-1/artifacts/versions/version-1/download-ticket",
            headers={"Authorization": "Bearer developer"},
        )
        assert download.status_code == 200
        assert gateway.signer.verify(download.json()["token"], operation="download")["digest"] == digest

    assert services.prepared is not None
    assert services.registered is not None
    assert gateway.started is False


def test_configuration_protects_gateway_secret(tmp_path: Path) -> None:
    from kraken_server.configuration import ServerConfig, write_config

    executable = tmp_path / "KrakenBlobGateway.exe"
    executable.touch()
    config = write_config(
        tmp_path / "server.toml",
        database_url="postgresql+psycopg://kraken:secret@localhost/kraken",
        blob_root=tmp_path / "blobs",
        blob_gateway_public_url="http://127.0.0.1:8081",
        blob_gateway_executable=executable,
    )
    loaded = ServerConfig.load(config.path)

    assert loaded.blob_gateway_secret_value is not None
    assert len(loaded.blob_gateway_secret_value) >= 32
    assert "test-secret" not in config.path.read_text(encoding="utf-8")
    assert loaded.blob_gateway_executable == executable.resolve()


def test_configuration_requires_tls_files_for_https_gateway(tmp_path: Path) -> None:
    from kraken_server.configuration import write_config

    executable = tmp_path / "KrakenBlobGateway.exe"
    executable.touch()

    with pytest.raises(ValueError, match="requires both TLS certificate"):
        write_config(
            tmp_path / "server.toml",
            database_url="postgresql+psycopg://kraken:secret@localhost/kraken",
            blob_root=tmp_path / "blobs",
            blob_gateway_public_url="https://files.example.test:8081",
            blob_gateway_executable=executable,
        )


def test_rust_gateway_streams_python_signed_upload_and_range_download(tmp_path: Path) -> None:
    executable = Path(__file__).parents[1] / "blob_gateway" / "target" / "debug" / "kraken-blob-gateway.exe"
    if not executable.is_file():
        pytest.skip("Rust Blob Gateway debug executable is not built")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    secret = "integration-secret-which-is-at-least-thirty-two-bytes"
    environment = os.environ.copy()
    environment.update(
        {
            "KRAKEN_BLOB_ROOT": str(tmp_path / "blobs"),
            "KRAKEN_BLOB_GATEWAY_BIND": f"127.0.0.1:{port}",
            "KRAKEN_BLOB_GATEWAY_SECRET": secret,
        }
    )
    process = subprocess.Popen(  # noqa: S603 - test launches the repository-built binary
        [str(executable)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=0.2) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                time.sleep(0.05)
        else:
            pytest.fail("Rust Blob Gateway did not become ready")

        content = (b"Kraken direct data plane\n" * 1024) + b"end"
        digest = hashlib.sha256(content).hexdigest()
        signer = BlobTicketSigner(secret)
        upload = signer.issue("upload", digest, len(content)).token
        request = urllib.request.Request(
            f"{base_url}/v1/blobs/{digest}",
            data=content,
            method="PUT",
            headers={"Authorization": f"Bearer {upload}", "Content-Length": str(len(content))},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.loads(response.read())
        assert result == {"sha256": digest, "size_bytes": len(content), "already_existed": False}
        assert (tmp_path / "blobs" / digest[:2] / digest[2:4] / digest).read_bytes() == content

        download = signer.issue("download", digest, len(content)).token
        request = urllib.request.Request(
            f"{base_url}/v1/blobs/{digest}",
            headers={"Authorization": f"Bearer {download}", "Range": "bytes=7-19"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == f"bytes 7-19/{len(content)}"
            assert response.read() == content[7:20]
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_blob_benchmark_runs_parallel_streams(tmp_path: Path) -> None:
    executable = Path(__file__).parents[1] / "blob_gateway" / "target" / "debug" / "kraken-blob-gateway.exe"
    if not executable.is_file():
        pytest.skip("Rust Blob Gateway debug executable is not built")
    from kraken_server.blob_benchmark import run_blob_benchmark
    from kraken_server.configuration import write_config

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    config = write_config(
        tmp_path / "server.toml",
        database_url="postgresql+psycopg://unused:unused@localhost/unused",
        blob_root=tmp_path / "blobs",
        blob_gateway_public_url=f"http://127.0.0.1:{port}",
        blob_gateway_bind=f"127.0.0.1:{port}",
        blob_gateway_executable=executable,
    )

    result = run_blob_benchmark(config.path, clients=4, size_mib=2)

    assert result.clients == 4
    assert result.total_bytes == 8 * 1024 * 1024
    assert result.gigabits_per_second > 0
