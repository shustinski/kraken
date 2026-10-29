"""Actual Python/Rust kernel gate interoperability over signed HTTP streams."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import threading
import time

import httpx
import pytest

from kraken_admin.server_backup import ServerBackupError, _gateway_fence
from kraken_manager.infrastructure.blob import FilesystemBlobStore
from kraken_server.blob_gateway import BlobTicketSigner
from tests.test_blob_gateway_junctions import gateway, SECRET, PROJECT


def test_python_backup_gate_pauses_rust_upload_and_download(tmp_path):
    root = tmp_path / "blobs"
    payload = b"own cross-language backup bytes"
    blob = FilesystemBlobStore(root).put(payload).blob
    with gateway(root) as (base, ready):
        assert ready
        with httpx.Client(base_url=base, trust_env=False, timeout=5) as client:
            assert client.get("/health").json()["storage_backup_barrier"] == 1
            signer = BlobTicketSigner(SECRET)
            upload = signer.issue("upload", blob.sha256, len(payload), context={"project": PROJECT}).token
            download = signer.issue("download", blob.sha256, len(payload), context={"project": PROJECT}).token
            with _gateway_fence(root):
                response = client.put("/v1/blobs/" + blob.sha256, content=payload, headers={"Authorization": "Bearer " + upload})
                assert response.status_code == 503
                response = client.get("/v1/blobs/" + blob.sha256, headers={"Authorization": "Bearer " + download})
                assert response.status_code == 503
                assert not list((root / ".transfers").glob("*/*"))
            response = client.put("/v1/blobs/" + blob.sha256, content=payload, headers={"Authorization": "Bearer " + upload})
            assert response.status_code == 200
            response = client.get("/v1/blobs/" + blob.sha256, headers={"Authorization": "Bearer " + download})
            assert response.status_code == 200 and response.content == payload


def test_active_rust_stream_prevents_backup_and_survives_second_gateway_start(tmp_path):
    root = tmp_path / "blobs"
    payload = b"own active upload" * 65536
    digest = hashlib.sha256(payload).hexdigest()
    release = threading.Event()
    with gateway(root) as (base, ready):
        assert ready
        signer = BlobTicketSigner(SECRET)
        token = signer.issue("upload", digest, len(payload), context={"project": PROJECT}).token

        def upload():
            def body():
                yield payload[:4096]
                assert release.wait(10), "Own upload fixture was not released"
                yield payload[4096:]
            return httpx.put(base + "/v1/blobs/" + digest, content=body(), trust_env=False, timeout=15,
                headers={"Authorization": "Bearer " + token, "Content-Length": str(len(payload))})

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(upload)
            try:
                deadline = time.monotonic() + 5
                while not list((root / ".transfers" / PROJECT).glob("*")) and time.monotonic() < deadline:
                    time.sleep(.01)
                assert list((root / ".transfers" / PROJECT).glob("*"))
                with pytest.raises(ServerBackupError, match="busy"):
                    with _gateway_fence(root):
                        pytest.fail("Python acquired exclusive gate while Rust held its shared gate")
                staged = list((root / ".gateway-staging").glob("*.upload"))
                assert len(staged) == 1
                os.utime(staged[0], (1, 1))
                # A second process must not run cleanup against an active peer.
                with gateway(root) as (_second_base, second_ready):
                    assert second_ready and Path(staged[0]).is_file()
            finally:
                release.set()
            response = future.result(timeout=10)
            assert response.status_code == 200
    assert FilesystemBlobStore(root).read(digest) == payload
    with _gateway_fence(root):
        pass
