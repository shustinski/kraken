"""Signed HTTP attacks against the real Rust binary using native junctions."""

from contextlib import contextmanager
import hashlib
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request

import pytest

from kraken_server.blob_gateway import BlobTicketSigner
from test_blob_gateway import _built_gateway
from test_security_native_junctions import directory_link


SECRET = "junction-test-secret-with-at-least-thirty-two-bytes"
PROJECT = "11111111-1111-1111-1111-111111111111"
CONTENT = b"external-private-content"
DIGEST = hashlib.sha256(CONTENT).hexdigest()


@contextmanager
def gateway(root):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    environment = os.environ.copy()
    environment.update(KRAKEN_BLOB_ROOT=str(root), KRAKEN_BLOB_GATEWAY_BIND=f"127.0.0.1:{port}",
                       KRAKEN_BLOB_GATEWAY_SECRET=SECRET)
    process = subprocess.Popen([str(_built_gateway())], env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    base = f"http://127.0.0.1:{port}"
    try:
        ready = False
        for _ in range(100):
            if process.poll() is not None:
                break
            try:
                with urllib.request.urlopen(base + "/health", timeout=0.1):
                    ready = True
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(0.02)
        yield base, ready
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        process.communicate()


def request(base, operation):
    token = BlobTicketSigner(SECRET).issue(operation, DIGEST, len(CONTENT),
                                         context={"project": PROJECT}).token
    req = urllib.request.Request(base + "/v1/blobs/" + DIGEST,
        data=CONTENT if operation == "upload" else None,
        method="PUT" if operation == "upload" else "GET",
        headers={"Authorization": "Bearer " + token})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


@pytest.mark.parametrize("operation", ["download", "upload"])
@pytest.mark.parametrize("prefix_depth", [1, 2])
def test_signed_request_rejects_blob_prefix_junction(tmp_path, operation, prefix_depth):
    root = tmp_path / "objects"
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = outside / DIGEST[2:4] if prefix_depth == 1 else outside
    destination.mkdir(exist_ok=True)
    sentinel = destination / DIGEST
    sentinel.write_bytes(CONTENT)
    with gateway(root) as (base, ready):
        assert ready
        parent = root if prefix_depth == 1 else root / DIGEST[:2]
        parent.mkdir(exist_ok=True)
        with directory_link(outside, parent / (DIGEST[:2] if prefix_depth == 1 else DIGEST[2:4])):
            status, _body = request(base, operation)
            assert status >= 400, "Gateway followed the blob prefix junction"
    assert sentinel.read_bytes() == CONTENT


def test_signed_upload_never_commits_through_prefix_junction(tmp_path):
    root = tmp_path / "objects"
    outside = tmp_path / "outside"
    outside.mkdir()
    with gateway(root) as (base, ready):
        assert ready
        with directory_link(outside, root / DIGEST[:2]):
            status, _body = request(base, "upload")
            assert status >= 400, "Upload committed outside managed objects"
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("component", [".gateway-staging", ".transfers", ".deleting"])
def test_runtime_control_directory_junction_is_rejected(tmp_path, component):
    root = tmp_path / "objects"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "preserve"
    sentinel.write_bytes(b"preserved")
    with gateway(root) as (base, ready):
        assert ready
        link = root / component
        if link.exists():
            link.rmdir()
        with directory_link(outside, link):
            status, _body = request(base, "upload")
            assert status >= 400, "Gateway followed a control directory junction"
    assert sentinel.read_bytes() == b"preserved"
    assert sorted(path.name for path in outside.iterdir()) == ["preserve"]


def test_nested_transfer_project_junction_is_rejected(tmp_path):
    root = tmp_path / "objects"
    outside = tmp_path / "outside"
    outside.mkdir()
    with gateway(root) as (base, ready):
        assert ready
        (root / ".transfers").mkdir()
        with directory_link(outside, root / ".transfers" / PROJECT):
            status, _body = request(base, "upload")
            assert status >= 400, "Gateway created a transfer lease outside its root"
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("component", ["root", ".gateway-staging"])
def test_gateway_refuses_startup_junction_without_cleaning_external_files(tmp_path, component):
    root = tmp_path / "objects"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "old.upload"
    sentinel.write_bytes(b"private-preserved")
    os.utime(sentinel, (time.time() - 172800, time.time() - 172800))
    if component == "root":
        root.rmdir()
        link = root
    else:
        link = root / component
    with directory_link(outside, link):
        with gateway(root) as (_base, ready):
            assert sentinel.exists(), "Startup cleanup deleted an external old file"
            assert not ready, "Gateway started with a reparse root"
    assert sentinel.exists(), "Startup cleanup deleted an external old file"
    assert sentinel.read_bytes() == b"private-preserved"


def test_gateway_rejects_root_replaced_with_junction_after_startup(tmp_path):
    root = tmp_path / "objects"
    outside = tmp_path / "outside"
    target = outside / DIGEST[:2] / DIGEST[2:4] / DIGEST
    target.parent.mkdir(parents=True)
    target.write_bytes(CONTENT)
    with gateway(root) as (base, ready):
        assert ready
        root.rename(tmp_path / "original-objects")
        with directory_link(outside, root):
            status, _body = request(base, "download")
            assert status >= 400, "Gateway followed its substituted root"
    assert target.read_bytes() == CONTENT
    assert not (outside / ".transfers").exists()


def test_gateway_rejects_junction_ancestor_before_creating_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "alias"
    with directory_link(outside, link):
        with gateway(link / "objects") as (_base, ready):
            assert not ready, "Gateway followed a root ancestor junction"
    assert list(outside.iterdir()) == []
