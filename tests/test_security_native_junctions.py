"""Actual Windows junction attacks; no symlink privilege and no foreign cleanup."""

from contextlib import contextmanager
import hashlib
import io
import os
from email.message import EmailMessage

import pytest

from kraken_agent.jobs import StagingWorkspace
from kraken_agent.server_worker import ServerAgentWorker
from kraken_core.safe_files import (
    UnsafeFilesystemPath, contained_path, is_link_or_reparse,
    make_contained_directories, open_regular_append, open_regular_read,
)
from kraken_manager.infrastructure.blob.filesystem import FilesystemBlobStore, UnsafeBlobPath
from kraken_manager.infrastructure.review.package import ReviewPackageReader, UnsafeReviewPackage


@contextmanager
def directory_link(target, link):
    target = target.resolve(strict=True)
    link = link.absolute()
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
        assert link.is_junction()
    else:
        os.symlink(target, link, target_is_directory=True)
    assert is_link_or_reparse(link)
    try:
        yield link
    finally:
        # Remove only this test's link entry, never recurse through its target.
        if is_link_or_reparse(link):
            os.rmdir(link) if os.name == "nt" else link.unlink()


def download_worker(monkeypatch):
    worker = object.__new__(ServerAgentWorker)
    worker.server_url, worker.token, worker.headers = "http://127.0.0.1", "test-only-token", {}
    worker._request = lambda *_args, **_kwargs: {"mode": "proxy"}
    class Response(io.BytesIO):
        headers = EmailMessage()
    Response.headers["Content-Type"] = "image/png"
    monkeypatch.setattr("kraken_agent.server_worker.urllib.request.urlopen",
                        lambda *_args, **_kwargs: Response(b"downloaded"))
    return worker


def test_staging_rejects_actual_output_junction_without_touching_target(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "preserve.cif"
    sentinel.write_bytes(b"preserved")
    workspace = StagingWorkspace(tmp_path / "staging", "job")
    workspace.create()
    outputs = workspace.path / "outputs"
    outputs.rmdir()
    with directory_link(outside, outputs):
        with pytest.raises(UnsafeFilesystemPath):
            workspace.resolve("outputs/preserve.cif")
    assert sentinel.read_bytes() == b"preserved"


def test_review_reader_rejects_actual_junction_in_package(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.cif").write_bytes(b"private")
    package = tmp_path / "package"
    package.mkdir()
    with directory_link(outside, package / "vectors"):
        with pytest.raises(UnsafeReviewPackage):
            ReviewPackageReader(b"unused").list_relative_paths(str(package))
    assert (outside / "private.cif").read_bytes() == b"private"


def test_blob_reader_rejects_actual_prefix_junction(tmp_path):
    store = FilesystemBlobStore(tmp_path / "blobs")
    digest = hashlib.sha256(b"private").hexdigest()
    outside = tmp_path / "outside"
    (outside / digest[2:4]).mkdir(parents=True)
    sentinel = outside / digest[2:4] / digest
    sentinel.write_bytes(b"private")
    with directory_link(outside, store.root / digest[:2]):
        with pytest.raises(UnsafeBlobPath):
            store.read(digest)
    assert sentinel.read_bytes() == b"private"


def test_blob_enumeration_never_descends_an_actual_directory_junction(tmp_path, monkeypatch):
    store = FilesystemBlobStore(tmp_path / "blobs")
    outside = tmp_path / "outside"
    outside.mkdir()
    original_scandir = os.scandir
    def audit_scandir(path):
        from pathlib import Path
        assert not Path(path).is_junction() if os.name == "nt" else not Path(path).is_symlink(), "Blob enumeration followed a reparse link"
        return original_scandir(path)
    with directory_link(outside, store.root / "linked-directory"):
        monkeypatch.setattr(os, "scandir", audit_scandir)
        assert list(store.iter_refs()) == []


def test_failed_exclusive_agent_download_preserves_preexisting_file(tmp_path, monkeypatch):
    worker = download_worker(monkeypatch)
    destination = tmp_path / "existing.png"
    destination.write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        worker._download("/input", destination, expected_sha256=hashlib.sha256(b"downloaded").hexdigest())
    assert destination.exists(), "Cleanup removed a file that the download did not create"
    assert destination.read_bytes() == b"preserved"


def test_agent_download_removes_only_its_partial_hash_mismatch(tmp_path, monkeypatch):
    worker = download_worker(monkeypatch)
    destination = tmp_path / "nested" / "new.png"
    with pytest.raises(ValueError, match="hash mismatch"):
        worker._download("/input", destination, expected_sha256="0" * 64)
    assert not destination.exists()


def test_agent_transfers_actual_bytes_after_containment_checks(tmp_path, monkeypatch):
    worker = download_worker(monkeypatch)
    destination = tmp_path / "nested" / "new.png"
    digest = hashlib.sha256(b"downloaded").hexdigest()
    assert worker._download("/input", destination, expected_sha256=digest) == "image/png"
    assert destination.read_bytes() == b"downloaded"
    transferred = []
    class Response(io.BytesIO):
        pass
    def consume(request, **_kwargs):
        transferred.append(request.data.read())
        return Response(b"{}")
    monkeypatch.setattr("kraken_agent.server_worker.urllib.request.urlopen", consume)
    worker._upload("/output?sha256=" + digest, destination, staging_root=tmp_path)
    assert transferred == [b"downloaded"]


def test_agent_download_rejects_junction_before_external_existing_file_cleanup(tmp_path, monkeypatch):
    worker = download_worker(monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "existing.png"
    sentinel.write_bytes(b"private-preserved")
    staging = tmp_path / "staging"
    staging.mkdir()
    with directory_link(outside, staging / "inputs"):
        with pytest.raises((ValueError, OSError)):
            worker._download("/input", staging / "inputs" / sentinel.name,
                             expected_sha256=hashlib.sha256(b"downloaded").hexdigest())
    assert sentinel.exists(), "An external file was deleted through the staging junction"
    assert sentinel.read_bytes() == b"private-preserved"


def test_agent_download_rejects_junction_before_external_creation(tmp_path, monkeypatch):
    worker = download_worker(monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    with directory_link(outside, staging / "inputs"):
        with pytest.raises((ValueError, OSError)):
            worker._download("/input", staging / "inputs" / "new.png",
                             expected_sha256=hashlib.sha256(b"downloaded").hexdigest())
    assert list(outside.iterdir()) == []


def test_agent_upload_rejects_actual_output_junction(tmp_path, monkeypatch):
    worker = download_worker(monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.png").write_bytes(b"private")
    staging = tmp_path / "staging"
    staging.mkdir()
    with directory_link(outside, staging / "outputs"):
        with pytest.raises((ValueError, OSError)):
            worker._upload("/output?sha256=" + hashlib.sha256(b"private").hexdigest(),
                           staging / "outputs" / "private.png")


@pytest.mark.parametrize("parts", [("..", "outside"), ("../outside",), ("absolute",), ("a/b",)])
@pytest.mark.parametrize("operation", [contained_path, make_contained_directories])
def test_containment_helpers_reject_non_component_paths(tmp_path, parts, operation):
    root = tmp_path / "trusted"
    root.mkdir()
    if parts == ("absolute",):
        parts = (str(tmp_path / "outside"),)
    with pytest.raises(UnsafeFilesystemPath):
        operation(root, parts)


def test_rooted_reader_rejects_parent_traversal(tmp_path):
    root = tmp_path / "trusted"
    root.mkdir()
    outside = tmp_path / "private"
    outside.write_bytes(b"private")
    with pytest.raises(UnsafeFilesystemPath):
        with open_regular_read(root / ".." / "private", root=root) as stream:
            stream.read()


def test_append_revalidates_intermediate_junction_after_open(tmp_path, monkeypatch):
    from contextlib import ExitStack
    from kraken_core import safe_files

    root = tmp_path / "staging"
    parent = root / "logs"
    parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    original_open = os.open
    with ExitStack() as cleanup:
        def replace_parent(path, flags, mode=0o777, **kwargs):
            parent.rename(root / "old-logs")
            cleanup.enter_context(directory_link(outside, parent))
            return original_open(path, flags, mode, **kwargs)
        monkeypatch.setattr(safe_files.os, "open", replace_parent)
        with pytest.raises(UnsafeFilesystemPath):
            with open_regular_append(parent / "plugin.log", root=root) as stream:
                stream.write(b"must not escape staging")
    assert (outside / "plugin.log").read_bytes() == b""
