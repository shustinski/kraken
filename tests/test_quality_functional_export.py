"""Export must preserve pre-existing destinations and clean up only its own partial file."""
from __future__ import annotations

import pytest

from kraken_manager.infrastructure.blob.filesystem import FilesystemBlobStore

pytest_plugins = ["tests.test_quality_workflow"]


def test_managed_quality_export_preserves_preexisting_file(quality, tmp_path):
    q = quality
    destination = tmp_path / "already-reviewed.cif"
    destination.write_bytes(b"existing independent user export")
    with pytest.raises(FileExistsError):
        q.service.export_artifact_version(q.project.id, q.corrected, destination)
    assert destination.read_bytes() == b"existing independent user export"


def test_external_quality_export_preserves_preexisting_file(quality, tmp_path):
    q = quality
    source = tmp_path / "external.cif"
    source.write_bytes(b"verified external source bytes")
    version = q.service.add_external_artifact_version(principal=q.owner, project_id=q.project.id,
        series_id=q.corrected.series_id, source=source, parent_version_id=q.corrected.id, idempotency_key="external-export-source")
    destination = tmp_path / "external-destination.cif"
    destination.write_bytes(b"existing external destination")
    with pytest.raises(FileExistsError):
        q.service.export_artifact_version(q.project.id, version, destination)
    assert destination.read_bytes() == b"existing external destination"
    assert source.read_bytes() == b"verified external source bytes"


def test_quality_export_removes_own_partial_output_on_stream_failure_and_can_retry(quality, tmp_path, monkeypatch):
    q = quality
    destination = tmp_path / "streamed.cif"
    original = FilesystemBlobStore.iter_bytes

    def broken(*_args, **_kwargs):
        yield b"partial created output"
        raise OSError("injected quality export interruption")

    monkeypatch.setattr(FilesystemBlobStore, "iter_bytes", broken)
    with pytest.raises(OSError, match="interruption"):
        q.service.export_artifact_version(q.project.id, q.corrected, destination)
    assert not destination.exists()
    monkeypatch.setattr(FilesystemBlobStore, "iter_bytes", original)
    q.service.export_artifact_version(q.project.id, q.corrected, destination)
    assert destination.read_bytes() == b"corrected image"


def test_export_cleanup_preserves_replacement_owned_by_another_writer(tmp_path):
    from kraken_hub.composition.parts.export_artifact_version import _cleanup_created_export

    destination = tmp_path / "export.cif"
    destination.write_bytes(b"our partial output")
    created = destination.stat()
    replacement = tmp_path / "replacement.cif"
    replacement.write_bytes(b"another writer's completed output")
    replacement.replace(destination)
    _cleanup_created_export(destination, (created.st_dev, created.st_ino))
    assert destination.read_bytes() == b"another writer's completed output"
