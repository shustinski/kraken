"""Returned CIF copies are complete, verified, and safe to retry."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from kraken_manager.domain.workflows import PluginFrameOutcome
from kraken_manager.infrastructure.plugin.vector_workspace import materialize_vector_result
from kraken_manager.infrastructure.plugin import vector_workspace
from kraken_manager.infrastructure.workspace_files import WorkspaceFileService, WorkspaceRegistry
from kraken_manager.workspace import WorkspaceValidationError


@pytest.fixture
def result(tmp_path):
    source, derived = tmp_path / "source", tmp_path / "derived"
    source.mkdir()
    derived.mkdir()
    files = WorkspaceFileService(WorkspaceRegistry(tmp_path / "catalog"))
    binding = files.create_project(
        project_id="project", project_name="Project", source_root=source, derived_root=derived
    )
    data = b"L M1; E;\n"
    output = SimpleNamespace(
        outcome=PluginFrameOutcome.SUCCEEDED,
        relative_path="outputs/0000.cif",
        sha256=hashlib.sha256(data).hexdigest(),
        role="vector",
        media_type="application/x-cif",
    )
    job = SimpleNamespace(
        id=str(uuid4()),
        project_id="project",
        capability="frames.vectorize.v1",
        created_at=datetime(2026, 10, 15, 6, 32, 53, tzinfo=UTC),
    )
    manifest = SimpleNamespace(results=(output,), outcome="partial")
    reader = SimpleNamespace(iter_output=lambda *_args: iter((data,)))
    arguments = dict(job=job, layer_name="Metal", manifest=manifest, reader=reader)
    destination = (
        Path(binding.derived_project_dir)
        / "vector"
        / "Metal"
        / ("Contour — " + job.created_at.astimezone().strftime("%d.%m.%Y %H-%M-%S"))
    )
    return files, arguments, destination, data


def test_partial_outputs_are_saved_without_accepting_the_domain_result_and_retry_is_idempotent(result):
    files, arguments, destination, data = result
    assert materialize_vector_result(files, **arguments) == destination
    assert (destination / "0000.cif").read_bytes() == data
    assert arguments["manifest"].outcome == "partial"
    assert materialize_vector_result(files, **arguments) == destination
    assert list(destination.parent.iterdir()) == [destination]


def test_bad_digest_never_publishes_incomplete_folder(result):
    files, arguments, destination, _data = result
    arguments["reader"].iter_output = lambda *_args: iter((b"damaged",))
    with pytest.raises(WorkspaceValidationError, match="SHA-256"):
        materialize_vector_result(files, **arguments)
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []


def test_existing_user_changes_are_preserved(result):
    files, arguments, destination, _data = result
    materialize_vector_result(files, **arguments)
    (destination / "0000.cif").write_bytes(b"user edit")
    with pytest.raises(WorkspaceValidationError, match="changed"):
        materialize_vector_result(files, **arguments)
    assert (destination / "0000.cif").read_bytes() == b"user edit"


def test_no_workspace_binding_does_not_create_paths(result):
    files, arguments, destination, _data = result
    arguments["job"].project_id = "legacy"
    assert materialize_vector_result(files, **arguments) is None
    assert not destination.exists()


def test_invalid_layer_name_is_rejected(result):
    files, arguments, destination, _data = result
    arguments["layer_name"] = "../outside"
    with pytest.raises(WorkspaceValidationError):
        materialize_vector_result(files, **arguments)
    assert not destination.exists()


def test_simultaneous_jobs_get_numbered_names_without_overwriting(result):
    files, arguments, destination, data = result
    materialize_vector_result(files, **arguments)
    arguments["job"].id = str(uuid4())
    second = materialize_vector_result(files, **arguments)
    assert second.name == destination.name + " (2)"
    assert materialize_vector_result(files, **arguments) == second
    assert (destination / "0000.cif").read_bytes() == data


def test_legacy_uuid_folder_is_renamed_without_changing_files(result):
    files, arguments, destination, data = result
    legacy = destination.parent / arguments["job"].id
    legacy.mkdir(parents=True)
    (legacy / "0000.cif").write_bytes(data)
    assert materialize_vector_result(files, **arguments) == destination
    assert not legacy.exists()
    assert (destination / "0000.cif").read_bytes() == data


def test_removing_result_keeps_other_jobs_and_cannot_be_recreated(result):
    files, arguments, destination, data = result
    materialize_vector_result(files, **arguments)
    sibling = destination.parent / "Another result"
    sibling.mkdir()
    (sibling / "keep.cif").write_bytes(data)
    vector_workspace.remove_vector_result(files, project_id="project", job_id=arguments["job"].id)
    assert not destination.exists()
    assert (sibling / "keep.cif").read_bytes() == data
    assert arguments["job"].id not in files.registry.plugin_result_directories("project")
    vector_workspace.remove_vector_result(files, project_id="project", job_id=arguments["job"].id)
    assert materialize_vector_result(files, **arguments) is None
    assert not destination.exists()


def test_deleting_before_return_prevents_late_materialization(result):
    files, arguments, destination, _data = result
    vector_workspace.remove_vector_result(files, project_id="project", job_id=arguments["job"].id)
    assert materialize_vector_result(files, **arguments) is None
    assert not destination.exists()


def test_deletion_rejects_path_outside_individual_vector_result(result):
    files, arguments, destination, data = result
    materialize_vector_result(files, **arguments)
    files.registry.save_plugin_result_directory("project", arguments["job"].id, "vector")
    with pytest.raises(WorkspaceValidationError):
        vector_workspace.remove_vector_result(files, project_id="project", job_id=arguments["job"].id)
    assert (destination / "0000.cif").read_bytes() == data


def test_failed_file_deletion_can_be_retried_without_republishing(result, monkeypatch):
    files, arguments, destination, _data = result
    materialize_vector_result(files, **arguments)
    original = vector_workspace.shutil.rmtree

    def fail(_path):
        raise PermissionError("File is open")

    monkeypatch.setattr(vector_workspace.shutil, "rmtree", fail)
    with pytest.raises(PermissionError, match="File is open"):
        vector_workspace.remove_vector_result(files, project_id="project", job_id=arguments["job"].id)
    assert destination.exists()
    assert materialize_vector_result(files, **arguments) is None
    monkeypatch.setattr(vector_workspace.shutil, "rmtree", original)
    vector_workspace.remove_vector_result(files, project_id="project", job_id=arguments["job"].id)
    assert not destination.exists()
