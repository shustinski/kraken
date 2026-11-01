"""Repeating local project creation with the same idempotency key is safe.

Reproduces LOCAL-PROJECT-IDEMPOTENT: the second call must return the project
already stored under that key and must not try to create its folder again.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from uuid import uuid4

import pytest

from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.composition.parts.create_project import project_id_for_idempotency_key
from kraken_manager.application.dto import CommandContext, CreateProjectCommand
from kraken_manager.domain.common import ProjectId
from kraken_manager.domain.project import GridOrientation, LayerType
from kraken_manager.infrastructure.filesystem.event_store import FilesystemEventStore


def _service(tmp_path: Path) -> EmbeddedProjectService:
    return EmbeddedProjectService(tmp_path / "data")


def _create(service: EmbeddedProjectService, principal, *, name: str, key: str):
    return service.create_project(
        principal=principal,
        name=name,
        width=2,
        height=1,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key=key,
    )


def test_same_idempotency_key_returns_the_stored_project(tmp_path: Path) -> None:
    service = _service(tmp_path)
    owner = service.create_initial_account("owner", "Owner", "secret")
    first = _create(service, owner.principal, name="Аудит", key="project-audit")

    second = _create(service, owner.principal, name="Аудит", key="project-audit")

    assert second.id == first.id
    assert second.name == "Аудит"
    assert second.width == 2
    assert second.height == 1
    source = tmp_path / "data" / "workspace-source"
    assert sorted(path.name for path in source.iterdir() if path.is_dir()) == ["Аудит"]

    reopened = _service(tmp_path)
    stored = reopened.get_project(first.id)
    assert stored is not None
    assert stored.id == first.id
    assert stored.name == "Аудит"
    assert stored.width == 2
    assert stored.height == 1


def test_retry_replaces_a_workspace_left_without_a_project_journal(tmp_path: Path) -> None:
    service = _service(tmp_path)
    owner = service.create_initial_account("owner", "Owner", "secret")
    project_id = project_id_for_idempotency_key(owner.principal.id, "project-audit")
    service.workspace_files.create_project(
        project_id=project_id,
        project_name="Аудит",
        source_root=service.default_source_root,
        derived_root=service.default_derived_root,
    )
    assert service.get_project(project_id) is None

    created = _create(service, owner.principal, name="Аудит", key="project-audit")

    assert str(created.id) == project_id
    reopened = _service(tmp_path)
    stored = reopened.get_project(project_id)
    assert stored is not None
    assert stored.name == "Аудит"
    source = tmp_path / "data" / "workspace-source"
    assert sorted(path.name for path in source.iterdir() if path.is_dir()) == ["Аудит"]


def test_same_key_returns_a_project_created_under_a_random_id(tmp_path: Path) -> None:
    service = _service(tmp_path)
    owner = service.create_initial_account("owner", "Owner", "secret")
    legacy_id = str(uuid4())
    binding = service.workspace_files.create_project(
        project_id=legacy_id,
        project_name="Аудит",
        source_root=service.default_source_root,
        derived_root=service.default_derived_root,
    )
    marker = Path(binding.source_project_dir) / "img" / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    service._commands(legacy_id).execute(
        CreateProjectCommand(
            context=CommandContext(actor=owner.principal, idempotency_key="legacy-key"),
            name="Аудит",
            width=2,
            height=1,
            orientation=GridOrientation.Y_DOWN,
            storage_profile_id=service.profile.id,
            project_id=ProjectId(legacy_id),
        )
    )

    found = _create(service, owner.principal, name="Аудит", key="legacy-key")

    assert str(found.id) == legacy_id
    assert marker.read_text(encoding="utf-8") == "keep"
    service.accounts.create_account("other", "Other", "secret")
    other = service.login("other", "secret")
    assert other is not None
    with pytest.raises(FileExistsError, match="Аудит"):
        _create(service, other.principal, name="Аудит", key="legacy-key")
    assert marker.read_text(encoding="utf-8") == "keep"


def test_retry_writes_the_workspace_event_missed_after_the_journal(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    owner = service.create_initial_account("owner", "Owner", "secret")
    original = service._record_workspace_event

    def fail_once(**kwargs):
        if kwargs.get("event_type") == "ProjectWorkspaceBoundV1":
            service._record_workspace_event = original
            raise OSError("injected workspace event failure")
        return original(**kwargs)

    service._record_workspace_event = fail_once
    with pytest.raises(OSError, match="injected workspace event failure"):
        _create(service, owner.principal, name="Аудит", key="project-audit")

    project = _create(service, owner.principal, name="Аудит", key="project-audit")

    again = _create(service, owner.principal, name="Аудит", key="project-audit")
    assert str(again.id) == str(project.id)
    events = FilesystemEventStore(service.catalog_root, str(project.id)).find_by_idempotency_key(
        project.id, "project-audit:workspace"
    )
    assert len(events) == 1
    assert events[0].event_type == "ProjectWorkspaceBoundV1"


def test_overlapping_creates_append_one_workspace_event(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path)
    owner = service.create_initial_account("owner", "Owner", "secret")
    original = FilesystemEventStore.append

    def slow_append(self, stream_id, *, expected_revision, events):
        batch = list(events)
        if any(getattr(event, "event_type", None) == "ProjectWorkspaceBoundV1" for event in batch):
            time.sleep(0.4)
        return original(self, stream_id, expected_revision=expected_revision, events=batch)

    monkeypatch.setattr(FilesystemEventStore, "append", slow_append)
    errors: list[BaseException] = []
    found: list[object] = []

    def run() -> None:
        try:
            found.append(_create(service, owner.principal, name="Аудит", key="project-audit"))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run), threading.Thread(target=run)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert {str(project.id) for project in found} == {str(found[0].id)}
    events = FilesystemEventStore(service.catalog_root, str(found[0].id)).find_by_idempotency_key(
        found[0].id, "project-audit:workspace"
    )
    assert len(events) == 1


def test_retry_finishes_a_layer_template_interrupted_after_the_project(tmp_path: Path) -> None:
    service = _service(tmp_path)
    owner = service.create_initial_account("owner", "Owner", "secret")
    real_create_layer = service.create_layer
    calls = {"count": 0}

    def fail_second(**kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("injected layer failure")
        return real_create_layer(**kwargs)

    service.create_layer = fail_second
    with pytest.raises(OSError, match="injected layer failure"):
        service.create_project(
            principal=owner.principal,
            name="Аудит",
            width=2,
            height=1,
            orientation=GridOrientation.Y_DOWN,
            idempotency_key="project-audit",
            layer_template=True,
        )

    service.create_layer = real_create_layer
    project = service.create_project(
        principal=owner.principal,
        name="Аудит",
        width=2,
        height=1,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key="project-audit",
        layer_template=True,
    )

    again = service.create_project(
        principal=owner.principal,
        name="Аудит",
        width=2,
        height=1,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key="project-audit",
        layer_template=True,
    )
    assert str(again.id) == str(project.id)
    layers = service.list_layers(project.id)
    assert {layer.type for layer in layers} == set(LayerType)
    assert len(layers) == len(LayerType)


def test_a_different_key_still_refuses_an_existing_project_folder(tmp_path: Path) -> None:
    service = _service(tmp_path)
    owner = service.create_initial_account("owner", "Owner", "secret")
    _create(service, owner.principal, name="Аудит", key="project-audit")

    with pytest.raises(FileExistsError, match="Аудит"):
        _create(service, owner.principal, name="Аудит", key="another-project")
