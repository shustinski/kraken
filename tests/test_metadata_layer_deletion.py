"""A layer created in the catalog can be deleted after a managed PNG import.

Reproduces LOCAL-LAYER-DELETE-METADATA: create_layer does not write a two-root
file binding, and import into a representation does not either. Deletion must
still archive the layer and record LayerDeletedV1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind
from kraken_manager.infrastructure.filesystem.event_store import FilesystemEventStore


def _imported_catalog_layer(tmp_path: Path):
    service = EmbeddedProjectService(tmp_path / "data")
    owner = service.create_initial_account("owner", "Owner", "secret")
    project = service.create_project(
        principal=owner.principal,
        name="Аудит",
        width=1,
        height=1,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key="project-audit",
    )
    layer = service.create_layer(
        principal=owner.principal,
        project=project,
        name="M1",
        layer_type=LayerType.METAL,
        order=1,
        idempotency_key="layer-m1",
    )
    images = service.create_representation(
        principal=owner.principal,
        project=service.get_project(project.id),
        layer=layer,
        name="Images",
        kind=RepresentationKind.IMAGE,
        active=True,
        idempotency_key="images",
    )
    source = tmp_path / "frames"
    source.mkdir()
    (source / "1_1.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01\x00\x05\xfe\x02\xfe\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    project = service.get_project(project.id)
    layer = service.list_layers(project.id)[0]
    service.commit_managed_import(
        principal=owner.principal,
        project=project,
        layer=layer,
        representation=images,
        plan=service.plan_import_directory(project=project, directory=source),
        idempotency_key="import-png",
    )
    layer = service.list_layers(project.id)[0]
    return service, owner, project, layer


def _deleted_events(service: EmbeddedProjectService, project_id) -> list:
    return [event for event in service.history(project_id) if event.event_type == "LayerDeletedV1"]


def test_imported_catalog_layer_can_be_deleted_and_stays_deleted(tmp_path: Path) -> None:
    service, owner, project, layer = _imported_catalog_layer(tmp_path)
    assert service.workspace_registry.get_layer(str(project.id), str(layer.id)) is None

    service.delete_layer(
        principal=owner.principal,
        project=service.get_project(project.id),
        layer=layer,
        confirmation_name="M1",
        idempotency_key="delete-m1",
    )

    assert service.list_layers(project.id) == ()
    deleted = _deleted_events(service, project.id)
    assert len(deleted) == 1
    assert deleted[0].payload["managed_files"] is False
    reopened = EmbeddedProjectService(tmp_path / "data")
    assert reopened.list_layers(project.id) == ()
    archived = reopened.list_layers(project.id, include_archived=True)
    assert [item.name for item in archived] == ["M1"]
    reopened_deleted = _deleted_events(reopened, project.id)
    assert len(reopened_deleted) == 1
    assert reopened_deleted[0].payload["managed_files"] is False


def test_unbound_delete_retry_writes_the_audit_after_archive(tmp_path: Path, monkeypatch) -> None:
    service, owner, project, layer = _imported_catalog_layer(tmp_path)
    original = FilesystemEventStore.append_preserved
    failed = {"done": False}

    def fail_first_audit(self, events):
        batch = list(events)
        if not failed["done"] and any(getattr(event, "event_type", None) == "LayerDeletedV1" for event in batch):
            failed["done"] = True
            raise OSError("injected audit failure")
        return original(self, batch)

    monkeypatch.setattr(FilesystemEventStore, "append_preserved", fail_first_audit)
    with pytest.raises(OSError, match="injected audit failure"):
        service.delete_layer(
            principal=owner.principal,
            project=service.get_project(project.id),
            layer=layer,
            confirmation_name="M1",
            idempotency_key="delete-m1",
        )
    assert _deleted_events(service, project.id) == []

    service.delete_layer(
        principal=owner.principal,
        project=service.get_project(project.id),
        layer=layer,
        confirmation_name="M1",
        idempotency_key="delete-m1",
    )

    deleted = _deleted_events(service, project.id)
    assert len(deleted) == 1
    assert deleted[0].payload["managed_files"] is False
    assert service.list_layers(project.id) == ()
