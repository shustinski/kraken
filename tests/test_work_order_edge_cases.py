"""Independent regression cases from the issued-work security review."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import PureWindowsPath
from zipfile import ZipFile

import pytest

from kraken_manager.domain.identity import PrincipalProvider
from kraken_manager.domain.project import LayerType
from kraken_manager.infrastructure import work_package
from kraken_server.services import ConflictError, ValidationError
from tests import test_work_order_security as helpers
from tests.test_work_order_security import Server, context

setup = helpers.setup


def issue_payload(setup, **changes):
    return {
        "kind": "stitch",
        "placement": "server",
        "assignee_id": str(setup.performer.id),
        "layer_id": str(setup.layer.id),
        "coordinates": [[1, 1]],
        "image_representation_id": str(setup.images.id),
        **changes,
    }


def test_server_issue_same_key_different_command_is_conflict(setup):
    server = Server(setup)
    payload = issue_payload(setup, instructions="First instructions")
    server.issue_work_order(str(setup.project.id), payload, context(setup, key="same-key"))
    with pytest.raises(ConflictError):
        server.issue_work_order(
            str(setup.project.id), payload | {"instructions": "Different instructions"}, context(setup, key="same-key")
        )


def test_server_issue_cannot_relabel_other_layer_images_as_target_layer(setup):
    other_layer = setup.service.create_layer(
        principal=setup.owner.principal,
        project=setup.service.get_project(setup.project.id),
        name="M2",
        layer_type=LayerType.METAL,
        order=1,
        idempotency_key="other-layer",
    )
    server = Server(setup)
    payload = issue_payload(setup, layer_id=str(other_layer.id))
    with pytest.raises((ValidationError, ConflictError)):
        server.issue_work_order(str(setup.project.id), payload, context(setup, key="wrong-layer"))
    assert server.projections.list_work_orders(setup.project.id) == ()
    assert not server.derived_root.exists()


def test_verified_gitlab_actor_can_issue_work_after_transport_live_check(setup):
    # The server endpoint has already run shared_mutation_actor's live identity
    # check. Its persistent handler must preserve that authorization fact.
    actor = replace(setup.owner.principal, provider=PrincipalProvider.GITLAB, issuer="https://gitlab.example")
    setup.service.identities.save(actor)
    server = Server(setup)
    issued = server.issue_work_order(
        str(setup.project.id), issue_payload(setup), context(setup, key="gitlab-work", actor=actor.id)
    )
    assert issued["created_by"] == actor.id


@pytest.mark.parametrize(
    "unsafe_name", ["C:payload.txt", "images/frame.png:stream", "C:/payload.txt", "../payload.txt"]
)
def test_dataset_archive_rejects_windows_path_escape_and_ads_without_writing(tmp_path, monkeypatch, unsafe_name):
    data = BytesIO()
    with ZipFile(data, "w") as archive:
        archive.writestr(unsafe_name, b"unsafe")
    writes = []
    monkeypatch.setattr(work_package, "write_exclusive", lambda path, content: writes.append(path))
    with pytest.raises(ValueError):
        work_package.extract_dataset_archive(data.getvalue(), tmp_path / "package" / "dataset")
    assert writes == []


def test_drive_relative_archive_member_is_not_contained_by_package_root():
    # A platform-independent proof for the lexical Windows drive-relative case.
    root = PureWindowsPath("D:/project/package")
    assert not (root / PureWindowsPath("C:payload.txt")).is_relative_to(root)


@pytest.mark.parametrize("field", ["image_representation_id", "vector_representation_id", "dataset_version_id"])
def test_registered_external_work_cannot_reference_unknown_or_foreign_artifacts(setup, field):
    from uuid import uuid4
    from kraken_manager.infrastructure.filesystem._codec import encode_model

    server = Server(setup)
    forged = replace(setup.order, **{field: str(uuid4())})
    with pytest.raises(ValidationError):
        server.issue_work_order(
            str(setup.project.id), {"record": encode_model(forged)}, context(setup, key=f"bad-{field}")
        )
    assert server.projections.get_work_order(forged.id) is None


def test_image_representation_cannot_be_used_as_vector_output_reference(setup):
    server = Server(setup)
    with pytest.raises(ValidationError, match="representation"):
        server.issue_work_order(
            str(setup.project.id),
            issue_payload(setup, vector_representation_id=str(setup.images.id)),
            context(setup, key="wrong-kind"),
        )
    assert not server.derived_root.exists()


def test_frame_materialization_rejects_alternate_stream_filename_before_writes(setup):
    from types import SimpleNamespace

    version = SimpleNamespace(filename="mask.png:stream", id="version", sha256="a" * 64, media_type="image/png")
    with pytest.raises(ValueError):
        work_package.lookup_frame_files(
            project=setup.project,
            representation_id=setup.images.id,
            selection=setup.order.selection,
            active_version=lambda series_id: version,
            role="image",
        )


def test_failed_package_creation_removes_only_its_new_directory(setup):
    from datetime import UTC, datetime

    parent = setup.tmp_path / 'removable'
    parent.mkdir()
    existing = parent / 'existing.txt'
    existing.write_text('keep', encoding='utf-8')
    request = work_package.WorkIssueRequest(
        project=setup.project, layer=setup.layer, kind='train', placement='external',
        assignee_id=setup.performer.id, created_by=setup.owner.principal.id,
        coordinates=(), image_representation_id='', vector_representation_id='',
        instructions='', due_at=None, destination_parent=parent,
        server_directory_for=lambda _id: setup.tmp_path / 'unused',
        idempotency_key='missing-dataset', transport='files', remote=False, now=datetime.now(UTC),
    )
    with pytest.raises(ValueError, match='выборку'):
        work_package.place_work_order(
            request, active_version=lambda _id: None, read_bytes=lambda _id: b'', dataset_version=lambda: None,
        )
    assert list(parent.iterdir()) == [existing]
    assert existing.read_text(encoding='utf-8') == 'keep'
