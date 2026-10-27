"""Work-order authorization, event durability, revision and path boundaries."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from kraken_manager.application.errors import ConcurrencyError
from kraken_manager.application.work_order_lifecycle import record_work_order
from kraken_manager.infrastructure.work_package import selection_from_coordinates
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.work_orders import WorkKind, WorkOrder, WorkOrderState, WorkPlacement
from kraken_manager.domain.workflows import PluginJob
from kraken_manager.infrastructure.filesystem._codec import decode_model, encode_model
from kraken_manager.infrastructure.projections import rebuild_filesystem_index
from kraken_server.persistent_services_parts.create_project import CreateProjectMixin
from kraken_server.persistent_services_parts.issue_work import IssueWorkMixin
from kraken_server.services import CommandContext, ConflictError, ForbiddenError, ValidationError
from tests.test_work_orders import _project_with_frames, _service


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("KRAKEN_PROJECT_ACCESS_MODE", "acl")
    service = _service(tmp_path)
    owner, performer, project, layer, images, _vectors = _project_with_frames(
        service, tmp_path / "source", with_vectors=False,
    )
    factory = service._uow(str(project.id))
    order = WorkOrder.create(
        project_id=project.id, layer_id=layer.id, kind=WorkKind.STITCH,
        placement=WorkPlacement.EXTERNAL, assignee_id=performer.id,
        created_by=owner.principal.id, destination=str(tmp_path / "client-only"),
        frame_count=1, selection=selection_from_coordinates(((1, 1),)),
        created_at=datetime.now(UTC),
    )

    def record(change, *, key="issue", operation="issue", principal=None, command=None):
        return record_work_order(
            uow_factory=factory, profiles=service.profiles,
            principal=principal or owner.principal, project_id=project.id,
            idempotency_key=key, operation=operation, change=change,
            command=command if command is not None else {"order_id": order.id},
            encode=encode_model, decode=decode_model,
        )

    return SimpleNamespace(
        service=service, owner=owner, performer=performer, project=project,
        layer=layer, images=images, factory=factory, order=order, record=record, tmp_path=tmp_path,
    )


def events(setup, order=None):
    with setup.factory() as unit:
        return unit.event_store.load_stream(f"work-order:{(order or setup.order).id}")


def test_work_order_records_actor_performer_and_replays_after_projection_rebuild(setup):
    setup.record(lambda _projections: setup.order)
    cancelled = setup.record(
        lambda projections: projections.get_work_order(setup.order.id).cancel(),
        key="cancel", operation="cancel",
    )
    history = events(setup)
    assert [event.event_type for event in history] == ["WorkOrderRecordedV1"] * 2
    assert [event.revision for event in history] == [1, 2]
    assert history[0].actor.principal_id == setup.owner.principal.id
    assert history[0].performer_id == setup.performer.id
    assert [event.payload["operation"] for event in history] == ["issue", "cancel"]
    with setup.factory() as unit:
        rebuild_filesystem_index(unit._backing_event_store, unit._backing_projections)
    assert setup.service._projection(setup.project.id).get_work_order(setup.order.id) == cancelled


def test_duplicate_operation_returns_original_without_executing_change(setup):
    original = setup.record(lambda _projections: setup.order)

    def must_not_run(_projections):
        pytest.fail("An idempotent retry executed the mutation again")

    assert setup.record(must_not_run) == original
    assert len(events(setup)) == 1
    with pytest.raises(ConcurrencyError):
        setup.record(must_not_run, operation="cancel")


def test_idempotency_key_cannot_return_another_work_order_snapshot(setup):
    setup.record(lambda _projections: setup.order)
    other = replace(setup.order, id=str(uuid4()))
    with pytest.raises(ConcurrencyError):
        setup.record(lambda _projections: other, command={"order_id": other.id})
    assert len(events(setup)) == 1
    assert setup.service._projection(setup.project.id).get_work_order(other.id) is None


def test_stale_aggregate_revision_does_not_append_event_or_overwrite_state(setup):
    setup.record(lambda _projections: setup.order)
    started = setup.record(lambda _projections: setup.order.mark_running(), key="start", operation="start")
    with pytest.raises(ConcurrencyError):
        setup.record(lambda _projections: setup.order.cancel(), key="stale-cancel", operation="cancel")
    assert setup.service._projection(setup.project.id).get_work_order(setup.order.id) == started
    assert len(events(setup)) == 2


def test_projection_failure_rolls_back_staged_event_and_projection(setup, monkeypatch):
    with setup.factory() as unit:
        projection_type = type(unit.projections)
    original = projection_type.save_work_order

    def fail_after_staging(self, order):
        original(self, order)
        raise OSError("injected projection failure")

    monkeypatch.setattr(projection_type, "save_work_order", fail_after_staging)
    with pytest.raises(OSError, match="injected"):
        setup.record(lambda _projections: setup.order)
    assert events(setup) == ()
    assert setup.service._projection(setup.project.id).get_work_order(setup.order.id) is None


class Server(IssueWorkMixin):
    _translate_lifecycle_error = staticmethod(CreateProjectMixin._translate_lifecycle_error)

    def __init__(self, setup):
        from kraken_manager.infrastructure.filesystem import FilesystemEventStore

        self.projections = setup.service._projection(setup.project.id)
        self.events = FilesystemEventStore(setup.service.catalog_root, setup.project.id)
        self.uow_factory = setup.factory
        self.identities = setup.service.identities
        self.profiles = setup.service.profiles
        self.performer_store = setup.service.performers
        self.derived_root = setup.tmp_path / "server-derived"


def context(setup, key="change", revision=0, actor=None):
    return CommandContext(str(actor or setup.owner.principal.id), key, revision)


@pytest.mark.parametrize("method", ["cancel_work_order", "complete_work_order"])
def test_server_cancel_and_complete_require_assign_work_permission(setup, method):
    setup.record(lambda _projections: setup.order)
    outsider = Principal.local(subject="outsider", display_name="Outsider")
    setup.service.identities.save(outsider)
    server = Server(setup)
    with pytest.raises(ForbiddenError):
        getattr(server, method)(str(setup.project.id), setup.order.id, context(setup, actor=outsider.id))
    assert server.projections.get_work_order(setup.order.id) == setup.order
    assert len(events(setup)) == 1


def test_server_cancel_enforces_revision_and_retry_is_idempotent(setup):
    setup.record(lambda _projections: setup.order)
    server = Server(setup)
    with pytest.raises(ConflictError):
        server.cancel_work_order(str(setup.project.id), setup.order.id, context(setup, revision=99))
    command = context(setup, key="cancel")
    first = server.cancel_work_order(str(setup.project.id), setup.order.id, command)
    assert first["state"] == "cancelled"
    assert server.cancel_work_order(str(setup.project.id), setup.order.id, command) == first
    assert len(events(setup)) == 2


@pytest.mark.parametrize("field,value", [
    ("destination", "C:/forged/server/path"),
    ("assignee_id", str(uuid4())),
    ("instructions", "altered"),
    ("placement", WorkPlacement.SERVER),
])
def test_registered_record_cannot_overwrite_existing_work_metadata(setup, field, value):
    setup.record(lambda _projections: setup.order)
    tampered = replace(setup.order.mark_running(), **{field: value})
    server = Server(setup)
    with pytest.raises((ValidationError, ConflictError)):
        server.issue_work_order(
            str(setup.project.id), {"record": encode_model(tampered)}, context(setup, key="register"),
        )
    assert server.projections.get_work_order(setup.order.id) == setup.order


def test_client_record_cannot_register_server_destination(setup):
    server = Server(setup)
    forged = replace(setup.order, placement=WorkPlacement.SERVER)
    with pytest.raises(ValidationError):
        server.issue_work_order(str(setup.project.id), {"record": encode_model(forged)}, context(setup))
    assert server.projections.get_work_order(forged.id) is None


def test_external_registration_and_start_are_audited(setup):
    server = Server(setup)
    issued = server.issue_work_order(
        str(setup.project.id), {"record": encode_model(setup.order)}, context(setup, key="register"),
    )
    stored = decode_model(WorkOrder, issued)
    job = _save_job(setup)
    started = stored.mark_running(job_id=job.id)
    response = server.issue_work_order(
        str(setup.project.id), {"record": encode_model(started)}, context(setup, key="start"),
    )
    assert decode_model(WorkOrder, response) == started
    assert [event.payload["operation"] for event in events(setup)] == [f"register:{setup.order.id}"] * 2


def _save_job(setup, *, project_id=None, layer_id=None):
    job = PluginJob.create(
        project_id=project_id or setup.project.id,
        layer_id=layer_id or setup.layer.id,
        selection=setup.order.selection,
        actor_principal_id=setup.owner.principal.id,
        target_representation_id=setup.images.id,
        capability="stitch",
    )
    with setup.factory() as unit:
        unit.projections.save_plugin_job(job)
        unit.commit()
    return job


@pytest.mark.parametrize("foreign_scope", ["project", "layer", "missing"])
def test_registered_start_rejects_job_outside_order_scope_without_mutation(setup, foreign_scope):
    setup.record(lambda _projections: setup.order)
    if foreign_scope == "project":
        job = _save_job(setup, project_id=str(uuid4()))
        job_id = job.id
    elif foreign_scope == "layer":
        job = _save_job(setup, layer_id=str(uuid4()))
        job_id = job.id
    else:
        job_id = str(uuid4())
    server = Server(setup)
    started = setup.order.mark_running(job_id=job_id)
    with pytest.raises(ValidationError, match="job reference"):
        server.issue_work_order(
            str(setup.project.id), {"record": encode_model(started)}, context(setup, key="start"),
        )
    assert server.projections.get_work_order(setup.order.id) == setup.order
    assert len(events(setup)) == 1


def test_registered_start_checks_expected_revision(setup):
    setup.record(lambda _projections: setup.order)
    server = Server(setup)
    with pytest.raises(ConflictError):
        server.issue_work_order(
            str(setup.project.id), {"record": encode_model(setup.order.mark_running())},
            context(setup, key="start", revision=99),
        )


def test_server_does_not_inspect_external_client_paths_on_completion(setup, monkeypatch):
    setup.record(lambda _projections: setup.order)
    server = Server(setup)

    def forbidden_probe(*_args, **_kwargs):
        pytest.fail("Server attempted to inspect an external client path")

    monkeypatch.setattr("kraken_server.persistent_services_parts.issue_work.package_has_result", forbidden_probe)
    with pytest.raises(ValidationError, match="external"):
        server.complete_work_order(str(setup.project.id), setup.order.id, context(setup))
    assert server.projections.get_work_order(setup.order.id).state is WorkOrderState.ISSUED


def test_server_rejects_forged_persisted_destination_without_reading_it(setup, monkeypatch):
    server = Server(setup)
    order = replace(setup.order, placement=WorkPlacement.SERVER)
    setup.record(lambda _projections: order)

    def forbidden_probe(*_args, **_kwargs):
        pytest.fail("Server inspected a work folder outside the server directory")

    monkeypatch.setattr("kraken_server.persistent_services_parts.issue_work.package_has_result", forbidden_probe)
    with pytest.raises(ValidationError, match="outside"):
        server.complete_work_order(str(setup.project.id), order.id, context(setup))


def test_server_completes_only_its_own_file_package_and_audits_change(setup):
    server = Server(setup)
    destination = server._server_work_directory(setup.project.id, setup.order.id)
    destination.mkdir(parents=True)
    (destination / "seam.bin").write_bytes(b"seam")
    order = replace(setup.order, placement=WorkPlacement.SERVER, destination=str(destination))
    setup.record(lambda _projections: order)
    result = server.complete_work_order(str(setup.project.id), order.id, context(setup, key="complete"))
    assert decode_model(WorkOrder, result).state is WorkOrderState.COMPLETED
    assert [event.payload["operation"] for event in events(setup)] == ["issue", f"complete:{order.id}"]


def test_server_issue_preserves_due_at_and_ignores_client_destination(setup):
    server = Server(setup)
    due_at = datetime.now(UTC) + timedelta(days=3)
    client_path = setup.tmp_path / "do-not-touch-client"
    issued = server.issue_work_order(str(setup.project.id), {
        "kind": "stitch", "placement": "server", "assignee_id": str(setup.performer.id),
        "layer_id": str(setup.layer.id), "coordinates": [[1, 1]],
        "image_representation_id": str(setup.images.id), "due_at": due_at.isoformat(),
        "destination": str(client_path), "destination_parent": str(client_path),
    }, context(setup, key="server-issue"))
    order = decode_model(WorkOrder, issued)
    assert order.due_at == due_at
    assert Path(order.destination).is_relative_to(server.derived_root)
    assert not client_path.exists()
    assert order.transport == "http"
    assert Path(order.destination, "kraken-work.json").is_file()


@pytest.mark.parametrize("kind", ["stitch", "vector_review"])
def test_server_invalid_due_at_returns_validation_error_before_creating_package(setup, kind):
    server = Server(setup)
    with pytest.raises(ValidationError):
        server.issue_work_order(str(setup.project.id), {
            "kind": kind, "placement": "server", "assignee_id": str(setup.performer.id),
            "layer_id": str(setup.layer.id), "coordinates": [[1, 1]],
            "image_representation_id": str(setup.images.id), "due_at": "invalid timestamp",
        }, context(setup, key="bad-due"))
    assert not server.derived_root.exists()


def test_server_review_preserves_due_date_in_batch_and_work_order(setup, monkeypatch):
    server = Server(setup)
    due_at = datetime.now(UTC) + timedelta(days=3)
    batch_id = str(uuid4())
    observed = []

    def create_review_batch(project_id, payload, context):
        observed.append((project_id, payload, context))
        return {"id": batch_id, "revision": 1}

    def export_review_package(_project_id, _batch_id, destination, _context):
        Path(destination).mkdir(parents=True)

    monkeypatch.setattr(server, "create_review_batch", create_review_batch, raising=False)
    monkeypatch.setattr(server, "export_review_package", export_review_package, raising=False)
    response = server.issue_work_order(str(setup.project.id), {
        "kind": "vector_review", "placement": "server", "assignee_id": str(setup.performer.id),
        "layer_id": str(setup.layer.id), "coordinates": [[1, 1]],
        "image_representation_id": str(setup.images.id), "due_at": due_at.isoformat(),
    }, context(setup, key="review-issue"))
    order = decode_model(WorkOrder, response)
    assert order.due_at == due_at
    assert order.related_batch_id == batch_id
    assert observed[0][1]["due_at"] == due_at.isoformat()
    assert observed[0][2].idempotency_key == "review-issue:review"
    assert len(events(setup, order)) == 1
