from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from tests.test_work_orders import _service, _project_with_frames
from kraken_hub.presentation.qt.parts.send_selection_review import SendSelectionReviewMixin
from kraken_hub.presentation.qt.parts.submit_agent_action import SubmitAgentActionMixin


@pytest.mark.parametrize("selected", [(), ((2, 1),), ((1, 1), (2, 1))])
def test_http_order_uses_immutable_assigned_selection(tmp_path, selected):
    service = _service(tmp_path)
    owner, performer, project, layer, images, vectors = _project_with_frames(service, tmp_path / "source")
    order = service.issue_work(
        principal=owner.principal,
        project_id=project.id,
        layer_id=layer.id,
        kind="vectorize",
        placement="server",
        assignee_id=performer.id,
        coordinates=((1, 1),),
        image_representation_id=str(images.id),
        vector_representation_id=str(vectors.id),
        idempotency_key="vectorize",
    )
    calls = []

    class Controller(SendSelectionReviewMixin, SubmitAgentActionMixin):
        pass

    controller = Controller()
    controller._project_id = project.id
    controller.session = owner
    controller._workspace = SimpleNamespace(
        matrix_view=SimpleNamespace(selected_coordinates=lambda **_: selected),
        vector_representation_combo=SimpleNamespace(currentData=lambda: str(vectors.id)),
    )
    controller.agent_runtime = SimpleNamespace(interactive_registry=lambda _: object())
    controller._run_remote_interactive_job = lambda *args, **kwargs: None
    controller.service = SimpleNamespace(
        get_project=lambda _: project,
        is_remote_project=lambda _: True,
        profile=service.profile,
        list_representations=lambda *args: (images, vectors),
        list_layers=lambda *_args: (layer,),
        create_representation=lambda **_kwargs: vectors,
        submit_plugin_job=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(id=str(uuid4())),
        mark_work_running=lambda *args, **kwargs: None,
    )
    controller.open_issued_work(order)
    assert tuple((c.x, c.y) for c in order.selection.iter_coordinates()) == ((1, 1),)
    assert calls[0]["coordinates"] == ((1, 1),)


@pytest.mark.parametrize("transport", [None, "http", "files"])
def test_server_stitch_is_rejected_before_assignment(tmp_path, transport):
    service = _service(tmp_path)
    owner, performer, project, layer, images, _ = _project_with_frames(service, tmp_path / "source")
    with pytest.raises(ValueError, match="stitch"):
        service.issue_work(
            principal=owner.principal,
            project_id=project.id,
            layer_id=layer.id,
            kind="stitch",
            placement="server",
            assignee_id=performer.id,
            coordinates=((1, 1),),
            image_representation_id=str(images.id),
            idempotency_key="stitch",
            transport=transport,
        )
    assert service.work_orders() == ()
    assert not (service.data_dir / "server-work").exists()


def _external_order(tmp_path):
    service = _service(tmp_path)
    owner, performer, project, layer, images, _ = _project_with_frames(service, tmp_path / "source", with_vectors=False)
    parent = tmp_path / "usb"
    parent.mkdir()
    order = service.issue_work(
        principal=owner.principal,
        project_id=project.id,
        layer_id=layer.id,
        kind="vectorize",
        placement="external",
        assignee_id=performer.id,
        coordinates=((1, 1),),
        image_representation_id=str(images.id),
        destination_parent=parent,
        idempotency_key="external",
    )
    return service, owner, project, order


def _write_result(order, project, *, digest=None, job_id=None):
    import hashlib
    from kraken_core.plugin_protocol import PluginResultManifest, PluginFrameOutput

    content = b"DS 1 1 1;\nDF;\nE\n"
    (Path(order.destination) / "outputs" / "1_1.cif").write_bytes(content)
    result = PluginResultManifest(
        job_id=job_id or str(order.id),
        outcome="succeeded",
        plugin_id="contour",
        plugin_version="test",
        outputs=(
            PluginFrameOutput(
                output_id=str(uuid4()),
                frame_id=str(project.frame_id_at(1, 1)),
                relative_path="outputs/1_1.cif",
                sha256=digest or hashlib.sha256(content).hexdigest(),
                media_type="application/x-cif",
                role="vector",
            ),
        ),
    )
    (Path(order.destination) / "result.json").write_text(result.to_json(), encoding="utf-8")
    return content


def test_external_vectorize_completion_imports_verified_result(tmp_path):
    from kraken_manager.domain.artifacts import deterministic_frame_series_id

    service, owner, project, order = _external_order(tmp_path)
    assert len(service.plugin_jobs()) == 1
    job = service.plugin_jobs()[0]
    assert str(job.id) == order.related_job_id == order.id
    assert str(job.target_representation_id) != order.image_representation_id
    content = _write_result(order, project)
    completed = service.complete_work_order(order, principal=owner.principal, idempotency_key="finish")
    assert completed.state.value == "completed"
    assert service.plugin_jobs()[0].state.value == "succeeded"
    series = deterministic_frame_series_id(job.target_representation_id, project.frame_id_at(1, 1))
    version = service._projection(project.id).get_active_artifact_version(series)
    assert version is not None
    assert service.read_project_blob(project.id, version.sha256) == content
    repeated = service.complete_work_order(order, principal=owner.principal, idempotency_key="finish")
    assert repeated == completed


@pytest.mark.parametrize("result", ["missing", "wrong_hash", "other_job"])
def test_external_vectorize_cannot_complete_without_valid_import(tmp_path, result):
    service, owner, project, order = _external_order(tmp_path)
    if result == "missing":
        (Path(order.destination) / "outputs" / "1_1.cif").write_bytes(b"unregistered")
    else:
        _write_result(
            order,
            project,
            digest="0" * 64 if result == "wrong_hash" else None,
            job_id=str(uuid4()) if result == "other_job" else None,
        )
    with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
        service.complete_work_order(order, principal=owner.principal, idempotency_key="finish")
    assert service.work_orders()[0].state.value == "issued"


def test_assigned_performer_returns_files_for_issuing_principal_to_accept(tmp_path):
    from dataclasses import replace
    from kraken_manager.domain.identity import Principal, ProjectRole, ProjectRoleAssignment

    service, owner, project, order = _external_order(tmp_path)
    performer_principal = Principal.local(subject="performer", display_name="Performer")
    service.identities.save(performer_principal)
    performer = service.performers.get(order.assignee_id)
    service.performers.update(replace(performer, principal_id=performer_principal.id))
    service.identities.assign(
        ProjectRoleAssignment.create(
            project_id=project.id,
            principal_id=performer_principal.id,
            role=ProjectRole.ADMIN,
            assigned_by=owner.principal.id,
        )
    )
    _write_result(order, project)
    with pytest.raises(ValueError, match="автор"):
        service.complete_work_order(order, principal=performer_principal, idempotency_key="performer-complete")
    assert service.work_orders()[0].state.value == "issued"
    accepted = service.complete_work_order(order, principal=owner.principal, idempotency_key="owner-accept")
    assert accepted.state.value == "completed"


def test_remote_external_vectorization_is_rejected_before_export(tmp_path):
    from kraken_hub.infrastructure.remote.parts.issue_work import IssueWorkMixin

    class Remote(IssueWorkMixin):
        pass

    with pytest.raises(ValueError, match="HTTP"):
        Remote().issue_work(
            principal=None,
            project_id=str(uuid4()),
            layer_id=str(uuid4()),
            kind="vectorize",
            placement="external",
            assignee_id=str(uuid4()),
            destination_parent=tmp_path,
            idempotency_key="remote-external",
        )
    assert list(tmp_path.iterdir()) == []


def test_external_file_jobs_are_not_polled_through_agent(tmp_path):
    service, owner, _project, order = _external_order(tmp_path)

    def unexpected_poll(job_id):
        pytest.fail(f"External work job {job_id} was sent to Agent")

    jobs = service.synchronize_plugin_jobs(
        principal=owner.principal,
        gateway=SimpleNamespace(get_job=unexpected_poll),
    )
    assert len(jobs) == 1 and str(jobs[0].id) == order.id
    assert jobs[0].state.value == "queued"
