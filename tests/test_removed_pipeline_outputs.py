"""A removed processing result cannot feed subsequent operations."""

import hashlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from kraken_manager.application import CommandContext, ImportPluginResultCommand, ImportPluginResultHandler
from kraken_manager.application.errors import ConflictError
from kraken_manager.domain.workflows import PluginFrameOutcome, PluginFrameResultV1, PluginResultManifestV1
from kraken_server.services import CommandContext as ServerContext, ConflictError as ServerConflictError
from test_server_workspace_inputs_parts.desktop_job import _desktop_job


def _result(local, session, project, job):
    data = b"L M1; E;\n"
    manifest = PluginResultManifestV1(
        job_id=job.id,
        plugin_name="Contour",
        plugin_version="1",
        parameters_applied={},
        results=tuple(
            PluginFrameResultV1(
                output_id=str(uuid4()),
                frame_id=coord.frame_id(project.id),
                outcome=PluginFrameOutcome.SUCCEEDED,
                relative_path=f"outputs/{coord.x}_{coord.y}.cif",
                sha256=hashlib.sha256(data).hexdigest(),
                media_type="application/x-cif",
                role="vector",
            )
            for coord in job.selection.iter_coordinates()
        ),
    )
    importer = ImportPluginResultHandler(
        local._uow(str(project.id)),
        local.profiles,
        local.clock,
        SimpleNamespace(iter_output=lambda *_args: iter((data,))),
    )
    command = ImportPluginResultCommand(
        context=CommandContext(actor=session.principal, idempotency_key="import"),
        manifest=manifest,
    )
    return importer, command


@pytest.mark.parametrize("backend", ["local", "server"])
def test_removed_vectors_are_archived_and_rejected_as_dataset_inputs(tmp_path, backend):
    server, local, session, project, layer, vectors, _gateway, submitted = _desktop_job(tmp_path, "frames.vectorize.v1")
    job = local.plugin_jobs()[0]
    importer, command = _result(local, session, project, job)
    imported = importer(command)
    creation = next(event for event in local.history(project.id) if event.event_type == "PluginJobCreated")
    if backend == "local":
        local.remove_layer_pipeline_action(
            principal=session.principal,
            project_id=project.id,
            layer_id=layer.id,
            action_event_id=creation.event_id,
        )
    else:
        server._pipeline_events = lambda _pid: local.history(project.id)
        stream = f"layer-pipeline:{layer.id}"
        server.append_pipeline_event(
            str(project.id),
            {
                "event_type": "LayerPipelineActionRemoved",
                "layer_id": str(layer.id),
                "action_event_id": creation.event_id,
            },
            ServerContext(str(session.principal.id), "remove", server.events.current_revision(stream)),
        )
    projection = local._projection(project.id)
    assert all(projection.get_artifact_series(v.series_id).archived for v in imported.versions)
    assert local.frame_cells(project.id, layer.id, vectors.id) == ()
    with pytest.raises(ServerConflictError, match="No published CIF"):
        server.submit_plugin_job(
            str(project.id),
            {
                "layer_id": str(layer.id),
                "source_representation_id": str(vectors.source_image_representation_id),
                "target_representation_id": str(vectors.source_image_representation_id),
                "coordinates": [[1, 1], [1, 2]],
                "capability": "frames.dataset.prepare.v1",
                "parameters": {"vector_representation_id": str(vectors.id)},
            },
            ServerContext(str(session.principal.id), "dataset", None),
        )
    assert all(projection.get_artifact_version(v.id) is not None for v in imported.versions)


def test_late_return_cannot_republish_removed_step(tmp_path):
    _server, local, session, project, layer, _vectors, _gateway, _submitted = _desktop_job(
        tmp_path, "frames.vectorize.v1"
    )
    job = local.plugin_jobs()[0]
    importer, command = _result(local, session, project, job)
    creation = next(event for event in local.history(project.id) if event.event_type == "PluginJobCreated")
    local.remove_layer_pipeline_action(
        principal=session.principal,
        project_id=project.id,
        layer_id=layer.id,
        action_event_id=creation.event_id,
    )
    with pytest.raises(ConflictError, match="removed"):
        importer(command)


def test_old_removal_receipt_also_blocks_dataset_inputs(tmp_path):
    from kraken_manager.domain.events import ActorSnapshot, EventEnvelope

    server, local, session, project, layer, vectors, _gateway, _submitted = _desktop_job(
        tmp_path, "frames.vectorize.v1"
    )
    job = local.plugin_jobs()[0]
    importer, command = _result(local, session, project, job)
    importer(command)
    # The previous application wrote only this event and deleted the folder.
    stream = f"layer-pipeline:{layer.id}"
    with local._uow(str(project.id))() as uow:
        revision = uow.event_store.current_revision(stream)
        event = EventEnvelope.create(
            stream_id=stream,
            project_id=project.id,
            revision=revision + 1,
            event_type="LayerPipelineActionRemoved",
            payload={"plugin_job_id": str(job.id), "layer_id": str(layer.id)},
            actor=ActorSnapshot.from_principal(session.principal),
        )
        uow.event_store.append(stream, expected_revision=revision, events=(event,))
        uow.commit()
    with pytest.raises(ServerConflictError, match="No published CIF"):
        server.submit_plugin_job(
            str(project.id),
            {
                "layer_id": str(layer.id),
                "source_representation_id": str(vectors.source_image_representation_id),
                "target_representation_id": str(vectors.source_image_representation_id),
                "coordinates": [[1, 1], [1, 2]],
                "capability": "frames.dataset.prepare.v1",
                "parameters": {"vector_representation_id": str(vectors.id)},
            },
            ServerContext(str(session.principal.id), "dataset-after-legacy-removal", None),
        )


def test_removal_preserves_newer_outputs_in_a_shared_legacy_representation(tmp_path):
    server, local, session, project, layer, vectors, _gateway, _submitted = _desktop_job(
        tmp_path, "frames.vectorize.v1"
    )
    old_job = local.plugin_jobs()[0]
    importer, command = _result(local, session, project, old_job)
    importer(command)
    creation = next(event for event in local.history(project.id) if event.event_type == "PluginJobCreated")
    submitted = server.submit_plugin_job(
        str(project.id),
        {
            "layer_id": str(layer.id),
            "source_representation_id": str(vectors.source_image_representation_id),
            "target_representation_id": str(vectors.id),
            "coordinates": [[1, 1], [1, 2]],
            "capability": "frames.vectorize.v1",
            "parameters": {},
        },
        ServerContext(str(session.principal.id), "new-vectors", None),
    )
    new_job = next(job for job in local.plugin_jobs() if str(job.id) == submitted["id"])
    importer, command = _result(local, session, project, new_job)
    newer = importer(command)
    local.remove_layer_pipeline_action(
        principal=session.principal,
        project_id=project.id,
        layer_id=layer.id,
        action_event_id=creation.event_id,
    )
    projection = local._projection(project.id)
    assert all(not projection.get_artifact_series(v.series_id).archived for v in newer.versions)
    assert {cell.artifact_version_id for cell in local.frame_cells(project.id, layer.id, vectors.id)} == {
        str(v.id) for v in newer.versions
    }
