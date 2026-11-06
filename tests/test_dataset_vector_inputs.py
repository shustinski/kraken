"""Dataset preparation pins the selected CIF representation, never session files."""

from pathlib import Path

import pytest
import sqlalchemy as sa
from types import SimpleNamespace

from kraken_server.services import CommandContext
from test_server_workspace_inputs_parts.desktop_job import _desktop_job


@pytest.fixture
def vector_job(tmp_path):
    server, local, session, project, layer, vectors, gateway, job = _desktop_job(tmp_path, "frames.vectorize.v1")
    source = tmp_path / "new-vectors"
    source.mkdir()
    for name in ("1_1.cif", "1_2.cif"):
        (source / name).write_text("L M1; P 0 0 10 0 10 10 0 10; E;\n", encoding="utf-8")
    imported = local.commit_managed_import(
        principal=session.principal,
        project=project,
        layer=layer,
        representation=vectors,
        plan=local.plan_import_directory(project=project, directory=source),
        idempotency_key="new-vectors",
    )
    return server, local, session, project, layer, vectors, gateway, imported


def test_dataset_job_pins_current_cif_versions_from_selected_representation(vector_job):
    server, _local, session, project, layer, vectors, gateway, imported = vector_job
    created = server.submit_plugin_job(
        str(project.id),
        {
            "layer_id": str(layer.id),
            "source_representation_id": str(vectors.source_image_representation_id),
            "target_representation_id": str(vectors.source_image_representation_id),
            "coordinates": [[1, 1], [1, 2]],
            "capability": "frames.dataset.prepare.v1",
            "parameters": {"vector_representation_id": str(vectors.id), "input_transport": "server"},
        },
        CommandContext(str(session.principal.id), "dataset-with-vectors", None),
    )
    with gateway.engine.connect() as connection:
        manifest = connection.execute(
            sa.select(gateway.jobs.c.payload).where(
                gateway.jobs.c.job_id == created["id"],
            )
        ).scalar_one()
    pinned = manifest["parameters"]["vector_inputs"]
    assert {item["artifact_version_id"] for item in pinned} == {str(version.id) for version in imported.versions}
    assert {item["sha256"] for item in pinned} == {version.sha256 for version in imported.versions}
    assert {Path(item["relative_path"]).name for item in pinned} == {"0000.cif", "0002.cif"}


def test_dataset_rejects_image_representation_as_vector_input(vector_job):
    server, _local, session, project, layer, vectors, _gateway, _imported = vector_job
    with pytest.raises((ValueError, RuntimeError), match="[Vv]ector|CIF"):
        server.submit_plugin_job(
            str(project.id),
            {
                "layer_id": str(layer.id),
                "source_representation_id": str(vectors.source_image_representation_id),
                "target_representation_id": str(vectors.source_image_representation_id),
                "coordinates": [[1, 1]],
                "capability": "frames.dataset.prepare.v1",
                "parameters": {"vector_representation_id": str(vectors.source_image_representation_id)},
            },
            CommandContext(str(session.principal.id), "bad-vector-input", None),
        )


def test_controller_passes_selected_vector_representation(vector_job, monkeypatch, tmp_path):
    from kraken_hub.presentation.qt import manager_app

    _server, local, session, project, layer, vectors, _gateway, _imported = vector_job
    manifests = []
    gateway = SimpleNamespace(is_available=lambda *_args: True, submit=manifests.append)
    monkeypatch.setattr(manager_app, "AgentPluginGateway", lambda **_kwargs: gateway)
    controller = object.__new__(manager_app.DesktopController)
    controller.service, controller.session, controller._project_id = local, session, str(project.id)
    controller._workspace = SimpleNamespace(
        matrix_view=SimpleNamespace(selected_coordinates=lambda **_kwargs: ()),
        vector_representation_combo=SimpleNamespace(currentData=lambda: str(vectors.id)),
    )
    controller.agent_runtime = SimpleNamespace(
        ensure_started=lambda: frozenset({"frames.dataset.prepare.v1"}),
        base_url="http://unused",
        token="test",
        protocol_by_capability={},
    )
    controller._submit_agent_action(
        layer_id=str(layer.id),
        source_representation_id=str(vectors.source_image_representation_id),
        capability="frames.dataset.prepare.v1",
        parameters={},
    )
    assert manifests[0].parameters["vector_representation_id"] == str(vectors.id)
    assert len(manifests[0].parameters["vector_inputs"]) == 2

    from kraken_manager.infrastructure.plugin.agent_gateway import AgentPluginGateway
    import hashlib

    manifest = manifests[0]
    coordinates = {str(item.frame_id(project.id)): (item.x, item.y) for item in manifest.selection.iter_coordinates()}
    staging = tmp_path / "agent-staging"
    actual_gateway = AgentPluginGateway(
        base_url="http://unused",
        token="test",
        staging_root=staging,
        source_for_version=lambda identifier: local.managed_artifact_path(project.id, identifier),
        coordinate_for_frame=lambda identifier: coordinates[str(identifier)],
        media_type_for_version=lambda identifier: local.artifact_version(project.id, identifier).media_type,
        capabilities=frozenset({"frames.dataset.prepare.v1"}),
    )
    monkeypatch.setattr(actual_gateway, "_request", lambda *_args: {"status": "ok", "api_version": "v1"})
    actual_gateway.submit(manifest)
    for vector in manifest.parameters["vector_inputs"]:
        path = staging / str(manifest.job_id) / vector["relative_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == vector["sha256"]
