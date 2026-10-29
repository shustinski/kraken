"""test_server_workspace_inputs: test contour launch creates."""
from __future__ import annotations
from types import SimpleNamespace
import pytest
from kraken_hub.presentation.qt.manager_app import DesktopController
from kraken_manager.domain.project import RepresentationKind

from test_server_workspace_inputs_parts.server_layer import _server_layer

@pytest.mark.parametrize("capability", ["frames.vectorize.v1", "frames.binary-segment.v2", "frames.confidence.v1"])
@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("existing_target", ["missing", "linked", "unrelated"])
def test_contour_launch_creates_or_reuses_empty_output(tmp_path, monkeypatch, remote, existing_target, capability):
    from kraken_hub.presentation.qt import manager_app
    from kraken_manager.domain.project import RepresentationPurpose

    output_kind = RepresentationKind.VECTOR if capability == "frames.vectorize.v1" else RepresentationKind.IMAGE
    output_purpose = (
        RepresentationPurpose.VECTOR if capability == "frames.vectorize.v1"
        else RepresentationPurpose.CONFIDENCE if capability == "frames.confidence.v1" else RepresentationPurpose.BINARY
    )

    server, local, session, project, layer, _binding, source, context = _server_layer(tmp_path)
    server.materialize_representation_inputs(str(project.id), str(layer.id), str(source.id), context)
    target = None
    if existing_target != "missing":
        target_source = source
        if existing_target == "unrelated":
            target_source = local.create_representation(
                principal=session.principal,
                project=project,
                layer=layer,
                name="Other images",
                kind=RepresentationKind.IMAGE,
                idempotency_key="other-source",
            )
        target = local.create_representation(
            principal=session.principal,
            project=project,
            layer=layer,
            name="Existing CIF",
            kind=output_kind,
            purpose=output_purpose,
            source_image_representation_id=target_source.id,
            idempotency_key="existing-output",
        )
    manifests = []
    gateway = SimpleNamespace(is_available=lambda *_args: True, submit=manifests.append)
    monkeypatch.setattr(manager_app, "AgentPluginGateway", lambda **_kwargs: gateway)
    local.is_remote_project = lambda _id: remote
    submit = local.submit_plugin_job
    local.submit_plugin_job = lambda **kwargs: submit(**{**kwargs, "gateway": gateway})
    controller = object.__new__(DesktopController)
    controller.service = local
    controller.session = session
    controller._project_id = str(project.id)
    controller._workspace = SimpleNamespace(
        matrix_view=SimpleNamespace(selected_coordinates=lambda **_kwargs: ()),
        vector_representation_combo=SimpleNamespace(currentData=lambda: str(target.id) if target is not None else None),
    )
    controller.agent_runtime = SimpleNamespace(
        interactive_registry=lambda _operation: "interactive-registry",
        execution_registry=lambda _operation: "execution-registry",
        ensure_started=lambda: frozenset({capability}),
        base_url="http://localhost:9999",
        token="test",
        protocol_by_capability={},
    )
    launched = []
    controller._run_remote_interactive_job = lambda job, **kwargs: launched.append((job, kwargs))
    job = controller._submit_agent_action(
        layer_id=str(layer.id),
        source_representation_id=str(source.id),
        capability=capability,
        parameters={},
    )
    vectors = [
        item
        for item in local.list_representations(project.id, layer.id)
        if item.kind is output_kind
        and item.purpose is output_purpose
        and item.source_image_representation_id == source.id
    ]
    assert len(vectors) == 1
    assert vectors[0].source_image_representation_id == source.id
    assert job.target_representation_id == vectors[0].id
    if existing_target == "linked":
        assert vectors[0].id == target.id
    assert len(manifests) == 1
    assert len(manifests[0].inputs) == 2
    assert local.frame_cells(project.id, layer.id, vectors[0].id) == ()
    retry = controller._submit_agent_action(
        layer_id=str(layer.id),
        source_representation_id=str(source.id),
        capability=capability,
        parameters={},
    )
    assert retry.target_representation_id == job.target_representation_id
    assert len(manifests) == 2
    assert len(launched) == (2 if remote else 0)
    if remote:
        assert manifests[0].parameters["execution_host"] == "desktop"
