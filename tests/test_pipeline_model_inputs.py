"""Functional multi-input binding without training or measuring inference quality."""
from dataclasses import replace
import hashlib
from uuid import uuid4

import pytest

from kraken_manager.application.dto import AddArtifactVersionCommand, CommandContext, SubmitPluginJobCommand, ImportPluginResultCommand
from kraken_manager.application.errors import ConflictError, AuthorizationError
from kraken_manager.application.model_inputs import resolve_model_input
from kraken_manager.application.plugin_jobs import SubmitPluginJobHandler, ImportPluginResultHandler
from kraken_manager.application.use_cases import AddArtifactVersionHandler
from kraken_manager.domain.artifacts import ArtifactScope
from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.pipeline import LayerPipelineDefinition, PipelineOperation, PipelineModelInput
from kraken_manager.domain.project import RepresentationKind, RepresentationPurpose
from kraken_manager.domain.workflows import PluginInputV1, PluginResultManifestV1, PluginFrameResultV1
from kraken_manager.domain.selection import FrameSelectionV1
from kraken_manager.domain.identity import Principal
from kraken_manager.infrastructure.pipeline_bindings import PipelineBindings
from tests.test_pipeline_runtime_integration import Gateway
from tests.test_work_orders import _service, _project_with_frames


@pytest.fixture
def catalog(tmp_path):
    service = _service(tmp_path)
    owner, _, project, layer, images, _ = _project_with_frames(service, tmp_path / "source")
    target = service.create_representation(principal=owner.principal, project=service.get_project(project.id),
        layer=service.list_layers(project.id)[0], name="Mask", kind=RepresentationKind.IMAGE,
        purpose=RepresentationPurpose.BINARY, active=True, idempotency_key="mask")
    series = service.create_artifact_series(principal=owner.principal, project_id=project.id,
        layer_id=layer.id, scope=ArtifactScope.LAYER_ATTACHMENT, name="Model", idempotency_key="model-series")
    model = AddArtifactVersionHandler(service._uow(str(project.id)), service.profiles, service.clock)(
        AddArtifactVersionCommand(context=CommandContext(actor=owner.principal, idempotency_key="model-version"),
            project_id=project.id, series_id=series.id, filename="published.pth", media_type="application/octet-stream",
            expected_series_revision=service.artifact_stream_revision(project.id, series.id),
            parameters={"artifact_role": "model"}), (b"opaque test fixture, no training",))
    operation = PipelineOperation(id=str(uuid4()), title="Infer", capability="frames.binary-segment.v2",
        algorithm_version="test-build", source_representation_id=images.id, target_representation_id=target.id,
        input_kind="image", output_kind="binary_mask", coordinates=((1, 1), (2, 1)),
        model_input=PipelineModelInput(version_id=model.id))
    return service, owner.principal, project, layer, operation, model


def test_model_port_roundtrip_and_dependency_type_validation(catalog):
    _, _, project, layer, op, _ = catalog
    definition = LayerPipelineDefinition(project.id, layer.id, (op,))
    assert LayerPipelineDefinition.from_dict(definition.to_dict()) == definition
    with pytest.raises(DomainValidationError, match="exactly one"):
        PipelineModelInput()
    with pytest.raises(DomainValidationError, match="dependency"):
        replace(op, model_input=PipelineModelInput(producer_id=str(uuid4())))
    producer = replace(op, id=str(uuid4()), target_representation_id=str(uuid4()), model_input=None)
    invalid = replace(op, dependencies=(producer.id,), model_input=PipelineModelInput(producer_id=producer.id))
    with pytest.raises(DomainValidationError, match="publish a model"):
        LayerPipelineDefinition(project.id, layer.id, (producer, invalid))


def test_binding_pins_auxiliary_model_without_duplicate_frame_inputs(catalog):
    service, actor, project, layer, op, model = catalog
    with service._uow(str(project.id))() as unit:
        bindings = PipelineBindings(unit.projections, unit.event_store, project, layer.id)
        bindings.validate(LayerPipelineDefinition(project.id, layer.id, (op,)))
        request = bindings.bind(op, ())
        assert len(request["inputs"]) == 2
        assert model.id not in {item["artifact_version_id"] for item in request["inputs"]}
        assert request["input_heads"][model.series_id] == model.id
        assert request["parameters"]["model_sha256"] == model.sha256
        assert request["parameters"]["model_filename"] == "published.pth"
        assert bindings.current(request)
        producer_op = replace(op, dependencies=(str(uuid4()),))
        producer_op = replace(producer_op, model_input=PipelineModelInput(producer_id=producer_op.dependencies[0]))
        assert bindings.bind(producer_op, (), (model.id,))["parameters"]["model_version_id"] == model.id
        with pytest.raises(ConflictError, match="exactly one"):
            bindings.bind(producer_op, (), ())


@pytest.mark.parametrize("change", [dict(model_sha256="f" * 64), dict(model_filename="elsewhere.pth"),
                                     dict(model_source_path="C:/untrusted.pth"), dict(model_relative_path="../model.pth")])
def test_managed_model_metadata_cannot_be_spoofed(catalog, change):
    service, _, project, layer, _, model = catalog
    with service._uow(str(project.id))() as unit:
        with pytest.raises(ConflictError):
            resolve_model_input(unit.projections, project.id, layer.id, {"model_version_id": model.id, **change})
        with pytest.raises(ConflictError):
            resolve_model_input(unit.projections, str(uuid4()), layer.id, {"model_version_id": model.id})


def test_real_submit_import_preserves_image_and_model_lineage(catalog):
    service, actor, project, layer, op, model = catalog
    with service._uow(str(project.id))() as unit:
        request = PipelineBindings(unit.projections, unit.event_store, project, layer.id).bind(op, ())
    gateway = Gateway()
    job = SubmitPluginJobHandler(service._uow(str(project.id)), service.profiles, service.clock, gateway)(
        SubmitPluginJobCommand(context=CommandContext(actor=actor, idempotency_key="infer"),
            project_id=project.id, layer_id=layer.id, target_representation_id=op.target_representation_id,
            selection=FrameSelectionV1.rectangle(1, 1, 2, 1), capability=op.capability,
            inputs=tuple(PluginInputV1(**item) for item in request["inputs"]), parameters=request["parameters"]))
    payload = b"opaque PNG bytes: functional importer test"
    result = PluginResultManifestV1(job_id=job.id, plugin_name="NeuralImage", plugin_version="test-build", parameters_applied={},
        results=tuple(PluginFrameResultV1(output_id=str(uuid4()), frame_id=item.frame_id, outcome="succeeded",
            relative_path=f"outputs/{index}.png", sha256=hashlib.sha256(payload).hexdigest(),
            media_type="image/png", role="binary-image") for index, item in enumerate(gateway.manifests[job.id].inputs)))
    class Reader:
        def iter_output(self, *_):
            yield payload
    imported = ImportPluginResultHandler(service._uow(str(project.id)), service.profiles, service.clock, Reader())(
        ImportPluginResultCommand(context=CommandContext(actor=actor, idempotency_key="import"), manifest=result))
    image_ids = {item["artifact_version_id"] for item in request["inputs"]}
    assert len(imported.versions) == 2
    for version in imported.versions:
        assert model.id in version.input_version_ids
        assert len(image_ids.intersection(version.input_version_ids)) == 1


def test_model_reference_is_checked_only_after_project_authorization(catalog):
    service, _, project, layer, op, _ = catalog
    outsider = Principal.local(subject="outsider", display_name="Outsider")
    with service._uow(str(project.id))() as unit:
        request = PipelineBindings(unit.projections, unit.event_store, project, layer.id).bind(op, ())
    with pytest.raises(AuthorizationError):
        SubmitPluginJobHandler(service._uow(str(project.id)), service.profiles, service.clock, Gateway())(
            SubmitPluginJobCommand(context=CommandContext(actor=outsider, idempotency_key="unauthorized"),
                project_id=project.id, layer_id=layer.id, target_representation_id=op.target_representation_id,
                selection=FrameSelectionV1.rectangle(1, 1, 2, 1), capability=op.capability,
                inputs=tuple(PluginInputV1(**item) for item in request["inputs"]),
                parameters={"model_version_id": "invalid-but-not-disclosed"}))


def test_scheduler_binds_training_output_to_secondary_model_port(tmp_path):
    from tests.test_pipeline_execution import setup, operation, NOW
    training = operation(capability="dataset.model.train.v1", input_kind="dataset", output_kind="model")
    inference = operation(capability="frames.binary-segment.v2", input_kind="image", output_kind="binary_mask",
        dependencies=(training.id,), model_input=PipelineModelInput(producer_id=training.id))
    service, definition, actor, _ = setup(tmp_path, (training, inference))
    calls = []
    def bind(op, primary, secondary=()):
        calls.append((op.id, tuple(primary), tuple(secondary)))
        return {"parameters": {}, "input_heads": {}}
    state, intents = service.advance(definition.layer_id, principal=actor, at=NOW,
        observe=lambda _: None, bind_inputs=bind, current_inputs=lambda _: True)
    assert len(intents) == 1 and calls == [(training.id, (), ())]
    training_job = intents[0]["job_id"]
    service.dispatched(definition.layer_id, training_job, principal=actor, at=NOW)
    model_version = str(uuid4())
    state, intents = service.advance(definition.layer_id, principal=actor, at=NOW,
        observe=lambda _: {"state": "succeeded", "output_version_ids": [model_version]},
        bind_inputs=bind, current_inputs=lambda _: True)
    assert calls[-1] == (inference.id, (), (model_version,))
    assert intents[0]["request"]["parameters"]["pipeline_algorithm_version"] == inference.algorithm_version
