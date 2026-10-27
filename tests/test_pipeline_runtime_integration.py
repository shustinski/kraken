"""Actual filesystem UOW/job handlers: durable handoff, cancellation and ACL.

The recording gateway tests queue delivery only; it does not pretend to run an
image algorithm or certify the full plugin pipeline.
"""

from dataclasses import replace
from uuid import uuid4

import pytest

from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.application.errors import AuthorizationError, ConflictError
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.pipeline import LayerPipelineDefinition, PipelineOperation
from tests.test_work_orders import _service, _project_with_frames


class Gateway:
    def __init__(self):
        self.manifests = {}
        self.deliveries = []
        self.cancelled = []
        self.fail_next = False

    def is_available(self, capability, protocol_version):
        return True

    def submit(self, manifest):
        self.deliveries.append(manifest.job_id)
        if self.fail_next:
            self.fail_next = False
            raise OSError("Agent disconnected after authoritative job commit")
        self.manifests[manifest.job_id] = manifest

    def cancel(self, job_id, **kwargs):
        self.cancelled.append(job_id)


def setup(tmp_path):
    local = _service(tmp_path)
    owner, _, project, layer, image, vector = _project_with_frames(local, tmp_path / "source")
    operation = PipelineOperation(
        id=str(uuid4()),
        title="Contour",
        capability="frames.vectorize.v1",
        algorithm_version="contour-test-build",
        source_representation_id=image.id,
        target_representation_id=vector.id,
        input_kind="image",
        output_kind="vector",
        coordinates=((1, 1), (2, 1)),
    )
    definition = LayerPipelineDefinition(str(project.id), str(layer.id), (operation,))
    state = local.define_layer_pipeline(
        project_id=project.id,
        layer_id=layer.id,
        definition=definition.to_dict(),
        principal=owner.principal,
        expected_revision=0,
        idempotency_key="pipeline-define",
    )
    state = local.command_layer_pipeline(
        project_id=project.id,
        layer_id=layer.id,
        operation="start",
        principal=owner.principal,
        expected_revision=state["revision"],
        idempotency_key="pipeline-start",
    )
    return local, owner.principal, project, layer, definition, state


def test_real_uow_repairs_commit_to_agent_handoff_after_restart(tmp_path):
    local, actor, project, layer, definition, _ = setup(tmp_path)
    gateway = Gateway()
    gateway.fail_next = True
    with pytest.raises(OSError, match="disconnected"):
        local.tick_layer_pipeline(project_id=project.id, layer_id=layer.id, principal=actor, gateway=gateway)
    state = local.get_layer_pipeline(project.id, layer.id)
    attempt = state["nodes"][definition.operations[0].id]["attempts"][-1]
    job_id = attempt["job_id"]
    assert local._projection(project.id).get_plugin_job(job_id).state.value == "queued"
    assert not gateway.manifests
    reopened = EmbeddedProjectService(local.data_dir, secret_store=local.secrets)
    state = reopened.tick_layer_pipeline(project_id=project.id, layer_id=layer.id, principal=actor, gateway=gateway)
    assert list(gateway.manifests) == [job_id]
    assert gateway.deliveries == [job_id, job_id]
    assert state["nodes"][definition.operations[0].id]["attempts"][-1]["dispatch_acknowledged"]
    manifest = gateway.manifests[job_id]
    assert len(manifest.inputs) == 2
    assert {str(i.artifact_version_id) for i in manifest.inputs} == set(attempt["request"]["input_heads"].values())
    local.tick_layer_pipeline(project_id=project.id, layer_id=layer.id, principal=actor, gateway=gateway)
    assert gateway.deliveries == [job_id, job_id]


def test_pipeline_cancel_uses_authoritative_plugin_job_cancellation(tmp_path):
    local, actor, project, layer, definition, state = setup(tmp_path)
    gateway = Gateway()
    state = local.tick_layer_pipeline(project_id=project.id, layer_id=layer.id, principal=actor, gateway=gateway)
    job_id = next(iter(gateway.manifests))
    state = local.command_layer_pipeline(
        project_id=project.id,
        layer_id=layer.id,
        operation="cancel",
        principal=actor,
        expected_revision=state["revision"],
        idempotency_key="cancel",
    )
    local.tick_layer_pipeline(project_id=project.id, layer_id=layer.id, principal=actor, gateway=gateway)
    state = local.tick_layer_pipeline(project_id=project.id, layer_id=layer.id, principal=actor, gateway=gateway)
    assert local._projection(project.id).get_plugin_job(job_id).state.value == "cancelled"
    assert state["status"] == "cancelled"


def test_unknown_actor_cannot_start_or_replace_pipeline_and_cross_project_rejected(tmp_path):
    local, actor, project, layer, definition, state = setup(tmp_path)
    outsider = Principal.local(subject="outsider", display_name="Outsider")
    with pytest.raises(AuthorizationError):
        local.command_layer_pipeline(
            project_id=project.id,
            layer_id=layer.id,
            operation="cancel",
            principal=outsider,
            expected_revision=state["revision"],
            idempotency_key="attack",
        )
    with pytest.raises(ConflictError):
        local.define_layer_pipeline(
            project_id=project.id,
            layer_id=layer.id,
            definition=replace(definition, project_id=str(uuid4())).to_dict(),
            principal=actor,
            expected_revision=state["revision"],
            idempotency_key="wrong-project",
        )


def test_quality_gate_blocks_exact_input_until_an_audited_exception(tmp_path):
    from kraken_manager.domain.artifacts import deterministic_frame_series_id

    local, actor, project, layer, definition, _ = setup(tmp_path)
    op = definition.operations[0]
    version = local._projection(project.id).get_active_artifact_version(
        deterministic_frame_series_id(op.source_representation_id, project.frame_id_at(1, 1))
    )
    issue = local.create_quality_issue(
        project_id=project.id,
        principal=actor,
        idempotency_key="quality",
        arguments=dict(
            title="Ambiguous conductor",
            description="Independent check required",
            severity="critical",
            scope={"layer_ids": [str(layer.id)]},
            source_version_ids=[str(version.id)],
        ),
    )
    gateway = Gateway()
    state = local.tick_layer_pipeline(project_id=project.id, layer_id=layer.id, principal=actor, gateway=gateway)
    assert state["nodes"][op.id]["status"] == "blocked"
    assert issue.id in state["nodes"][op.id]["error"]
    assert not gateway.manifests
    local.mutate_quality_issue(
        project_id=project.id,
        issue_id=issue.id,
        principal=actor,
        expected_revision=issue.revision,
        operation="exception",
        arguments={"note": "Authorized exploratory run; electrical acceptance remains unverified"},
        idempotency_key="exception",
    )
    state = local.command_layer_pipeline(
        project_id=project.id,
        layer_id=layer.id,
        operation="retry",
        node_id=op.id,
        principal=actor,
        expected_revision=state["revision"],
        idempotency_key="retry-after-exception",
    )
    state = local.tick_layer_pipeline(project_id=project.id, layer_id=layer.id, principal=actor, gateway=gateway)
    assert len(gateway.manifests) == 1


def test_managed_reimport_retains_old_version_and_replay_does_not_duplicate(tmp_path):
    from kraken_manager.domain.artifacts import deterministic_frame_series_id
    from kraken_manager.infrastructure.blob.filesystem import FilesystemBlobStore
    import hashlib

    local = _service(tmp_path)
    owner, _, project, layer, image, _ = _project_with_frames(local, tmp_path / "source")
    series_id = deterministic_frame_series_id(image.id, project.frame_id_at(1, 1))
    previous = local._projection(project.id).get_active_artifact_version(series_id)
    source = tmp_path / "source/images/1_1.png"
    source.write_bytes(b"new immutable image content")
    plan = local.plan_import_directory(project=project, directory=source.parent)
    args = dict(
        principal=owner.principal,
        project=local.get_project(project.id),
        layer=layer,
        representation=image,
        plan=plan,
        idempotency_key="second-import",
    )
    result = local.commit_managed_import(**args)
    repeated = local.commit_managed_import(**args)
    assert [version.id for version in result.versions] == [version.id for version in repeated.versions]
    current = local._projection(project.id).get_active_artifact_version(series_id)
    assert current.id != previous.id
    assert current.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    blobs = FilesystemBlobStore.for_project(local.catalog_root, project.id)
    assert blobs.path_for_read(previous.blob).read_bytes() == b"image-a"


def test_fingerprinted_import_rejects_source_change_before_committing_version(tmp_path):
    import hashlib
    from kraken_manager.application.imports import ImportPlanner, ImportMappingMode, ImportSource
    from kraken_manager.domain.artifacts import deterministic_frame_series_id
    from kraken_manager.infrastructure.blob.filesystem import BlobIntegrityError

    local = _service(tmp_path)
    owner, _, project, layer, image, _ = _project_with_frames(local, tmp_path / "source")
    path = tmp_path / "source/images/1_1.png"
    source = ImportSource(str(path), path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
    plan = ImportPlanner().plan(width=project.width, height=project.height, sources=(source,), mode=ImportMappingMode.XY_FILENAME)
    series_id = deterministic_frame_series_id(image.id, project.frame_id_at(1, 1))
    original = local._projection(project.id).get_active_artifact_version(series_id)
    path.write_bytes(b"changed after audited import planning")
    with pytest.raises(BlobIntegrityError, match="digest mismatch"):
        local.commit_managed_import(principal=owner.principal, project=local.get_project(project.id),
            layer=layer, representation=image, plan=plan, idempotency_key="fingerprinted")
    assert local._projection(project.id).get_active_artifact_version(series_id) == original
