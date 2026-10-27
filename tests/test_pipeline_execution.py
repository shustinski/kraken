from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from kraken_manager.application.errors import ConflictError
from kraken_manager.application.pipeline_execution import PipelineExecutionService
from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.pipeline import (
    LayerPipelineDefinition,
    PipelineOperation,
    PipelinePolicy,
    PipelineResources,
)
from kraken_manager.infrastructure.filesystem.event_store import FilesystemEventStore


NOW = datetime(2026, 10, 24, tzinfo=UTC)


def uid():
    return str(uuid4())


def operation(**kwargs):
    return PipelineOperation(
        **(
            dict(
                id=uid(),
                title="Vectorize",
                capability="frames.vectorize.v1",
                algorithm_version="contour-build-1",
                source_representation_id=uid(),
                target_representation_id=uid(),
                input_kind="image",
                output_kind="vector",
                coordinates=((1, 1),),
            )
            | kwargs
        )
    )


def setup(tmp_path, operations=None, **policy):
    ops = operations or (operation(),)
    definition = LayerPipelineDefinition(uid(), uid(), ops, PipelinePolicy(**policy))
    service = PipelineExecutionService(FilesystemEventStore(tmp_path, definition.project_id))
    principal = Principal.local(subject="engineer", display_name="Engineer")
    state = service.define(definition, principal=principal, expected_revision=0, key="define", at=NOW)
    state = service.command(
        definition.layer_id, "start", principal=principal, expected_revision=state["revision"], key="start", at=NOW
    )
    return service, definition, principal, state


def advance(service, definition, principal, jobs=None, *, at=NOW, current=True):
    return service.advance(
        definition.layer_id,
        principal=principal,
        at=at,
        observe=lambda job_id: (jobs or {}).get(job_id),
        bind_inputs=lambda op, outputs: {
            "parameters": {},
            "input_heads": {"source": "v1"},
            "producer_outputs": outputs,
        },
        current_inputs=lambda request: current,
    )


def test_ports_cycles_and_scope_are_checked():
    a = operation()
    b = operation(
        source_representation_id=a.target_representation_id,
        input_operation_id=a.id,
        dependencies=(a.id,),
        input_kind="binary_mask",
    )
    with pytest.raises(DomainValidationError, match="incompatible"):
        LayerPipelineDefinition(uid(), uid(), (a, b))
    a, b = operation(), operation()
    with pytest.raises(DomainValidationError, match="cycle"):
        LayerPipelineDefinition(uid(), uid(), (replace(a, dependencies=(b.id,)), replace(b, dependencies=(a.id,))))
    with pytest.raises(DomainValidationError, match="ports"):
        operation(output_kind="model")
    with pytest.raises(DomainValidationError, match="capacity"):
        LayerPipelineDefinition(uid(), uid(), (replace(a, resources=PipelineResources(gpu=1)),))


def test_dispatch_intent_survives_restart_and_keeps_exact_job_and_inputs(tmp_path):
    service, definition, principal, state = setup(tmp_path)
    state, intents = advance(service, definition, principal)
    assert len(intents) == 1
    assert intents[0]["request"]["parameters"]["pipeline_algorithm_version"] == "contour-build-1"
    reopened = PipelineExecutionService(FilesystemEventStore(tmp_path, definition.project_id))
    again, repeated = advance(reopened, definition, principal)
    assert repeated == intents
    assert again["revision"] == state["revision"]
    assert len(again["nodes"][definition.operations[0].id]["attempts"]) == 1


def test_independent_branches_reserve_resources_and_only_ready_dependency_runs(tmp_path):
    a, b, c = operation(), operation(), operation()
    c = replace(c, dependencies=(a.id, b.id))
    service, definition, principal, _ = setup(
        tmp_path, (a, b, c), max_parallel=3, capacity=PipelineResources(cpu=2, memory_mb=512)
    )
    state, intents = advance(service, definition, principal)
    assert len(intents) == 2
    jobs = {i["job_id"]: dict(state="succeeded", output_version_ids=[uid()]) for i in intents}
    state, intents = advance(service, definition, principal, jobs)
    assert len(intents) == 1
    assert state["nodes"][c.id]["status"] == "dispatching"


def test_success_without_published_artifacts_does_not_unlock_dependents(tmp_path):
    a, b = operation(), operation()
    b = replace(b, dependencies=(a.id,))
    service, definition, principal, _ = setup(tmp_path, (a, b))
    _, intents = advance(service, definition, principal)
    state, intents = advance(service, definition, principal, {intents[0]["job_id"]: {"state": "succeeded"}})
    assert state["nodes"][a.id]["status"] == "failed"
    assert state["nodes"][b.id]["status"] == "blocked"
    assert not intents


def test_bounded_retry_waits_and_uses_new_job_with_history(tmp_path):
    service, definition, principal, _ = setup(tmp_path, max_attempts=2, retry_delay_seconds=5)
    _, intents = advance(service, definition, principal)
    first = intents[0]["job_id"]
    state, intents = advance(service, definition, principal, {first: {"state": "failed", "error": "GPU OOM"}})
    assert not intents
    state, intents = advance(service, definition, principal, at=NOW + timedelta(seconds=6))
    second = intents[0]["job_id"]
    assert second != first
    state, intents = advance(
        service, definition, principal, {second: {"state": "failed"}}, at=NOW + timedelta(seconds=7)
    )
    assert state["status"] == "failed" and not intents
    assert len(state["nodes"][definition.operations[0].id]["attempts"]) == 2


def test_manual_approval_and_pause_prevent_new_dispatch(tmp_path):
    op = operation(approval_required=True)
    service, definition, principal, state = setup(tmp_path, (op,))
    state, intents = advance(service, definition, principal)
    assert not intents
    state = service.command(
        definition.layer_id,
        "approve",
        node_id=op.id,
        principal=principal,
        expected_revision=state["revision"],
        key="approve",
        at=NOW,
    )
    state = service.command(
        definition.layer_id, "pause", principal=principal, expected_revision=state["revision"], key="pause", at=NOW
    )
    _, intents = advance(service, definition, principal)
    assert not intents
    state = service.command(
        definition.layer_id, "resume", principal=principal, expected_revision=state["revision"], key="resume", at=NOW
    )
    _, intents = advance(service, definition, principal)
    assert len(intents) == 1


def test_stale_inputs_are_visible_and_do_not_recompute_without_policy(tmp_path):
    service, definition, principal, _ = setup(tmp_path)
    _, intents = advance(service, definition, principal)
    jobs = {intents[0]["job_id"]: {"state": "succeeded", "output_version_ids": [uid()]}}
    state, _ = advance(service, definition, principal, jobs)
    assert state["status"] == "completed"
    state, intents = advance(service, definition, principal, current=False)
    assert state["nodes"][definition.operations[0].id]["status"] == "stale"
    assert not intents


def test_cancel_reconciles_dispatch_before_becoming_terminal(tmp_path):
    service, definition, principal, _ = setup(tmp_path)
    state, intents = advance(service, definition, principal)
    state = service.command(
        definition.layer_id, "cancel", principal=principal, expected_revision=state["revision"], key="cancel", at=NOW
    )
    state, repeated = advance(service, definition, principal)
    assert state["status"] == "cancelling" and repeated == intents
    state, intents = advance(service, definition, principal, {repeated[0]["job_id"]: {"state": "cancelled"}})
    assert state["status"] == "cancelled" and not intents


def test_idempotency_is_bound_to_command_and_actor(tmp_path):
    service, definition, principal, state = setup(tmp_path)
    replay = service.command(
        definition.layer_id, "start", principal=principal, expected_revision=1, key="start", at=NOW
    )
    assert replay == state
    with pytest.raises(ConflictError):
        service.command(
            definition.layer_id, "cancel", principal=principal, expected_revision=state["revision"], key="start", at=NOW
        )
    with pytest.raises(ConflictError):
        service.command(
            definition.layer_id,
            "start",
            principal=Principal.local(subject="other", display_name="Other"),
            expected_revision=1,
            key="start",
            at=NOW,
        )
