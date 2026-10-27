"""Durable orchestration on the existing layer-pipeline stream and PluginJobs.

Dispatch intents are committed before calling the existing job submit handler.
An intent has a deterministic job ID/idempotency key and is replayed after a
crash; no second queue or execution engine is introduced.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from uuid import UUID, uuid4, uuid5

from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.errors import ConcurrencyError, ConflictError, NotFoundError
from kraken_manager.domain.common import DomainValidationError, as_utc, validate_uuid
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.pipeline import LayerPipelineDefinition


ACTIVE = {"dispatching", "running"}
TERMINAL_JOBS = {"succeeded", "failed", "cancelled"}


def deepcopy(value):
    """Thaw recursively frozen event JSON without copying mappingproxy objects."""
    if isinstance(value, Mapping):
        return {key: deepcopy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [deepcopy(item) for item in value]
    return value


class PipelineExecutionService:
    def __init__(self, events):
        self.events = events

    @staticmethod
    def stream(layer_id):
        return f"layer-pipeline:{validate_uuid(str(layer_id))}"

    def load(self, layer_id):
        events = self.events.load_stream(self.stream(layer_id))
        for event in reversed(events):
            if event.event_type == "LayerPipelineWorkflowV1":
                value = deepcopy(dict(event.payload["workflow"]))
                value["revision"] = events[-1].revision
                return value
        raise NotFoundError("Executable layer pipeline was not defined")

    def save(self, state, *, revision, principal, command, key=None, at):
        previous = self.events.find_by_idempotency_key(state["project_id"], key) if key else ()
        for event in previous:
            if (
                event.stream_id != self.stream(state["layer_id"])
                or deepcopy(event.payload.get("command")) != deepcopy(command)
                or event.actor.principal_id != principal.id
            ):
                raise ConflictError("Pipeline idempotency key belongs to a different command or actor")
            return deepcopy(event.payload["workflow"])
        state = deepcopy(state)
        state["revision"] = revision + 1
        state["updated_at"] = as_utc(at, field="pipeline time").isoformat()
        event = EventEnvelope.create(
            stream_id=self.stream(state["layer_id"]),
            project_id=state["project_id"],
            revision=revision + 1,
            event_type="LayerPipelineWorkflowV1",
            payload={"workflow": state, "command": command},
            actor=ActorSnapshot.from_principal(principal),
            recorded_at=at,
            idempotency_key=key,
        )
        self.events.append(event.stream_id, expected_revision=revision, events=(event,))
        return state

    @staticmethod
    def authorize(principal, storage, roles, *, verified=False):
        AuthorizationPolicy().require(
            principal=principal,
            storage=storage,
            roles=roles,
            permission=Permission.RUN_PLUGIN,
            gitlab_identity_verified=verified,
        )

    def define(self, definition, *, principal, expected_revision, key, at):
        command = {"operation": "define", "definition": definition.to_dict()}
        try:
            current = self.load(definition.layer_id)
        except NotFoundError:
            current = None
        prior = self.events.find_by_idempotency_key(definition.project_id, key)
        if not prior and current and current["status"] in {"running", "paused", "cancelling"}:
            raise ConflictError("Stop the current pipeline before changing its definition")
        if not prior and current and definition.version != current["definition"]["version"] + 1:
            raise ConflictError("A new pipeline definition must increment its version")
        current_revision = self.events.current_revision(self.stream(definition.layer_id))
        # save() checks command replay before EventStore's revision CAS.
        state = dict(
            project_id=definition.project_id,
            layer_id=definition.layer_id,
            definition=definition.to_dict(),
            definition_fingerprint=definition.fingerprint,
            status="draft",
            run_id=None,
            actor_id=str(principal.id),
            nodes={},
            error=None,
        )
        if expected_revision != current_revision and not self.events.find_by_idempotency_key(
            definition.project_id, key
        ):
            raise ConcurrencyError("Pipeline revision changed")
        return self.save(state, revision=expected_revision, principal=principal, command=command, key=key, at=at)

    def command(self, layer_id, operation, *, principal, expected_revision, key, at, node_id=None):
        state = self.load(layer_id)
        command = dict(operation=operation, node_id=node_id)
        prior = self.events.find_by_idempotency_key(state["project_id"], key)
        if prior:
            return self.save(state, revision=expected_revision, principal=principal, command=command, key=key, at=at)
        if state["revision"] != expected_revision:
            raise ConcurrencyError("Pipeline revision changed")
        definition = LayerPipelineDefinition.from_dict(state["definition"])
        if operation == "start":
            if state["status"] in {"running", "paused", "cancelling"}:
                raise ConflictError("Pipeline is already active")
            state.update(
                run_id=str(uuid4()),
                status="running",
                actor_id=str(principal.id),
                error=None,
                nodes={
                    op.id: dict(status="pending", approved=not op.approval_required, attempts=[], retry_at=None)
                    for op in definition.operations
                },
            )
        elif operation == "pause":
            if state["status"] != "running":
                raise ConflictError("Only a running pipeline can be paused")
            state["status"] = "paused"
        elif operation == "resume":
            if state["status"] != "paused":
                raise ConflictError("Only a paused pipeline can be resumed")
            state["status"] = "running"
        elif operation == "cancel":
            if state["status"] not in {"running", "paused", "failed"}:
                raise ConflictError("Pipeline is not active")
            state["status"] = "cancelling"
            for node in state["nodes"].values():
                if node["status"] not in ACTIVE:
                    node["status"] = "cancelled"
        elif operation == "approve":
            node = state["nodes"].get(node_id)
            if state["status"] not in {"running", "paused"} or node is None or node["status"] != "pending":
                raise ConflictError("Only a pending active operation can be approved")
            node["approved"] = True
        elif operation == "retry":
            if state["status"] not in {"failed", "paused", "running", "completed", "stale"}:
                raise ConflictError("Pipeline cannot be retried in this state")
            selected = state["nodes"].get(node_id)
            if selected is None or selected["status"] not in {"failed", "stale", "blocked"}:
                raise ConflictError("Select a failed or stale operation")
            affected = {node_id}
            while True:
                expanded = affected | {op.id for op in definition.operations if set(op.dependencies) & affected}
                if expanded == affected:
                    break
                affected = expanded
            if any(state["nodes"][item]["status"] in ACTIVE for item in affected):
                raise ConflictError("Wait for affected operations to stop before retry")
            for item in affected:
                node = state["nodes"][item]
                node.update(
                    status="pending",
                    retry_at=None,
                    approved=not next(op.approval_required for op in definition.operations if op.id == item),
                )
                # Manual retries start a new bounded automatic retry group.
                node["retry_group_start"] = len(node["attempts"])
            state["status"] = "running"
        else:
            raise DomainValidationError("Unknown pipeline command")
        return self.save(state, revision=expected_revision, principal=principal, command=command, key=key, at=at)

    def advance(self, layer_id, *, principal, at, observe, bind_inputs, current_inputs):
        """Reserve ready jobs and return intents; caller commits before dispatch.

        observe(job_id) -> None or {state, output_version_ids, error} from the
        authoritative PluginJob/results. bind_inputs(operation, producer_outputs)
        resolves immutable versions and returns a complete existing job request.
        current_inputs(request) detects replaced input heads without rebinding.
        """
        state = self.load(layer_id)
        if state["status"] == "draft" or not state["run_id"]:
            return state, ()
        before = deepcopy(state)
        definition = LayerPipelineDefinition.from_dict(state["definition"])
        policy = definition.policy
        at = as_utc(at, field="pipeline time")
        for op in definition.operations:
            node = state["nodes"][op.id]
            if not node["attempts"]:
                continue
            attempt = node["attempts"][-1]
            if node["status"] in ACTIVE:
                job = observe(attempt["job_id"])
                if job is not None:
                    attempt["job_state"] = job["state"]
                    if job["state"] in TERMINAL_JOBS:
                        attempt.update(finished_at=at.isoformat(), error=job.get("error"))
                        if job["state"] == "succeeded":
                            outputs = list(job.get("output_version_ids", ()))
                            if not outputs:
                                node["status"] = "failed"
                                attempt["error"] = "Successful job has no verified published outputs"
                            else:
                                node["status"] = "succeeded"
                                attempt["output_version_ids"] = outputs
                        else:
                            node["status"] = job["state"]
                    elif job["state"] != "queued" or attempt.get("dispatch_acknowledged"):
                        node["status"] = "running"
                # A dispatch intent must be reconciled with the authoritative
                # queue before cancellation becomes terminal: a crashed sender
                # may already have committed the job without recording its ack.
            if node["status"] == "failed" and state["status"] == "running":
                group_start = node.get("retry_group_start", 0)
                if len(node["attempts"]) - group_start < policy.max_attempts:
                    node["status"] = "pending"
                    node["retry_at"] = (at + timedelta(seconds=policy.retry_delay_seconds)).isoformat()
            if node["status"] == "succeeded" and not current_inputs(attempt["request"]):
                node["status"] = "stale"
        # Propagate invalidation transitively, including already produced results.
        stale = {key for key, node in state["nodes"].items() if node["status"] == "stale"}
        while True:
            expanded = stale | {op.id for op in definition.operations if set(op.dependencies) & stale}
            if expanded == stale:
                break
            stale = expanded
        for key in stale:
            node = state["nodes"][key]
            if node["status"] not in ACTIVE:
                node["status"] = "stale"
        if stale and state["status"] == "completed":
            state["status"] = "running" if policy.recompute_stale else "stale"
        if (
            policy.recompute_stale
            and state["status"] == "running"
            and not any(state["nodes"][key]["status"] in ACTIVE for key in stale)
        ):
            for key in stale:
                op = next(op for op in definition.operations if op.id == key)
                state["nodes"][key].update(status="pending", approved=not op.approval_required)
        intents = []
        if state["status"] == "running":
            active = [op for op in definition.operations if state["nodes"][op.id]["status"] in ACTIVE]
            used = {key: sum(getattr(op.resources, key) for op in active) for key in ("cpu", "gpu", "memory_mb")}
            for op in definition.operations:
                node = state["nodes"][op.id]
                if node["status"] != "pending" or not node["approved"]:
                    continue
                dependencies = [state["nodes"][key]["status"] for key in op.dependencies]
                if any(status in {"failed", "cancelled", "blocked"} for status in dependencies):
                    node["status"] = "blocked"
                    continue
                if any(status != "succeeded" for status in dependencies):
                    continue
                if node["retry_at"] and datetime.fromisoformat(node["retry_at"]) > at:
                    continue
                if len(active) >= policy.max_parallel or any(
                    used[key] + getattr(op.resources, key) > getattr(policy.capacity, key) for key in used
                ):
                    continue
                producer_outputs = (
                    ()
                    if not op.input_operation_id
                    else state["nodes"][op.input_operation_id]["attempts"][-1]["output_version_ids"]
                )
                try:
                    if op.model_input is not None:
                        model_outputs = (() if op.model_input.producer_id is None else
                                         state["nodes"][op.model_input.producer_id]["attempts"][-1]["output_version_ids"])
                        request = bind_inputs(op, producer_outputs, model_outputs)
                    else:
                        request = bind_inputs(op, producer_outputs)
                except (DomainValidationError, ConflictError, NotFoundError) as exc:
                    node.update(status="blocked", error=str(exc))
                    continue
                job_id = str(uuid5(UUID(state["run_id"]), f"{op.id}:{len(node['attempts']) + 1}"))
                request = deepcopy(request)
                request["job_id"] = job_id
                request.setdefault("parameters", {}).update(
                    pipeline_attempt_id=job_id, pipeline_algorithm_version=op.algorithm_version,
                    execution_resources=op.resources.to_dict(),
                )
                if request.get("kind") == "layer":
                    request["parameters"]["_pipeline"] = {
                        "run_id": state["run_id"], "node_id": op.id, "layer_id": layer_id,
                        "definition_fingerprint": state["definition_fingerprint"],
                        "algorithm_version": op.algorithm_version, "quality_gate": op.quality_gate,
                        "resources": op.resources.to_dict(),
                    }
                attempt = dict(
                    job_id=job_id,
                    idempotency_key=f"pipeline:{job_id}",
                    request=request,
                    created_at=at.isoformat(),
                    job_state=None,
                    output_version_ids=[],
                )
                node["attempts"].append(attempt)
                node["status"] = "dispatching"
                active.append(op)
                for key in used:
                    used[key] += getattr(op.resources, key)
        if state["status"] == "running":
            statuses = {node["status"] for node in state["nodes"].values()}
            if statuses == {"succeeded"}:
                state["status"] = "completed"
            elif not statuses & (ACTIVE | {"pending", "stale"}):
                state["status"] = "failed"
        elif state["status"] == "cancelling" and not any(node["status"] in ACTIVE for node in state["nodes"].values()):
            state["status"] = "cancelled"
        for node in state["nodes"].values():
            if node["status"] == "dispatching" and state["status"] in {"running", "cancelling"}:
                intents.append(deepcopy(node["attempts"][-1]))
        if state != before:
            state = self.save(
                state, revision=before["revision"], principal=principal, command={"operation": "advance"}, at=at
            )
        return state, tuple(intents)

    def dispatch_failed(self, layer_id, job_id, error, *, principal, at):
        state = self.load(layer_id)
        revision = state["revision"]
        for node in state["nodes"].values():
            if node["status"] == "dispatching" and node["attempts"][-1]["job_id"] == job_id:
                node["status"] = "failed"
                node["attempts"][-1].update(error=str(error), finished_at=at.isoformat())
                return self.save(
                    state,
                    revision=revision,
                    principal=principal,
                    command={"operation": "dispatch_failed", "job_id": job_id},
                    at=at,
                )
        return state

    def dispatched(self, layer_id, job_id, *, principal, at):
        state = self.load(layer_id)
        if state["status"] != "running":
            return state
        revision = state["revision"]
        for node in state["nodes"].values():
            if (
                node["status"] in ACTIVE
                and node["attempts"][-1]["job_id"] == job_id
                and not node["attempts"][-1].get("dispatch_acknowledged")
            ):
                node["attempts"][-1]["dispatch_acknowledged"] = True
                node["status"] = "running"
                return self.save(
                    state,
                    revision=revision,
                    principal=principal,
                    command={"operation": "dispatched", "job_id": job_id},
                    at=at,
                )
        return state
