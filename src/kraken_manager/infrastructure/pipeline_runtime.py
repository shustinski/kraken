"""Transactional layer-pipeline facade shared by local and server composition."""

from __future__ import annotations

from kraken_manager.application.dto import CommandContext, SubmitPluginJobCommand, CancelPluginJobCommand
from kraken_manager.application.errors import ConflictError, NotFoundError
from kraken_manager.application.pipeline_execution import PipelineExecutionService
from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.pipeline import LayerPipelineDefinition
from kraken_manager.domain.project import ProjectState
from kraken_manager.domain.selection import FrameRowRange, FrameSelectionV1
from kraken_manager.domain.workflows import PluginInputV1
from .pipeline_bindings import PipelineBindings
from .pipeline_stitch import ensure_stitch_intent, cancel_stitch_intent


class PipelineRuntime:
    def __init__(self, uow_factory, profiles, clock, submit, cancel, *, verified=False,
                 stitch_dispatch=None, stitch_refresh=None, stitch_cancel=None):
        self.uow_factory, self.profiles, self.clock = uow_factory, profiles, clock
        self.submit, self.cancel, self.verified = submit, cancel, verified
        self.stitch_dispatch, self.stitch_refresh, self.stitch_cancel = stitch_dispatch, stitch_refresh, stitch_cancel

    def context(self, unit, project_id, layer_id, principal=None, *, mutate=False):
        project = unit.projections.get_project(project_id)
        if project is None:
            raise NotFoundError("Pipeline project was not found")
        if mutate:
            if project.state is not ProjectState.ACTIVE:
                raise ConflictError("Archived projects are read-only")
            PipelineExecutionService.authorize(
                principal,
                self.profiles.get(project.storage_profile),
                unit.acl.roles_for(project.id, principal.id),
                verified=self.verified,
            )
        layer = unit.projections.get_layer(layer_id)
        if layer is None or str(layer.project_id) != str(project.id):
            raise NotFoundError("Pipeline layer was not found in the project")
        return PipelineExecutionService(unit.event_store), PipelineBindings(
            unit.projections, unit.event_store, project, layer_id
        )

    def get(self, project_id, layer_id):
        with self.uow_factory() as unit:
            service, _ = self.context(unit, project_id, layer_id)
            return service.load(layer_id)

    def define(self, project_id, layer_id, payload, *, principal, expected_revision, idempotency_key):
        # Only definition fields are accepted: client-supplied runtime/job state
        # cannot be imported through this API.
        definition = LayerPipelineDefinition.from_dict(payload)
        with self.uow_factory() as unit:
            service, bindings = self.context(unit, project_id, layer_id, principal, mutate=True)
            bindings.validate(definition)
            state = service.define(
                definition,
                principal=principal,
                expected_revision=expected_revision,
                key=idempotency_key,
                at=self.clock.now(),
            )
            unit.commit()
            return state

    def command(self, project_id, layer_id, operation, *, principal, expected_revision, idempotency_key, node_id=None):
        with self.uow_factory() as unit:
            service, bindings = self.context(unit, project_id, layer_id, principal, mutate=True)
            bindings.validate(LayerPipelineDefinition.from_dict(service.load(layer_id)["definition"]))
            state = service.command(
                layer_id,
                operation,
                principal=principal,
                expected_revision=expected_revision,
                key=idempotency_key,
                at=self.clock.now(),
                node_id=node_id,
            )
            if operation == "cancel":
                for node in state["nodes"].values():
                    if node["attempts"] and node["attempts"][-1]["request"].get("kind") == "layer":
                        cancel_stitch_intent(unit, project_id, node["attempts"][-1]["job_id"], principal)
            unit.commit()
            return state

    def tick(self, project_id, layer_id, *, principal):
        if self.submit is None and self.stitch_dispatch is None:
            raise ConflictError("Pipeline plugin queue is not configured")
        if self.stitch_refresh is not None:
            previous = self.get(project_id, layer_id)
            for node in previous["nodes"].values():
                if node["attempts"] and node["attempts"][-1]["request"].get("kind") == "layer":
                    self.stitch_refresh(project_id, node["attempts"][-1]["job_id"], principal)
        with self.uow_factory() as unit:
            service, bindings = self.context(unit, project_id, layer_id, principal, mutate=True)
            bindings.validate(LayerPipelineDefinition.from_dict(service.load(layer_id)["definition"]))
            state, intents = service.advance(
                layer_id,
                principal=principal,
                at=self.clock.now(),
                observe=bindings.observe,
                bind_inputs=bindings.bind,
                current_inputs=bindings.current,
            )
            for intent in intents:
                if intent["request"].get("kind") == "layer":
                    ensure_stitch_intent(unit, bindings, intent, principal)
            unit.commit()
        for intent in intents:
            # Re-read after committing reservations, allowing a concurrent pause
            # or cancellation to stop new dispatch. Cancellation still reconciles
            # an already persisted intent before declaring the run stopped.
            current = self.get(project_id, layer_id)
            matching = any(
                node["status"] == "dispatching" and node["attempts"][-1]["job_id"] == intent["job_id"]
                for node in current["nodes"].values()
            )
            if not matching or current["status"] not in {"running", "cancelling"}:
                continue
            request = intent["request"]
            if request.get("kind") == "layer":
                if current["status"] == "cancelling":
                    with self.uow_factory() as unit:
                        cancel_stitch_intent(unit, project_id, intent["job_id"], principal)
                        unit.commit()
                    if self.stitch_cancel is not None:
                        self.stitch_cancel(project_id, intent["job_id"], principal)
                    continue
                if self.stitch_dispatch is None:
                    raise ConflictError("Pipeline layer execution is not configured")
                self.stitch_dispatch(project_id, intent["job_id"], principal)
                with self.uow_factory() as unit:
                    service, _ = self.context(unit, project_id, layer_id, principal, mutate=True)
                    service.dispatched(layer_id, intent["job_id"], principal=principal, at=self.clock.now())
                    unit.commit()
                continue
            selection = FrameSelectionV1(
                row_ranges=tuple(FrameRowRange(y=y, x_start=x, x_end=x) for x, y in request["coordinates"])
            )
            try:
                self.submit(
                    SubmitPluginJobCommand(
                        context=CommandContext(
                            actor=principal,
                            idempotency_key=intent["idempotency_key"],
                            gitlab_identity_verified=self.verified,
                        ),
                        project_id=project_id,
                        layer_id=layer_id,
                        job_id=intent["job_id"],
                        target_representation_id=request["target_representation_id"],
                        selection=selection,
                        capability=request["capability"],
                        parameters=request["parameters"],
                        inputs=tuple(PluginInputV1(**item) for item in request["inputs"]),
                    )
                )
                with self.uow_factory() as unit:
                    service, _ = self.context(unit, project_id, layer_id, principal, mutate=True)
                    service.dispatched(layer_id, intent["job_id"], principal=principal, at=self.clock.now())
                    unit.commit()
            except (ConflictError, DomainValidationError, NotFoundError) as exc:
                # If submission committed before Agent handoff failed, retain the
                # intent: replaying the existing handler repairs that exact job.
                with self.uow_factory() as unit:
                    service, _ = self.context(unit, project_id, layer_id, principal, mutate=True)
                    if unit.projections.get_plugin_job(intent["job_id"]) is None:
                        service.dispatch_failed(
                            layer_id, intent["job_id"], exc, principal=principal, at=self.clock.now()
                        )
                        unit.commit()
        current = self.get(project_id, layer_id)
        if current["status"] == "cancelling":
            for node in current["nodes"].values():
                if not node["attempts"]:
                    continue
                job_id = node["attempts"][-1]["job_id"]
                if node["attempts"][-1]["request"].get("kind") == "layer":
                    if self.stitch_cancel is not None:
                        self.stitch_cancel(project_id, job_id, principal)
                    continue
                with self.uow_factory() as unit:
                    job = unit.projections.get_plugin_job(job_id)
                if job and job.state.value not in {"succeeded", "failed", "cancelled"}:
                    self.cancel(
                        CancelPluginJobCommand(
                            context=CommandContext(
                                actor=principal,
                                idempotency_key=f"pipeline-cancel:{job_id}",
                                gitlab_identity_verified=self.verified,
                            ),
                            project_id=project_id,
                            job_id=job_id,
                            expected_revision=job.revision,
                        )
                    )
        return self.get(project_id, layer_id)
