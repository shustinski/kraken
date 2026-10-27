"""Persistent layer graph orchestration using existing server job handlers."""

import logging

from kraken_manager.infrastructure.pipeline_runtime import PipelineRuntime
from .layer_source_scan import SystemClock


class PipelineExecutionMixin:
    def _layer_runtime(self):
        return PipelineRuntime(
            self.uow_factory,
            self.profiles,
            SystemClock(),
            self._submit_plugin_job,
            self._cancel_plugin_job,
            verified=True,
            stitch_dispatch=lambda project_id, job_id, principal: self.dispatch_stitch_project(project_id),
            stitch_cancel=lambda project_id, job_id, principal: self.agent_gateway.cancel(job_id) if self.agent_gateway is not None else None,
        )

    def get_layer_pipeline(self, project_id, layer_id):
        try:
            return self._layer_runtime().get(project_id, layer_id)
        except Exception as exc:
            self._translate_planning_error(exc)
            raise

    def transport_layer_pipeline(self, project_id, layer_id, payload, context, *, define=False):
        try:
            actor = self._actor(context.actor_id)
            runtime = self._layer_runtime()
            common = dict(
                principal=actor, expected_revision=context.expected_revision, idempotency_key=context.idempotency_key
            )
            if define:
                return runtime.define(project_id, layer_id, payload, **common)
            if set(payload) - {"operation", "node_id"}:
                raise ValueError("Invalid pipeline command fields")
            return runtime.command(project_id, layer_id, payload["operation"], node_id=payload.get("node_id"), **common)
        except Exception as exc:
            self._translate_planning_error(exc)
            raise

    def run_pipeline_cycle(self):
        if getattr(self, "_submit_plugin_job", None) is None and not hasattr(getattr(self, "agent_gateway", None), "submit_stitch"):
            return 0
        count = 0
        for project in self.list_projects():
            project_id = str(project["project_id"])
            for layer in self.projections.list_layers(project_id):
                events = self.events.load_stream(f"layer-pipeline:{layer.id}")
                event = next((e for e in reversed(events) if e.event_type == "LayerPipelineWorkflowV1"), None)
                if event is None:
                    continue
                state = event.payload["workflow"]
                # Completed runs are also checked for subsequently stale inputs.
                if state["status"] in {"draft", "cancelled"}:
                    continue
                try:
                    actor = self._actor(state["actor_id"])
                    self._layer_runtime().tick(project_id, str(layer.id), principal=actor)
                    count += 1
                except Exception:
                    logging.getLogger(__name__).exception("Pipeline cycle failed for layer %s", layer.id)
        return count
