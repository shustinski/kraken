"""composition: remove layer pipeline action."""
from __future__ import annotations
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.domain.common import (
    LayerId,
    PluginJobId,
    ProjectId,
)
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope, ProgramSnapshot
from kraken_manager.domain.identity import (
    Permission,
    Principal,
)
from kraken_manager.infrastructure.filesystem import (
    FilesystemEventStore,
)


class RemoveLayerPipelineActionMixin:
    def remove_layer_pipeline_action(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        action_event_id: str,
        remove_task: bool = False,
    ) -> EventEnvelope:
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Project does not exist")
        AuthorizationPolicy().require(
            principal=principal,
            storage=self.profiles.get(project.storage_profile),
            permission=Permission.RUN_PLUGIN,
            roles=self.project_roles(project.id, principal.id),
        )
        target = next(
            (
                event
                for event in self.history(project.id)
                if event.event_id == str(action_event_id)
                and event.event_type in {"LayerPipelineActionRequested", "PluginJobCreated"}
                and str(event.payload.get("layer_id", event.payload.get("manifest", {}).get("layer_id", "")))
                == str(layer_id)
            ),
            None,
        )
        if target is None:
            raise ValueError("Pipeline step does not exist in this layer")
        if remove_task:
            job = self._projection(project.id).get_plugin_job(PluginJobId(str(target.payload.get("plugin_job_id", ""))))
            if job is None or job.state.value not in {"succeeded", "failed", "cancelled"}:
                raise ValueError("Cancel the task before deleting it")
        prior = next(
            (
                event
                for event in self.history(project.id)
                if event.event_type == "LayerPipelineActionRemoved"
                and str(event.payload.get("action_event_id", "")) == target.event_id
                and (not remove_task or event.payload.get("remove_task"))
            ),
            None,
        )
        if prior is not None:
            return prior
        stream_id = f"layer-pipeline:{layer_id}"
        store = FilesystemEventStore(self.catalog_root, str(project.id))
        revision = store.current_revision(stream_id)
        event = EventEnvelope.create(
            stream_id=stream_id,
            project_id=project.id,
            revision=revision + 1,
            event_type="LayerPipelineActionRemoved",
            payload={
                "layer_id": str(layer_id),
                "action_event_id": target.event_id,
                "remove_task": remove_task,
                "plugin_job_id": str(target.payload.get("plugin_job_id", "")),
                "action": str(target.payload.get("action", "")),
            },
            actor=ActorSnapshot.from_principal(principal),
            program=ProgramSnapshot("Kraken"),
        )
        store.append(stream_id, expected_revision=revision, events=(event,))
        return event
