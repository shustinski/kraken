"""composition: record layer pipeline action."""
from __future__ import annotations
from kraken_manager.domain.common import (
    LayerId,
    ProjectId,
)
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope, ProgramSnapshot
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.infrastructure.filesystem import (
    FilesystemEventStore,
)


class RecordLayerPipelineActionMixin:
    def record_layer_pipeline_action(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        action: str,
        node_id: str,
        plugin_id: str,
        capability: str,
        mode: str,
        parameters: dict[str, object] | None = None,
    ) -> EventEnvelope:
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Project does not exist")
        if not any(str(value.id) == str(layer_id) for value in self.list_layers(project.id)):
            raise ValueError("Layer does not exist")
        stream_id = f"layer-pipeline:{layer_id}"
        store = FilesystemEventStore(self.catalog_root, str(project.id))
        revision = store.current_revision(stream_id)
        event = EventEnvelope.create(
            stream_id=stream_id,
            project_id=project.id,
            revision=revision + 1,
            event_type="LayerPipelineActionRequested",
            payload={
                "layer_id": str(layer_id),
                "action": str(action),
                "node_id": str(node_id),
                "plugin_id": str(plugin_id),
                "capability": str(capability),
                "mode": str(mode),
                "parameters": dict(parameters or {}),
                "state": "launched",
            },
            actor=ActorSnapshot.from_principal(principal),
            program=ProgramSnapshot(str(plugin_id)),
        )
        store.append(stream_id, expected_revision=revision, events=(event,))
        return event
