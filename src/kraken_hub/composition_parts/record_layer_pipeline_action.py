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
    def set_layer_stitching_status(
        self, *, principal: Principal, project_id: object, layer_id: object,
        kind: str, completed: bool, version_id: str = "",
    ) -> EventEnvelope:
        from kraken_manager.application.authorization import AuthorizationPolicy
        from kraken_manager.domain.common import ArtifactVersionId
        from kraken_manager.domain.identity import Permission
        from kraken_manager.domain.stitching import StitchingKind, validate_stitching_file

        kind = StitchingKind(kind).value
        if not isinstance(completed, bool):
            raise ValueError("completed must be a boolean")
        with self._uow(str(project_id))() as uow:
            project = uow.projections.get_project(ProjectId(str(project_id)))
            layer = uow.projections.get_layer(LayerId(str(layer_id)))
            if project is None or layer is None or str(layer.project_id) != str(project_id):
                raise ValueError("Слой не найден")
            AuthorizationPolicy().require(
                principal=principal, storage=self.profiles.get(project.storage_profile),
                permission=Permission.MANAGE_STRUCTURE, roles=uow.acl.roles_for(project.id, principal.id),
            )
            if completed:
                version = uow.projections.get_artifact_version(ArtifactVersionId(version_id)) if version_id else None
                series = None if version is None else uow.projections.get_artifact_series(version.series_id)
                validate_stitching_file(project_id, layer_id, kind, version, series)
            stream = f"layer-pipeline:{layer_id}"
            revision = uow.event_store.current_revision(stream)
            event = EventEnvelope.create(
                stream_id=stream, project_id=project.id, revision=revision + 1,
                event_type="LayerStitchingStatusChangedV1",
                payload={"layer_id": str(layer_id), "kind": kind, "completed": completed, "version_id": version_id},
                actor=ActorSnapshot.from_principal(principal), recorded_at=self.clock.now(),
            )
            uow.event_store.append(stream, expected_revision=revision, events=(event,))
            uow.commit()
            return event

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
