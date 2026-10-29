"""composition: create artifact series."""
from __future__ import annotations
import json
from datetime import datetime
from kraken_manager.application.dto import (
    ArchiveLayerCommand,
    CommandContext,
    CreateArtifactSeriesCommand,
    ReorderLayerCommand,
)
from kraken_manager.application.use_cases import (
    CreateArtifactSeriesHandler,
)
from kraken_manager.application.lifecycle import (
    ArchiveLayerHandler,
    ReorderLayerHandler,
)
from kraken_manager.domain.artifacts import (
    ArtifactScope,
    ArtifactSeries,
)
from kraken_manager.domain.common import (
    ArtifactSeriesId,
    FrameId,
    LayerId,
    ProjectId,
    RepresentationId,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.project import (
    Layer,
    Project,
    Representation,
)
from kraken_manager.infrastructure.filesystem import (
    FileProjectLayout,
    FilesystemEventStore,
    SQLiteProjectionStore,
)
from kraken_manager.infrastructure.projections import rebuild_filesystem_index
from kraken_manager.infrastructure.projections.legacy_work_orders import LegacyWorkOrderMigrationRequired


class CreateArtifactSeriesMixin:
    def reorder_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        order: int,
        idempotency_key: str,
    ) -> Layer:
        return ReorderLayerHandler(self._uow(str(project.id)), self.profiles, self.clock)(
            ReorderLayerCommand(
                context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                project_id=project.id,
                layer_id=layer.id,
                order=order,
                expected_revision=layer.revision,
            )
        )

    def archive_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        idempotency_key: str,
    ) -> Layer:
        return ArchiveLayerHandler(self._uow(str(project.id)), self.profiles, self.clock)(
            ArchiveLayerCommand(
                context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                project_id=project.id,
                layer_id=layer.id,
                expected_revision=layer.revision,
            )
        )

    def get_project(self, project_id: ProjectId | str, *, as_of: datetime | None = None) -> Project | None:
        projections = self._projection(project_id)
        return projections.get_project(ProjectId(str(project_id)), as_of=as_of)

    def _projection(self, project_id: ProjectId | str) -> SQLiteProjectionStore:
        identifier = str(project_id)
        events = FilesystemEventStore(self.catalog_root, identifier)
        with events.lock.hold(events.lock_timeout):
            projections = SQLiteProjectionStore(self.catalog_root, identifier)
            if projections.checkpoint != events.last_global_position():
                rebuild_filesystem_index(events, projections, acl=self.identities)
                projections = SQLiteProjectionStore(self.catalog_root, identifier)
        return projections

    def list_projects(self, *, include_archived: bool = False) -> tuple[Project, ...]:
        projects_root = self.catalog_root / "projects"
        if not projects_root.is_dir():
            return ()
        result: list[Project] = []
        for directory in sorted(path for path in projects_root.iterdir() if path.is_dir()):
            descriptor = directory / "project.json"
            try:
                project_id = ProjectId(
                    str(json.loads(descriptor.read_text(encoding="utf-8"))["project_id"])
                    if descriptor.is_file()
                    else directory.name
                )
                project = self._projection(project_id).get_project(project_id)
            except LegacyWorkOrderMigrationRequired:
                # Keep the catalog entry visible so administration can migrate
                # it. Opening/writing the project still goes through _projection
                # and cannot silently consume a lagging work cache.
                project = SQLiteProjectionStore(self.catalog_root, str(project_id)).get_project(project_id)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            if project is not None and not descriptor.is_file():
                FileProjectLayout(self.catalog_root, str(project.id)).initialize(
                    {
                        "project_id": str(project.id),
                        "name": project.name,
                        "width": project.width,
                        "height": project.height,
                        "orientation": project.orientation.value,
                        "storage_profile": project.storage_profile,
                    }
                )
            if project is not None and (include_archived or project.state.value != "archived"):
                result.append(project)
        return tuple(sorted(result, key=lambda project: project.name.casefold()))

    def list_layers(
        self,
        project_id: ProjectId | str,
        *,
        as_of: datetime | None = None,
        include_archived: bool = False,
    ) -> tuple[Layer, ...]:
        return self._projection(project_id).list_layers(
            ProjectId(str(project_id)), as_of=as_of, include_archived=include_archived
        )

    def list_representations(
        self,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        *,
        include_archived: bool = False,
        as_of: datetime | None = None,
    ) -> tuple[Representation, ...]:
        return self._projection(project_id).list_representations(
            LayerId(str(layer_id)),
            include_archived=include_archived,
            as_of=as_of,
        )

    def list_artifact_series(
        self,
        project_id: ProjectId | str,
        *,
        layer_id: LayerId | str | None = None,
        representation_id: RepresentationId | str | None = None,
        include_archived: bool = False,
    ) -> tuple[ArtifactSeries, ...]:
        return self._projection(project_id).list_artifact_series(
            ProjectId(str(project_id)),
            layer_id=None if layer_id is None else LayerId(str(layer_id)),
            representation_id=(
                None
                if representation_id is None
                else RepresentationId(str(representation_id))
            ),
            include_archived=include_archived,
        )

    def create_artifact_series(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        scope: ArtifactScope,
        name: str,
        layer_id: LayerId | str | None = None,
        representation_id: RepresentationId | str | None = None,
        frame_id: str | None = None,
        series_id: ArtifactSeriesId | str | None = None,
        idempotency_key: str,
    ) -> ArtifactSeries:
        from kraken_manager.domain.common import new_uuid

        return CreateArtifactSeriesHandler(
            self._uow(str(project_id)),
            self.profiles,
            self.clock,
        )(
            CreateArtifactSeriesCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=ProjectId(str(project_id)),
                scope=scope,
                name=name,
                layer_id=None if layer_id is None else LayerId(str(layer_id)),
                representation_id=(
                    None
                    if representation_id is None
                    else RepresentationId(str(representation_id))
                ),
                frame_id=None if frame_id is None else FrameId(str(frame_id)),
                series_id=ArtifactSeriesId(str(series_id)) if series_id is not None else ArtifactSeriesId(new_uuid()),
            )
        )

    def artifact_stream_revision(
        self,
        project_id: ProjectId | str,
        series_id: ArtifactSeriesId | str,
    ) -> int:
        return FilesystemEventStore(
            self.catalog_root,
            str(project_id),
        ).current_revision(f"artifact-series:{series_id}")
