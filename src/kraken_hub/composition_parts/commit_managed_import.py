"""composition: commit managed import."""
from __future__ import annotations
from kraken_core.media_types import artifact_media_type
from pathlib import Path
from kraken_core.safe_files import open_regular_read
from kraken_manager.application.dto import (
    AddArtifactVersionCommand,
    CommandContext,
    CreateArtifactSeriesCommand,
)
from kraken_manager.application.imports import ImportPlan
from kraken_manager.application.use_cases import (
    AddArtifactVersionHandler,
    CreateArtifactSeriesHandler,
)
from kraken_manager.domain.artifacts import (
    ArtifactScope,
    ArtifactVersion,
    deterministic_frame_series_id,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.project import (
    Layer,
    Project,
    Representation,
)

from kraken_hub.composition_parts.frame_cell_snapshot import ManagedImportResult


class CommitManagedImportMixin:
    def commit_managed_import(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
        plan: ImportPlan,
        idempotency_key: str,
    ) -> ManagedImportResult:
        if not plan.ready:
            raise ValueError("Import plan contains blocking issues")
        if representation.project_id != project.id or representation.layer_id != layer.id:
            raise ValueError("Representation does not belong to the selected project and layer")
        versions: list[ArtifactVersion] = []
        for item in plan.items:
            coordinate = project.coordinate(item.x, item.y)
            frame_id = coordinate.frame_id(project.id)
            series_id = deterministic_frame_series_id(representation.id, frame_id)
            per_frame_key = f"{idempotency_key}:{item.x}:{item.y}"
            series = self._projection(project.id).get_artifact_series(series_id)
            if series is None:
                series = CreateArtifactSeriesHandler(
                    self._uow(str(project.id)), self.profiles, self.clock
                )(
                    CreateArtifactSeriesCommand(
                        context=CommandContext(actor=principal, idempotency_key=f"{per_frame_key}:series"),
                        project_id=project.id,
                        scope=ArtifactScope.FRAME_REPRESENTATION,
                        name=item.source.display_name,
                        layer_id=layer.id,
                        representation_id=representation.id,
                        frame_id=frame_id,
                        series_id=series_id,
                    )
                )
            elif (series.archived or series.project_id != project.id or series.layer_id != layer.id
                  or series.representation_id != representation.id or series.frame_id != frame_id
                  or series.scope is not ArtifactScope.FRAME_REPRESENTATION):
                raise ValueError("Existing frame artifact is archived or has inconsistent ownership")
            stream_revision = self.artifact_stream_revision(project.id, series.id)
            source_path = Path(item.source.source_key)
            root = source_path.parent
            media_type = artifact_media_type(source_path.name)
            with open_regular_read(source_path, root=root) as stream:
                chunks = iter(lambda: stream.read(1024 * 1024), b"")
                version = AddArtifactVersionHandler(
                    self._uow(str(project.id)), self.profiles, self.clock
                )(
                    AddArtifactVersionCommand(
                        context=CommandContext(actor=principal, idempotency_key=f"{per_frame_key}:version"),
                        project_id=project.id,
                        series_id=series.id,
                        filename=source_path.name,
                        media_type=media_type,
                        expected_series_revision=stream_revision,
                        expected_sha256=getattr(item.source, "expected_sha256", None),
                        tool_name="Импорт файлов Kraken",
                        parameters={"x": item.x, "y": item.y, "mapping": "preflight"},
                    ),
                    chunks,
                )
            versions.append(version)
        return ManagedImportResult(plan, tuple(versions))
