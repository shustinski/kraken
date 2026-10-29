"""Import a validated frame plan through the shared application handlers."""

from __future__ import annotations
from kraken_manager.application.dto import (
    AddArtifactVersionCommand,
    CommandContext,
    CreateArtifactSeriesCommand,
)
from kraken_manager.application.use_cases import (
    AddArtifactVersionHandler,
    CreateArtifactSeriesHandler,
)
from kraken_manager.domain.artifacts import (
    ArtifactScope,
    ArtifactVersion,
    deterministic_frame_series_id,
)


def commit_managed_import(
    *,
    principal,
    project,
    layer,
    representation,
    plan,
    idempotency_key,
    projections,
    events,
    uow_factory,
    profiles,
    clock,
    content_reader,
    gitlab_identity_verified=False,
):
    if not plan.ready:
        raise ValueError("Import plan contains blocking issues")
    if layer.project_id != project.id or representation.project_id != project.id or representation.layer_id != layer.id:
        raise ValueError("Representation does not belong to the selected project and layer")
    if plan.width != project.width or plan.height != project.height:
        raise ValueError("Import plan dimensions do not match the project")
    coordinates = [(item.x, item.y) for item in plan.items]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("Import plan contains duplicate coordinates")
    versions: list[ArtifactVersion] = []
    for item in plan.items:
        coordinate = project.coordinate(item.x, item.y)
        frame_id = coordinate.frame_id(project.id)
        series_id = deterministic_frame_series_id(representation.id, frame_id)
        per_frame_key = f"{idempotency_key}:{item.x}:{item.y}"
        series = projections.get_artifact_series(series_id)
        if series is None:
            series = CreateArtifactSeriesHandler(uow_factory, profiles, clock)(
                CreateArtifactSeriesCommand(
                    context=CommandContext(
                        actor=principal,
                        gitlab_identity_verified=gitlab_identity_verified,
                        idempotency_key=f"{per_frame_key}:series",
                    ),
                    project_id=project.id,
                    scope=ArtifactScope.FRAME_REPRESENTATION,
                    name=item.source.display_name,
                    layer_id=layer.id,
                    representation_id=representation.id,
                    frame_id=frame_id,
                    series_id=series_id,
                    file_history="batch",
                    batch_key=idempotency_key,
                )
            )
        elif (
            series.archived
            or series.project_id != project.id
            or series.layer_id != layer.id
            or series.representation_id != representation.id
            or series.frame_id != frame_id
            or series.scope is not ArtifactScope.FRAME_REPRESENTATION
        ):
            raise ValueError("Existing frame artifact is archived or has inconsistent ownership")
        stream_revision = events.current_revision(f"artifact-series:{series.id}")
        metadata = content_reader.describe(item.source)
        version = AddArtifactVersionHandler(uow_factory, profiles, clock)(
            AddArtifactVersionCommand(
                context=CommandContext(
                    actor=principal,
                    gitlab_identity_verified=gitlab_identity_verified,
                    idempotency_key=f"{per_frame_key}:version",
                ),
                project_id=project.id,
                series_id=series.id,
                filename=metadata.filename,
                media_type=metadata.media_type,
                expected_series_revision=stream_revision,
                expected_sha256=getattr(item.source, "expected_sha256", None),
                tool_name="Импорт файлов Kraken",
                parameters={"x": item.x, "y": item.y, "mapping": "preflight"},
                file_history="batch",
                batch_key=idempotency_key,
            ),
            content_reader.iter_bytes(item.source),
        )
        versions.append(version)
    return tuple(versions)
