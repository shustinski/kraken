"""composition: add external artifact version."""
from __future__ import annotations
import hashlib
from kraken_core.media_types import artifact_media_type
from pathlib import Path
from kraken_core.safe_files import open_regular_read
from kraken_manager.application.dto import (
    AddArtifactVersionCommand,
    ActivateArtifactVersionCommand,
    AddExternalArtifactVersionCommand,
    ArchiveArtifactSeriesCommand,
    CommandContext,
    RenameArtifactSeriesCommand,
)
from kraken_manager.domain.artifacts import (
    ArtifactSeries,
    ArtifactVersion,
)
from kraken_manager.domain.common import (
    ArtifactSeriesId,
    ArtifactVersionId,
    ProjectId,
)
from kraken_manager.domain.identity import (
    Principal,
)


class AddExternalArtifactVersionMixin:
    def add_managed_artifact_version(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        series_id: ArtifactSeriesId | str,
        source: Path | str,
        parent_version_id: ArtifactVersionId | str | None = None,
        idempotency_key: str,
    ) -> ArtifactVersion:
        path = Path(source).resolve(strict=True)
        media_type = artifact_media_type(path.name)
        with open_regular_read(path, root=path.parent) as stream:
            return self._commands(str(project_id)).execute(
                AddArtifactVersionCommand(
                    context=CommandContext(
                        actor=principal,
                        idempotency_key=idempotency_key,
                    ),
                    project_id=ProjectId(str(project_id)),
                    series_id=ArtifactSeriesId(str(series_id)),
                    filename=path.name,
                    media_type=media_type,
                    expected_series_revision=self.artifact_stream_revision(
                        project_id,
                        series_id,
                    ),
                    parent_version_id=(
                        None
                        if parent_version_id is None
                        else ArtifactVersionId(str(parent_version_id))
                    ),
                    tool_name="Kraken Desktop",
                ),
                iter(lambda: stream.read(1024 * 1024), b""),
            )

    def add_external_artifact_version(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        series_id: ArtifactSeriesId | str,
        source: Path | str,
        parent_version_id: ArtifactVersionId | str | None = None,
        idempotency_key: str,
    ) -> ArtifactVersion:
        path = Path(source).resolve(strict=True)
        digest = hashlib.sha256()
        size = 0
        with open_regular_read(path, root=path.parent) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        return self._commands(str(project_id)).execute(
            AddExternalArtifactVersionCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=ProjectId(str(project_id)),
                series_id=ArtifactSeriesId(str(series_id)),
                filename=path.name,
                media_type=artifact_media_type(path.name),
                uri=path.as_uri(),
                fingerprint_sha256=digest.hexdigest(),
                observed_size_bytes=size,
                expected_series_revision=self.artifact_stream_revision(
                    project_id,
                    series_id,
                ),
                parent_version_id=(
                    None
                    if parent_version_id is None
                    else ArtifactVersionId(str(parent_version_id))
                ),
                parameters={"local_path": str(path)},
            )
        )

    def rename_artifact_series(
        self,
        *,
        principal: Principal,
        series: ArtifactSeries,
        name: str,
        idempotency_key: str,
    ) -> ArtifactSeries:
        return self._commands(str(series.project_id)).execute(
            RenameArtifactSeriesCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=series.project_id,
                series_id=series.id,
                name=name,
                expected_series_revision=series.revision,
            )
        )

    def archive_artifact_series(
        self,
        *,
        principal: Principal,
        series: ArtifactSeries,
        idempotency_key: str,
    ) -> ArtifactSeries:
        return self._commands(str(series.project_id)).execute(
            ArchiveArtifactSeriesCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=series.project_id,
                series_id=series.id,
                expected_series_revision=series.revision,
            )
        )

    def activate_artifact_version(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        series_id: ArtifactSeriesId | str,
        version_id: ArtifactVersionId | str,
        idempotency_key: str,
    ) -> ArtifactVersion:
        return self._commands(str(project_id)).execute(
            ActivateArtifactVersionCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=ProjectId(str(project_id)),
                series_id=ArtifactSeriesId(str(series_id)),
                version_id=ArtifactVersionId(str(version_id)),
            )
        )
