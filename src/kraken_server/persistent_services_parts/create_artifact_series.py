"""persistent_services: create artifact series."""
from __future__ import annotations
from collections.abc import Iterable, Mapping
from typing import Any
from uuid import uuid4
from kraken_manager.application.dto import (
    AddArtifactVersionCommand,
    CreateArtifactSeriesCommand,
    StoredContent,
)
from kraken_manager.application.errors import (
    ConflictError as ApplicationConflictError,
)
from kraken_manager.domain.artifacts import ArtifactScope, BlobRef
from kraken_manager.domain.common import (
    ArtifactSeriesId,
    ArtifactVersionId,
    FrameId,
    LayerId,
    RepresentationId,
    validate_uuid,
)
from kraken_manager.infrastructure.filesystem._codec import encode_model
from kraken_server.services import CommandContext, NotFoundError

from kraken_server.persistent_services_parts.layer_source_scan import _project_id


class CreateArtifactSeriesMixin:
    def list_artifact_series(
        self,
        project_id: str,
        *,
        layer_id: str | None = None,
        representation_id: str | None = None,
        frame_id: str | None = None,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        self.get_project(project_id)
        return [
            encode_model(item)
            for item in self.projections.list_artifact_series(
                _project_id(project_id),
                layer_id=None if layer_id is None else LayerId(validate_uuid(layer_id, field="layer_id")),
                representation_id=(
                    None
                    if representation_id is None
                    else RepresentationId(validate_uuid(representation_id, field="representation_id"))
                ),
                frame_id=None if frame_id is None else FrameId(validate_uuid(frame_id, field="frame_id")),
                include_archived=include_archived,
            )
        ]

    def list_artifact_versions(self, project_id: str, series_id: str) -> list[dict[str, Any]]:
        self.get_project(project_id)
        identifier = ArtifactSeriesId(validate_uuid(series_id, field="series_id"))
        series = self.projections.get_artifact_series(identifier)
        if series is None or str(series.project_id) != project_id:
            raise NotFoundError("Artifact series was not found")
        return [encode_model(item) for item in self.projections.list_artifact_versions(identifier)]

    def artifact_stream_revision(self, project_id: str, series_id: str) -> int:
        self.get_project(project_id)
        identifier = ArtifactSeriesId(validate_uuid(series_id, field="series_id"))
        series = self.projections.get_artifact_series(identifier)
        if series is None or str(series.project_id) != project_id:
            raise NotFoundError("Artifact series was not found")
        return self.events.current_revision(f"artifact-series:{identifier}")

    def get_active_artifact_version(self, project_id: str, series_id: str) -> dict[str, Any] | None:
        self.get_project(project_id)
        identifier = ArtifactSeriesId(validate_uuid(series_id, field="series_id"))
        series = self.projections.get_artifact_series(identifier)
        if series is None or str(series.project_id) != project_id:
            raise NotFoundError("Artifact series was not found")
        value = self.projections.get_active_artifact_version(identifier)
        return None if value is None else encode_model(value)

    def get_artifact_version(self, project_id: str, version_id: str) -> dict[str, Any]:
        self.get_project(project_id)
        version = self.projections.get_artifact_version(
            ArtifactVersionId(validate_uuid(version_id, field="version_id"))
        )
        if version is None:
            raise NotFoundError("Artifact version was not found")
        series = self.projections.get_artifact_series(version.series_id)
        if series is None or str(series.project_id) != project_id:
            raise NotFoundError("Artifact version was not found in this project")
        return encode_model(version)

    def create_artifact_series(
        self,
        project_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        try:
            value = self._create_artifact_series(
                CreateArtifactSeriesCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    scope=ArtifactScope(str(payload.get("scope", "project_attachment"))),
                    name=str(payload.get("name", "")),
                    layer_id=(
                        None
                        if payload.get("layer_id") is None
                        else LayerId(validate_uuid(str(payload["layer_id"]), field="layer_id"))
                    ),
                    representation_id=(
                        None
                        if payload.get("representation_id") is None
                        else RepresentationId(
                            validate_uuid(str(payload["representation_id"]), field="representation_id")
                        )
                    ),
                    frame_id=(
                        None
                        if payload.get("frame_id") is None
                        else FrameId(validate_uuid(str(payload["frame_id"]), field="frame_id"))
                    ),
                    series_id=ArtifactSeriesId(
                        validate_uuid(str(payload["series_id"]), field="series_id")
                        if payload.get("series_id") is not None
                        else str(uuid4())
                    ),
                )
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def add_managed_artifact_version(
        self,
        project_id: str,
        series_id: str,
        payload: Mapping[str, Any],
        chunks: Iterable[bytes],
        context: CommandContext,
    ) -> dict[str, Any]:
        try:
            value = self._add_artifact_version(
                self._managed_artifact_command(project_id, series_id, payload, context),
                chunks,
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def prepare_managed_artifact_upload(
        self,
        project_id: str,
        series_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any] | None:
        try:
            value = self._add_artifact_version.preflight(
                self._managed_artifact_command(project_id, series_id, payload, context)
            )
            return None if value is None else encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def register_managed_artifact_upload(
        self,
        project_id: str,
        series_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        try:
            digest = str(payload.get("sha256", ""))
            size_bytes = int(payload.get("size_bytes", -1))
            reference = self.uow_factory.blobs.stat(BlobRef(sha256=digest, size_bytes=size_bytes))
            if reference.size_bytes != size_bytes:
                raise ApplicationConflictError("Stored blob size does not match the completed upload")
            value = self._add_artifact_version(
                self._managed_artifact_command(project_id, series_id, payload, context),
                stored=StoredContent(blob=reference, already_existed=True),
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def _managed_artifact_command(
        self,
        project_id: str,
        series_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> AddArtifactVersionCommand:
        return AddArtifactVersionCommand(
            context=self._application_context(context),
            project_id=_project_id(project_id),
            series_id=ArtifactSeriesId(validate_uuid(series_id, field="series_id")),
            filename=str(payload.get("filename", "")),
            media_type=str(payload.get("media_type", "application/octet-stream")),
            expected_series_revision=self._require_revision(context),
            parent_version_id=(
                None
                if payload.get("parent_version_id") is None
                else ArtifactVersionId(validate_uuid(str(payload["parent_version_id"]), field="parent_version_id"))
            ),
            expected_sha256=(None if payload.get("sha256") is None else str(payload["sha256"])),
        )
