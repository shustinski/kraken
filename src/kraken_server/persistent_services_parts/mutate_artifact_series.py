"""persistent_services: mutate artifact series."""
from __future__ import annotations
from collections.abc import Iterator, Mapping
from typing import Any
from kraken_manager.application.dto import (
    ActivateArtifactVersionCommand,
    AddExternalArtifactVersionCommand,
    ArchiveArtifactSeriesCommand,
    CreateNoteCommand,
    RenameArtifactSeriesCommand,
    ReviseNoteCommand,
)
from kraken_manager.domain.common import (
    ArtifactSeriesId,
    ArtifactVersionId,
    FrameId,
    LayerId,
    validate_uuid,
)
from kraken_manager.infrastructure.filesystem._codec import encode_model
from kraken_server.services import CommandContext, NotFoundError, ValidationError

from kraken_server.persistent_services_parts.layer_source_scan import _project_id


class MutateArtifactSeriesMixin:
    def add_external_artifact_version(
        self,
        project_id: str,
        series_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        try:
            value = self._add_external_artifact_version(
                AddExternalArtifactVersionCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    series_id=ArtifactSeriesId(validate_uuid(series_id, field="series_id")),
                    filename=str(payload.get("filename", "")),
                    media_type=str(payload.get("media_type", "application/octet-stream")),
                    uri=str(payload.get("uri", "")),
                    fingerprint_sha256=str(payload.get("sha256", "")),
                    observed_size_bytes=int(payload.get("size_bytes", -1)),
                    expected_series_revision=self._require_revision(context),
                    parent_version_id=(
                        None
                        if payload.get("parent_version_id") is None
                        else ArtifactVersionId(
                            validate_uuid(
                                str(payload["parent_version_id"]),
                                field="parent_version_id",
                            )
                        )
                    ),
                    parameters=(
                        dict(payload.get("parameters", {}))
                        if isinstance(payload.get("parameters", {}), Mapping)
                        else {}
                    ),
                )
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def mutate_artifact_series(
        self,
        project_id: str,
        series_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        common = {
            "context": self._application_context(context),
            "project_id": _project_id(project_id),
            "series_id": ArtifactSeriesId(validate_uuid(series_id, field="series_id")),
        }
        try:
            if "name" in payload:
                value = self._rename_artifact_series(
                    RenameArtifactSeriesCommand(
                        name=str(payload["name"]),
                        expected_series_revision=self._require_revision(context),
                        **common,
                    )
                )
            elif bool(payload.get("archive")):
                value = self._archive_artifact_series(
                    ArchiveArtifactSeriesCommand(
                        expected_series_revision=self._require_revision(context),
                        **common,
                    )
                )
            elif payload.get("active_version_id") is not None:
                value = self._activate_artifact_version(
                    ActivateArtifactVersionCommand(
                        version_id=ArtifactVersionId(
                            validate_uuid(str(payload["active_version_id"]), field="version_id")
                        ),
                        **common,
                    )
                )
            else:
                raise ValidationError("Exactly one artifact-series operation is required")
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def list_notes(
        self,
        project_id: str,
        *,
        layer_id: str | None = None,
        frame_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self.get_project(project_id)
        return [
            encode_model(item)
            for item in self.projections.list_notes(
                _project_id(project_id),
                layer_id=layer_id,
                frame_id=frame_id,
            )
        ]

    def create_note(
        self,
        project_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        try:
            value = self._create_note(
                CreateNoteCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    body=str(payload.get("body", "")),
                    layer_id=(
                        None
                        if payload.get("layer_id") is None
                        else LayerId(validate_uuid(str(payload["layer_id"]), field="layer_id"))
                    ),
                    frame_id=(
                        None
                        if payload.get("frame_id") is None
                        else FrameId(validate_uuid(str(payload["frame_id"]), field="frame_id"))
                    ),
                )
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def revise_note(
        self,
        project_id: str,
        note_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        try:
            value = self._revise_note(
                ReviseNoteCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    note_id=validate_uuid(note_id, field="note_id"),
                    body=str(payload.get("body", "")),
                    expected_revision=self._require_revision(context),
                )
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def iter_artifact_bytes(
        self,
        project_id: str,
        version_id: str,
        *,
        chunk_size: int = 1024 * 1024,
    ) -> Iterator[bytes]:
        version_payload = self.get_artifact_version(project_id, version_id)
        blob = version_payload.get("blob")
        if not isinstance(blob, Mapping):
            raise ValidationError("External artifact versions do not have managed bytes")
        reference = self.projections.get_artifact_version(
            ArtifactVersionId(validate_uuid(version_id, field="version_id"))
        )
        assert reference is not None and reference.blob is not None
        return self.uow_factory.blobs.iter_bytes(reference.blob, chunk_size=chunk_size)

    def list_review_batches(
        self, project_id: str, *, active_only: bool = False
    ) -> list[dict[str, Any]]:
        project = self.projections.get_project(_project_id(project_id))
        if project is None:
            raise NotFoundError("Project was not found")
        batches = self.projections.list_review_batches(project.id)
        if active_only:
            active_ids = {
                str(batch.id)
                for layer in self.projections.list_layers(project.id)
                for batch in self.projections.list_active_review_batches(project.id, layer.id)
            }
            batches = tuple(batch for batch in batches if str(batch.id) in active_ids)
        return [encode_model(batch) for batch in batches]
