"""Immutable artifact identities, content references, notes, and attachments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Self
from uuid import UUID, uuid5

from .common import (
    ArtifactSeriesId,
    ArtifactVersionId,
    DomainValidationError,
    FrameId,
    LayerId,
    PrincipalId,
    ProjectId,
    RepresentationId,
    as_utc,
    freeze_mapping,
    new_uuid,
    require_non_empty,
    utc_now,
    validate_uuid,
)


def validate_sha256(value: str, *, field: str = "sha256") -> str:
    digest = value.lower().strip()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise DomainValidationError(f"{field} must be a 64-character hexadecimal SHA-256 digest")
    return digest


def _safe_filename(value: str) -> str:
    filename = require_non_empty(value, field="filename", maximum=512)
    if filename in {".", ".."} or "/" in filename or "\\" in filename or "\x00" in filename:
        raise DomainValidationError("filename must be a plain filename without path components")
    return filename


@dataclass(frozen=True, slots=True)
class BlobRef:
    """Content-addressed managed blob reference."""

    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "sha256", validate_sha256(self.sha256))
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise DomainValidationError("blob size_bytes must be a non-negative integer")

    @property
    def storage_key(self) -> str:
        return f"sha256/{self.sha256[:2]}/{self.sha256[2:4]}/{self.sha256}"


@dataclass(frozen=True, slots=True)
class ExternalReference:
    """An explicitly unmanaged URI with a fingerprint observed at import time."""

    uri: str
    fingerprint_sha256: str
    observed_size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", require_non_empty(self.uri, field="external_reference.uri", maximum=4096))
        object.__setattr__(
            self,
            "fingerprint_sha256",
            validate_sha256(self.fingerprint_sha256, field="external_reference.fingerprint_sha256"),
        )
        if (
            isinstance(self.observed_size_bytes, bool)
            or not isinstance(self.observed_size_bytes, int)
            or self.observed_size_bytes < 0
        ):
            raise DomainValidationError("external_reference.observed_size_bytes must be non-negative")

    @property
    def history_is_guaranteed(self) -> bool:
        """External bytes may move or change, so only their fingerprint is historical."""

        return False


class ArtifactScope(StrEnum):
    FRAME_REPRESENTATION = "frame_representation"
    PROJECT_ATTACHMENT = "project_attachment"
    LAYER_ATTACHMENT = "layer_attachment"
    PROJECT_EXTERNAL_LINK = "project_external_link"
    LAYER_EXTERNAL_LINK = "layer_external_link"


@dataclass(frozen=True, slots=True)
class ArtifactSeries:
    """Stable logical identity whose content is an immutable version chain."""

    id: ArtifactSeriesId
    project_id: ProjectId
    scope: ArtifactScope
    name: str
    layer_id: LayerId | None = None
    representation_id: RepresentationId | None = None
    frame_id: FrameId | None = None
    revision: int = 0
    archived: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", ArtifactSeriesId(validate_uuid(str(self.id), field="artifact_series.id")))
        object.__setattr__(
            self,
            "project_id",
            ProjectId(validate_uuid(str(self.project_id), field="artifact_series.project_id")),
        )
        object.__setattr__(self, "name", require_non_empty(self.name, field="artifact_series.name", maximum=512))
        if not isinstance(self.scope, ArtifactScope):
            object.__setattr__(self, "scope", ArtifactScope(self.scope))
        if self.layer_id is not None:
            object.__setattr__(
                self, "layer_id", LayerId(validate_uuid(str(self.layer_id), field="artifact_series.layer_id"))
            )
        if self.representation_id is not None:
            object.__setattr__(
                self,
                "representation_id",
                RepresentationId(validate_uuid(str(self.representation_id), field="artifact_series.representation_id")),
            )
        if self.frame_id is not None:
            object.__setattr__(
                self, "frame_id", FrameId(validate_uuid(str(self.frame_id), field="artifact_series.frame_id"))
            )
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise DomainValidationError("artifact_series.revision must not be negative")
        self._validate_scope()

    def _validate_scope(self) -> None:
        if self.scope is ArtifactScope.FRAME_REPRESENTATION:
            if self.layer_id is None or self.representation_id is None or self.frame_id is None:
                raise DomainValidationError("frame representation series requires layer, representation, and frame IDs")
            return
        if self.representation_id is not None or self.frame_id is not None:
            raise DomainValidationError("attachments and external links cannot reference a representation or frame")
        layer_scopes = {ArtifactScope.LAYER_ATTACHMENT, ArtifactScope.LAYER_EXTERNAL_LINK}
        if (self.scope in layer_scopes) != (self.layer_id is not None):
            raise DomainValidationError("layer-scoped series requires a layer; project-scoped series forbids one")

    @classmethod
    def for_frame(
        cls,
        *,
        project_id: ProjectId,
        layer_id: LayerId,
        representation_id: RepresentationId,
        frame_id: FrameId,
        name: str,
        series_id: ArtifactSeriesId | str | None = None,
    ) -> Self:
        return cls(
            id=ArtifactSeriesId(str(series_id) if series_id is not None else new_uuid()),
            project_id=project_id,
            scope=ArtifactScope.FRAME_REPRESENTATION,
            name=name,
            layer_id=layer_id,
            representation_id=representation_id,
            frame_id=frame_id,
        )

    def rename(self, name: str, *, expected_revision: int | None = None) -> Self:
        if expected_revision is not None and expected_revision != self.revision:
            raise DomainValidationError(
                f"expected artifact series revision {expected_revision}, found {self.revision}"
            )
        return replace(
            self,
            name=require_non_empty(name, field="artifact_series.name", maximum=512),
            revision=self.revision + 1,
        )

    def archive(self, *, expected_revision: int | None = None) -> Self:
        if self.archived:
            return self
        if expected_revision is not None and expected_revision != self.revision:
            raise DomainValidationError(
                f"expected artifact series revision {expected_revision}, found {self.revision}"
            )
        return replace(self, archived=True, revision=self.revision + 1)


def deterministic_frame_series_id(
    representation_id: RepresentationId | str,
    frame_id: FrameId | str,
) -> ArtifactSeriesId:
    """Return the stable logical file ID for a representation/frame pair."""

    representation = validate_uuid(str(representation_id), field="representation_id")
    frame = validate_uuid(str(frame_id), field="frame_id")
    return ArtifactSeriesId(str(uuid5(UUID(representation), f"frame-artifact-series:{frame}")))


@dataclass(frozen=True, slots=True)
class ArtifactVersion:
    """An immutable managed or external version with complete provenance."""

    id: ArtifactVersionId
    series_id: ArtifactSeriesId
    sha256: str
    size_bytes: int
    media_type: str
    filename: str
    created_at: datetime
    author_principal_id: PrincipalId
    blob: BlobRef | None = None
    external: ExternalReference | None = None
    parent_version_id: ArtifactVersionId | None = None
    input_version_ids: tuple[ArtifactVersionId, ...] = ()
    tool_name: str | None = None
    tool_version: str | None = None
    parameters: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", ArtifactVersionId(validate_uuid(str(self.id), field="artifact_version.id")))
        object.__setattr__(
            self,
            "series_id",
            ArtifactSeriesId(validate_uuid(str(self.series_id), field="artifact_version.series_id")),
        )
        object.__setattr__(self, "sha256", validate_sha256(self.sha256, field="artifact_version.sha256"))
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise DomainValidationError("artifact_version.size_bytes must be non-negative")
        media_type = require_non_empty(self.media_type, field="artifact_version.media_type", maximum=255).lower()
        if "/" not in media_type:
            raise DomainValidationError("artifact_version.media_type must be a MIME type")
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "filename", _safe_filename(self.filename))
        object.__setattr__(self, "created_at", as_utc(self.created_at, field="artifact_version.created_at"))
        object.__setattr__(
            self,
            "author_principal_id",
            PrincipalId(validate_uuid(str(self.author_principal_id), field="artifact_version.author_principal_id")),
        )
        if (self.blob is None) == (self.external is None):
            raise DomainValidationError("artifact version requires exactly one managed blob or external reference")
        reference_sha = self.blob.sha256 if self.blob is not None else self.external.fingerprint_sha256
        reference_size = self.blob.size_bytes if self.blob is not None else self.external.observed_size_bytes
        if reference_sha != self.sha256 or reference_size != self.size_bytes:
            raise DomainValidationError("artifact content metadata must match its content reference")
        if self.parent_version_id is not None:
            parent_id = ArtifactVersionId(
                validate_uuid(str(self.parent_version_id), field="artifact_version.parent_version_id")
            )
            if parent_id == self.id:
                raise DomainValidationError("an artifact version cannot be its own parent")
            object.__setattr__(self, "parent_version_id", parent_id)
        inputs = tuple(
            ArtifactVersionId(validate_uuid(str(item), field="artifact_version.input_version_ids"))
            for item in self.input_version_ids
        )
        if len(inputs) != len(set(inputs)):
            raise DomainValidationError("artifact input versions must be unique")
        object.__setattr__(self, "input_version_ids", inputs)
        if self.tool_name is not None:
            object.__setattr__(self, "tool_name", require_non_empty(self.tool_name, field="artifact_version.tool_name"))
        if self.tool_version is not None:
            if self.tool_name is None:
                raise DomainValidationError("tool_version requires tool_name")
            object.__setattr__(
                self, "tool_version", require_non_empty(self.tool_version, field="artifact_version.tool_version")
            )
        mapping = {} if self.parameters is None else self.parameters
        object.__setattr__(self, "parameters", freeze_mapping(mapping, field="artifact_version.parameters"))

    @classmethod
    def managed(
        cls,
        *,
        series_id: ArtifactSeriesId,
        blob: BlobRef,
        media_type: str,
        filename: str,
        author_principal_id: PrincipalId,
        version_id: ArtifactVersionId | str | None = None,
        created_at: datetime | None = None,
        parent_version_id: ArtifactVersionId | None = None,
        input_version_ids: tuple[ArtifactVersionId, ...] = (),
        tool_name: str | None = None,
        tool_version: str | None = None,
        parameters: Mapping[str, object] | None = None,
    ) -> ArtifactVersion:
        return cls(
            id=ArtifactVersionId(str(version_id) if version_id is not None else new_uuid()),
            series_id=series_id,
            sha256=blob.sha256,
            size_bytes=blob.size_bytes,
            media_type=media_type,
            filename=filename,
            created_at=created_at or utc_now(),
            author_principal_id=author_principal_id,
            blob=blob,
            parent_version_id=parent_version_id,
            input_version_ids=input_version_ids,
            tool_name=tool_name,
            tool_version=tool_version,
            parameters=parameters,
        )

    @classmethod
    def external_link(
        cls,
        *,
        series_id: ArtifactSeriesId,
        reference: ExternalReference,
        media_type: str,
        filename: str,
        author_principal_id: PrincipalId,
        version_id: ArtifactVersionId | str | None = None,
        created_at: datetime | None = None,
        parent_version_id: ArtifactVersionId | None = None,
        parameters: Mapping[str, object] | None = None,
    ) -> ArtifactVersion:
        return cls(
            id=ArtifactVersionId(str(version_id) if version_id is not None else new_uuid()),
            series_id=series_id,
            sha256=reference.fingerprint_sha256,
            size_bytes=reference.observed_size_bytes,
            media_type=media_type,
            filename=filename,
            created_at=created_at or utc_now(),
            author_principal_id=author_principal_id,
            external=reference,
            parent_version_id=parent_version_id,
            parameters=parameters,
        )

    @property
    def managed_content(self) -> bool:
        return self.blob is not None


@dataclass(frozen=True, slots=True)
class NoteRevision:
    note_id: str
    revision: int
    project_id: ProjectId
    author_principal_id: PrincipalId
    body: str
    recorded_at: datetime
    layer_id: LayerId | None = None
    frame_id: FrameId | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "note_id", validate_uuid(self.note_id, field="note.id"))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise DomainValidationError("note.revision must start at 1")
        object.__setattr__(
            self, "project_id", ProjectId(validate_uuid(str(self.project_id), field="note.project_id"))
        )
        object.__setattr__(
            self,
            "author_principal_id",
            PrincipalId(validate_uuid(str(self.author_principal_id), field="note.author_principal_id")),
        )
        object.__setattr__(self, "body", require_non_empty(self.body, field="note.body", maximum=100_000))
        object.__setattr__(self, "recorded_at", as_utc(self.recorded_at, field="note.recorded_at"))
        if self.layer_id is not None:
            object.__setattr__(self, "layer_id", LayerId(validate_uuid(str(self.layer_id), field="note.layer_id")))
        if self.frame_id is not None:
            if self.layer_id is None:
                raise DomainValidationError("a frame note must also identify its layer")
            object.__setattr__(self, "frame_id", FrameId(validate_uuid(str(self.frame_id), field="note.frame_id")))


@dataclass(frozen=True, slots=True)
class Attachment:
    """A named project/layer attachment backed by an artifact series."""

    id: str
    project_id: ProjectId
    series_id: ArtifactSeriesId
    name: str
    layer_id: LayerId | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", validate_uuid(self.id, field="attachment.id"))
        object.__setattr__(
            self, "project_id", ProjectId(validate_uuid(str(self.project_id), field="attachment.project_id"))
        )
        object.__setattr__(
            self,
            "series_id",
            ArtifactSeriesId(validate_uuid(str(self.series_id), field="attachment.series_id")),
        )
        object.__setattr__(self, "name", require_non_empty(self.name, field="attachment.name", maximum=512))
        if self.layer_id is not None:
            object.__setattr__(
                self, "layer_id", LayerId(validate_uuid(str(self.layer_id), field="attachment.layer_id"))
            )


__all__ = [
    "ArtifactScope",
    "ArtifactSeries",
    "ArtifactVersion",
    "Attachment",
    "BlobRef",
    "deterministic_frame_series_id",
    "ExternalReference",
    "NoteRevision",
    "validate_sha256",
]
