"""composition: export artifact version."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
from kraken_core.safe_files import open_regular_read
from kraken_manager.application.dto import (
    CommandContext,
    CreateNoteCommand,
    ReviseNoteCommand,
)
from kraken_manager.domain.artifacts import (
    ArtifactVersion,
    NoteRevision,
)
from kraken_manager.domain.common import (
    FrameId,
    LayerId,
    ProjectId,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.infrastructure.blob.filesystem import FilesystemBlobStore


def _cleanup_created_export(target: Path, created_identity: tuple[int, int] | None) -> None:
    if created_identity is None:
        return
    try:
        current = target.lstat()
    except FileNotFoundError:
        return
    if (current.st_dev, current.st_ino) == created_identity:
        target.unlink(missing_ok=True)


class ExportArtifactVersionMixin:
    def external_artifact_changed(self, version: ArtifactVersion) -> bool:
        if version.external is None:
            return False
        from urllib.parse import unquote, urlparse

        parsed = urlparse(version.external.uri)
        if parsed.scheme != "file":
            raise ValueError("Desktop supports only local external file links")
        path = Path(unquote(parsed.path.lstrip("/") if os.name == "nt" else parsed.path))
        if os.name == "nt" and parsed.netloc:
            path = Path(f"//{parsed.netloc}/{unquote(parsed.path.lstrip('/'))}")
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            return True
        digest = hashlib.sha256()
        size = 0
        with open_regular_read(resolved, root=resolved.parent) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        return (
            size != version.external.observed_size_bytes
            or digest.hexdigest() != version.external.fingerprint_sha256
        )

    def export_managed_artifact(
        self,
        project_id: ProjectId | str,
        version: ArtifactVersion,
        destination: Path | str,
    ) -> Path:
        if version.blob is None:
            raise ValueError("The selected artifact version is external")
        target = Path(destination).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        created_identity = None
        try:
            with target.open("xb") as stream:
                created = os.fstat(stream.fileno())
                created_identity = (created.st_dev, created.st_ino)
                for chunk in FilesystemBlobStore.for_project(
                    self.catalog_root,
                    str(project_id),
                ).iter_bytes(version.blob):
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            _cleanup_created_export(target, created_identity)
            raise
        return target

    def export_artifact_version(
        self,
        project_id: ProjectId | str,
        version: ArtifactVersion,
        destination: Path | str,
    ) -> Path:
        """Export managed bytes or a verified local external reference."""

        if version.blob is not None:
            return self.export_managed_artifact(project_id, version, destination)
        if version.external is None:
            raise ValueError("Artifact version has no content reference")
        if self.external_artifact_changed(version):
            raise ValueError(
                "Внешний файл изменён или недоступен; выгрузка заблокирована."
            )
        from urllib.parse import unquote, urlparse

        parsed = urlparse(version.external.uri)
        if parsed.scheme != "file":
            raise ValueError("Desktop supports only local external file links")
        source = Path(
            unquote(parsed.path.lstrip("/") if os.name == "nt" else parsed.path)
        )
        if os.name == "nt" and parsed.netloc:
            source = Path(
                f"//{parsed.netloc}/{unquote(parsed.path.lstrip('/'))}"
            )
        source = source.resolve(strict=True)
        target = Path(destination).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        created_identity = None
        try:
            with (
                open_regular_read(source, root=source.parent) as input_stream,
                target.open("xb") as output_stream,
            ):
                created = os.fstat(output_stream.fileno())
                created_identity = (created.st_dev, created.st_ino)
                while chunk := input_stream.read(1024 * 1024):
                    output_stream.write(chunk)
                output_stream.flush()
                os.fsync(output_stream.fileno())
        except Exception:
            _cleanup_created_export(target, created_identity)
            raise
        return target

    def list_notes(
        self,
        project_id: ProjectId | str,
        *,
        layer_id: LayerId | str | None = None,
        frame_id: str | None = None,
    ) -> tuple[NoteRevision, ...]:
        return self._projection(project_id).list_notes(
            ProjectId(str(project_id)),
            layer_id=None if layer_id is None else LayerId(str(layer_id)),
            frame_id=frame_id,
        )

    def create_note(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        body: str,
        layer_id: LayerId | str | None = None,
        frame_id: str | None = None,
        idempotency_key: str,
    ) -> NoteRevision:
        return self._commands(str(project_id)).execute(
            CreateNoteCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=ProjectId(str(project_id)),
                body=body,
                layer_id=None if layer_id is None else LayerId(str(layer_id)),
                frame_id=None if frame_id is None else FrameId(str(frame_id)),
            )
        )

    def revise_note(
        self,
        *,
        principal: Principal,
        note: NoteRevision,
        body: str,
        idempotency_key: str,
    ) -> NoteRevision:
        return self._commands(str(note.project_id)).execute(
            ReviseNoteCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=note.project_id,
                note_id=note.note_id,
                body=body,
                expected_revision=note.revision,
            )
        )
