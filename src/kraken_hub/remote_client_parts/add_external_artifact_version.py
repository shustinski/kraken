"""remote_client: add external artifact version."""
from __future__ import annotations
import hashlib
import mimetypes
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from kraken_manager.domain.artifacts import (
    ArtifactSeries,
    ArtifactVersion,
    NoteRevision,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.infrastructure.filesystem._codec import decode_model

from kraken_hub.remote_client_parts.extract_review_archive import RemoteServerError


class AddExternalArtifactVersionMixin:
    def add_external_artifact_version(
        self,
        *,
        principal: Principal,
        project_id: object,
        series_id: object,
        source: Path | str,
        parent_version_id: object | None = None,
        idempotency_key: str,
    ) -> ArtifactVersion:
        del principal
        path = Path(source).resolve(strict=True)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project_id}/artifacts/{series_id}/external",
            payload={
                "filename": path.name,
                "media_type": mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
                "uri": path.as_uri(),
                "sha256": digest.hexdigest(),
                "size_bytes": size,
                "parent_version_id": None
                if parent_version_id is None
                else str(parent_version_id),
                "parameters": {"source_path": str(path)},
            },
            idempotency_key=idempotency_key,
            if_match=self.artifact_stream_revision(project_id, series_id),
        )
        return decode_model(ArtifactVersion, payload)

    def rename_artifact_series(
        self,
        *,
        principal: Principal,
        series: ArtifactSeries,
        name: str,
        idempotency_key: str,
    ) -> ArtifactSeries:
        del principal
        payload = self._call(
            "PATCH",
            f"/api/v1/projects/{series.project_id}/artifacts/{series.id}",
            payload={"name": name},
            idempotency_key=idempotency_key,
            if_match=series.revision,
        )
        return decode_model(ArtifactSeries, payload)

    def archive_artifact_series(
        self, *, principal: Principal, series: ArtifactSeries, idempotency_key: str
    ) -> ArtifactSeries:
        del principal
        payload = self._call(
            "PATCH",
            f"/api/v1/projects/{series.project_id}/artifacts/{series.id}",
            payload={"archive": True},
            idempotency_key=idempotency_key,
            if_match=series.revision,
        )
        return decode_model(ArtifactSeries, payload)

    def activate_artifact_version(
        self,
        *,
        principal: Principal,
        project_id: object,
        series_id: object,
        version_id: object,
        idempotency_key: str,
    ) -> ArtifactVersion:
        del principal
        payload = self._call(
            "PATCH",
            f"/api/v1/projects/{project_id}/artifacts/{series_id}",
            payload={"active_version_id": str(version_id)},
            idempotency_key=idempotency_key,
            if_match=self.artifact_stream_revision(project_id, series_id),
        )
        return decode_model(ArtifactVersion, payload)

    def list_notes(
        self,
        project_id: object,
        *,
        layer_id: object | None = None,
        frame_id: object | None = None,
    ) -> tuple[NoteRevision, ...]:
        query = urllib.parse.urlencode(
            {
                key: value
                for key, value in {
                    "layer_id": None if layer_id is None else str(layer_id),
                    "frame_id": None if frame_id is None else str(frame_id),
                }.items()
                if value is not None
            }
        )
        payload = self._call("GET", f"/api/v1/projects/{project_id}/notes?{query}")
        items = payload.get("items", ()) if isinstance(payload, Mapping) else ()
        return tuple(
            decode_model(NoteRevision, item)
            for item in items
            if isinstance(item, Mapping)
        )

    def create_note(
        self,
        *,
        principal: Principal,
        project_id: object,
        body: str,
        layer_id: object | None = None,
        frame_id: object | None = None,
        idempotency_key: str,
    ) -> NoteRevision:
        del principal
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project_id}/notes",
            payload={
                "body": body,
                "layer_id": None if layer_id is None else str(layer_id),
                "frame_id": None if frame_id is None else str(frame_id),
            },
            idempotency_key=idempotency_key,
        )
        return decode_model(NoteRevision, payload)

    def revise_note(
        self,
        *,
        principal: Principal,
        note: NoteRevision,
        body: str,
        idempotency_key: str,
    ) -> NoteRevision:
        del principal
        payload = self._call(
            "PATCH",
            f"/api/v1/projects/{note.project_id}/notes/{note.note_id}",
            payload={"body": body},
            idempotency_key=idempotency_key,
            if_match=note.revision,
        )
        return decode_model(NoteRevision, payload)

    def _download_managed_artifact(self, project_id: object, version_id: object, destination: Path) -> Path:
        try:
            transfer = self._call(
                "GET",
                f"/api/v1/projects/{project_id}/artifacts/versions/{version_id}/download-ticket",
            )
        except RemoteServerError as exc:
            if exc.status != 404:
                raise
            transfer = {"mode": "proxy"}
        mode = str(transfer.get("mode", "")) if isinstance(transfer, Mapping) else ""
        if mode == "gateway":
            return self._http.download_url(
                str(transfer["url"]), token=str(transfer["token"]), destination=destination
            )
        if mode == "proxy":
            return self._http.download(
                f"/api/v1/projects/{project_id}/artifacts/versions/{version_id}",
                token=self.auth.access_token,
                destination=destination,
            )
        raise RemoteServerError("Kraken Server returned an unsupported blob transfer mode")

    def export_managed_artifact(
        self, project_id: object, version: ArtifactVersion, destination: Path | str
    ) -> Path:
        if version.blob is None:
            raise ValueError("The selected artifact version is external")
        return self._download_managed_artifact(project_id, version.id, Path(destination).resolve())
