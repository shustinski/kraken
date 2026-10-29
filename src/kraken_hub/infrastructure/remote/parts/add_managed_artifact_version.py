"""remote_client: add managed artifact version."""
from __future__ import annotations
import hashlib
from kraken_core.media_types import artifact_media_type
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from kraken_manager.domain.artifacts import (
    ArtifactVersion,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.infrastructure.filesystem._codec import decode_model
from kraken_hub.application.workspace import (
    ProjectEventWake,
)

from kraken_hub.infrastructure.remote.parts.extract_review_archive import RemoteServerError


class AddManagedArtifactVersionMixin:
    def _add_managed_artifact_version_proxy(
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
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        query = urllib.parse.urlencode(
            {
                "filename": path.name,
                "media_type": artifact_media_type(path.name),
                "sha256": digest.hexdigest(),
                **(
                    {}
                    if parent_version_id is None
                    else {"parent_version_id": str(parent_version_id)}
                ),
            }
        )
        try:
            payload = self._http.upload(
                f"/api/v1/projects/{project_id}/artifacts/{series_id}/versions?{query}",
                token=self.auth.access_token,
                source=path,
                headers={
                    "Idempotency-Key": idempotency_key,
                    "If-Match": str(
                        self.artifact_stream_revision(project_id, series_id)
                    ),
                    **(
                        {"X-Kraken-Role": str(self.acting_role)}
                        if getattr(self, "acting_role", None)
                        else {}
                    ),
                },
            )
        except RemoteServerError as exc:
            if exc.status == 409 and exc.code == "revision.conflict":
                wake = ProjectEventWake(
                    project_id=str(project_id),
                    event_type="revision.conflict",
                    event_id="",
                    revision=(
                        None
                        if exc.problem.get("actual_revision") is None
                        else int(exc.problem["actual_revision"])
                    ),
                    entity_kind=str(exc.problem.get("entity_kind", "artifact_series")),
                    entity_id=str(exc.problem.get("entity_id", series_id)),
                )
                for handler in list(self._wake_handlers):
                    handler(wake)
                raise RemoteServerError(
                    "Другой пользователь уже сохранил изменение. Экран обновлён; повторите текущую операцию.",
                    status=exc.status,
                    body=exc.body,
                    problem=exc.problem,
                ) from exc
            raise
        return decode_model(ArtifactVersion, payload)

    def add_managed_artifact_version(
        self,
        *,
        principal: Principal,
        project_id: object,
        series_id: object,
        source: Path | str,
        parent_version_id: object | None = None,
        idempotency_key: str,
    ) -> ArtifactVersion:
        path = Path(source).resolve(strict=True)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        revision = self.artifact_stream_revision(project_id, series_id)
        artifact_payload: dict[str, object] = {
            "filename": path.name,
            "media_type": artifact_media_type(path.name),
            "sha256": digest.hexdigest(),
            "size_bytes": path.stat().st_size,
            **({} if parent_version_id is None else {"parent_version_id": str(parent_version_id)}),
        }
        try:
            transfer = self._call(
                "POST",
                f"/api/v1/projects/{project_id}/artifacts/{series_id}/uploads",
                payload=artifact_payload,
                idempotency_key=idempotency_key,
                if_match=revision,
            )
        except RemoteServerError as exc:
            if exc.status != 404:
                raise
            transfer = {"mode": "proxy"}
        mode = str(transfer.get("mode", "")) if isinstance(transfer, Mapping) else ""
        if mode == "completed" and isinstance(transfer.get("artifact"), Mapping):
            return decode_model(ArtifactVersion, transfer["artifact"])
        if mode == "gateway":
            self._http.upload_url(str(transfer["url"]), token=str(transfer["token"]), source=path)
            payload = self._call(
                "POST",
                f"/api/v1/projects/{project_id}/artifacts/{series_id}/uploads/complete",
                payload={**artifact_payload, "upload_token": str(transfer["token"])},
                idempotency_key=idempotency_key,
                if_match=revision,
            )
            return decode_model(ArtifactVersion, payload)
        if mode == "proxy":
            return self._add_managed_artifact_version_proxy(
                principal=principal,
                project_id=project_id,
                series_id=series_id,
                source=path,
                parent_version_id=parent_version_id,
                idempotency_key=idempotency_key,
            )
        raise RemoteServerError("Kraken Server returned an unsupported blob transfer mode")
