"""remote_client: create artifact series."""
from __future__ import annotations
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from kraken_manager.domain.artifacts import (
    ArtifactScope,
    ArtifactSeries,
    ArtifactVersion,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.infrastructure.filesystem._codec import decode_model

from kraken_hub.infrastructure.remote.parts.extract_review_archive import RemoteServerError


class CreateArtifactSeriesMixin:
    def list_artifact_series(
        self,
        project_id: object,
        *,
        layer_id: object | None = None,
        representation_id: object | None = None,
        include_archived: bool = False,
    ) -> tuple[ArtifactSeries, ...]:
        query = urllib.parse.urlencode(
            {
                key: value
                for key, value in {
                    "layer_id": None if layer_id is None else str(layer_id),
                    "representation_id": None
                    if representation_id is None
                    else str(representation_id),
                    "include_archived": "true" if include_archived else None,
                }.items()
                if value is not None
            }
        )
        payload = self._call("GET", f"/api/v1/projects/{project_id}/artifacts?{query}")
        items = payload.get("items", ()) if isinstance(payload, Mapping) else ()
        return tuple(
            decode_model(ArtifactSeries, item)
            for item in items
            if isinstance(item, Mapping)
        )

    def create_artifact_series(
        self,
        *,
        principal: Principal,
        project_id: object,
        scope: ArtifactScope,
        name: str,
        layer_id: object | None = None,
        representation_id: object | None = None,
        frame_id: object | None = None,
        series_id: object | None = None,
        idempotency_key: str,
    ) -> ArtifactSeries:
        del principal
        payload = self._call(
            "POST",
            f"/api/v1/projects/{project_id}/artifacts",
            payload={
                "scope": scope.value,
                "name": name,
                "layer_id": None if layer_id is None else str(layer_id),
                "representation_id": None
                if representation_id is None
                else str(representation_id),
                "frame_id": None if frame_id is None else str(frame_id),
                "series_id": None if series_id is None else str(series_id),
            },
            idempotency_key=idempotency_key,
        )
        return decode_model(ArtifactSeries, payload)

    def artifact_stream_revision(self, project_id: object, series_id: object) -> int:
        payload = self._call(
            "GET", f"/api/v1/projects/{project_id}/artifacts/{series_id}/revision"
        )
        return int(payload.get("revision", 0)) if isinstance(payload, Mapping) else 0

    def artifact_versions(
        self, project_id: object, series_id: object
    ) -> tuple[ArtifactVersion, ...]:
        payload = self._call(
            "GET", f"/api/v1/projects/{project_id}/artifacts/{series_id}/versions"
        )
        items = payload.get("items", ()) if isinstance(payload, Mapping) else ()
        return tuple(
            decode_model(ArtifactVersion, item)
            for item in items
            if isinstance(item, Mapping)
        )

    def active_artifact_version(
        self, project_id: object, series_id: object
    ) -> ArtifactVersion | None:
        payload = self._call(
            "GET", f"/api/v1/projects/{project_id}/artifacts/{series_id}/active"
        )
        item = payload.get("item") if isinstance(payload, Mapping) else None
        return (
            decode_model(ArtifactVersion, item) if isinstance(item, Mapping) else None
        )

    def artifact_version(
        self, project_id: object, version_id: object
    ) -> ArtifactVersion | None:
        try:
            payload = self._call(
                "GET",
                f"/api/v1/projects/{project_id}/artifacts/versions/{version_id}/metadata",
            )
        except RemoteServerError as exc:
            if exc.status == 404:
                return None
            raise
        return (
            decode_model(ArtifactVersion, payload)
            if isinstance(payload, Mapping)
            else None
        )
