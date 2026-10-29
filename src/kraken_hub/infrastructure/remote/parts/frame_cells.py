"""remote_client: frame cells."""
from __future__ import annotations
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from types import SimpleNamespace


class FrameCellsMixin:
    def frame_cells(
        self,
        project_id: object,
        layer_id: object,
        representation_id: object,
        *,
        as_of: datetime | None = None,
    ) -> tuple[object, ...]:
        if as_of is not None:
            raise ValueError("Temporal server matrix views are not supported by API v1")
        query = urllib.parse.urlencode(
            {
                "layer_id": str(layer_id),
                "representation_id": str(representation_id),
            }
        )
        payload = self._call("GET", f"/api/v1/projects/{project_id}/frames?{query}")
        return tuple(
            SimpleNamespace(
                x=int(item["x"]),
                y=int(item["y"]),
                status=str(item.get("status", "empty")),
                frame_id=str(item.get("frame_id", "")),
                artifact_version_id=str(item.get("artifact_version_id", "")),
                sha256=str(item.get("sha256", "")),
                modified_at=str(item.get("modified_at", "")),
                performer_color=str(item.get("performer_color", "")),
                performer_initials=str(item.get("performer_initials", "")),
                review_status=str(item.get("review_status", "not_checked")),
                quality=item.get("quality"),
            )
            for item in payload.get("items", ())
            if not item.get("missing")
        )

    def frame_management_states(
        self,
        project_id: object,
        layer_id: object,
        representation_id: object,
        *,
        as_of: datetime | None = None,
    ) -> tuple[object, ...]:
        return tuple(
            SimpleNamespace(
                frame_id=item.frame_id,
                artifact_version_id=item.artifact_version_id,
                modified_at=item.modified_at,
                performer_color=item.performer_color,
                performer_initials=item.performer_initials,
                review_status=item.review_status,
            )
            for item in self.frame_cells(
                project_id, layer_id, representation_id, as_of=as_of
            )
        )
