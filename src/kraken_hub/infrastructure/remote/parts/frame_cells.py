"""remote_client: frame cells."""
from __future__ import annotations
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from kraken_manager.application.results import FrameCellSnapshot, FrameManagementState


class FrameCellsMixin:
    def frame_cells(
        self,
        project_id: object,
        layer_id: object,
        representation_id: object,
        *,
        as_of: datetime | None = None,
        bounds: tuple[int, int, int, int] | None = None,
    ) -> tuple[FrameCellSnapshot, ...]:
        query = urllib.parse.urlencode(
            {
                "layer_id": str(layer_id),
                "representation_id": str(representation_id),
                **({} if as_of is None else {"as_of": as_of.isoformat()}),
                **({} if bounds is None else dict(zip(("x1", "y1", "x2", "y2"), bounds, strict=True))),
            }
        )
        payload = self._call("GET", f"/api/v1/projects/{project_id}/frames?{query}")
        return tuple(
            FrameCellSnapshot(
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
            and (bounds is None or bounds[0] <= int(item["x"]) <= bounds[2]
                 and bounds[1] <= int(item["y"]) <= bounds[3])
        )

    def frame_management_states(
        self,
        project_id: object,
        layer_id: object,
        representation_id: object,
        *,
        as_of: datetime | None = None,
    ) -> tuple[FrameManagementState, ...]:
        return tuple(
            FrameManagementState(
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
