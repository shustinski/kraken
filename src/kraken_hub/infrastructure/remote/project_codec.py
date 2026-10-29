"""Translate Kraken Server JSON into application/domain values."""
from collections.abc import Mapping
from datetime import datetime
from kraken_hub.application.workspace import REMOTE_STORAGE_PROFILE
from kraken_manager.domain.project import GridOrientation, Project, ProjectState


def project_from_server_dict(payload: Mapping[str, object]) -> Project:
    created = payload.get("created_at")
    if isinstance(created, str):
        created_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
    elif isinstance(created, datetime):
        created_at = created
    else:
        created_at = datetime.now().astimezone()
    return Project(
        id=str(payload.get("project_id") or payload.get("id")),
        name=str(payload["name"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        orientation=GridOrientation(str(payload.get("orientation", GridOrientation.Y_DOWN.value))),
        storage_profile=str(payload.get("storage_profile") or REMOTE_STORAGE_PROFILE.id),
        state=ProjectState(str(payload.get("state", ProjectState.ACTIVE.value))),
        revision=int(payload.get("revision", 0)),
        created_at=created_at,
    )

