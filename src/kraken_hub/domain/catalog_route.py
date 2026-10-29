"""Choose the catalog that owns one desktop call."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


LOCAL_STORAGE_LABEL = "Локальный файл"
REMOTE_STORAGE_LABEL = "Сервер PostgreSQL"

# These calls only read values already loaded on the project object or scan a
# folder the user picked. They do not open the local project database.
REMOTE_PROJECT_LOCAL_CALLS = frozenset({"plan_import_directory"})


def project_id_from_call(kwargs: Mapping[str, Any]) -> str | None:
    project = kwargs.get("project")
    project_identifier = getattr(project, "id", None)
    if project_identifier is not None:
        return str(project_identifier)
    project_id = kwargs.get("project_id")
    if project_id is not None:
        return str(project_id)
    for key in ("job", "batch", "layer", "series", "note"):
        value = kwargs.get(key)
        identifier = getattr(value, "project_id", None)
        if identifier is not None:
            return str(identifier)
    return None


def storage_label(*, remote: bool) -> str:
    if remote:
        return REMOTE_STORAGE_LABEL
    return LOCAL_STORAGE_LABEL


def call_uses_remote_catalog(
    *,
    project_id: str | None,
    known_remote: bool,
    remote_available: bool,
    operation: str,
) -> bool:
    return bool(
        project_id is not None
        and known_remote
        and remote_available
        and operation not in REMOTE_PROJECT_LOCAL_CALLS
    )
