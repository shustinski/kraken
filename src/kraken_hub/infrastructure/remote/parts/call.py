"""remote_client: call."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from kraken_hub.workspace_service import (
    ProjectEventWake,
)

from kraken_hub.infrastructure.remote.parts.extract_review_archive import RemoteServerError


class CallMixin:
    def _call(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, object] | None = None,
        idempotency_key: str | None = None,
        if_match: int | None = None,
        timeout: float | None = None,
    ) -> Any:
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if if_match is not None:
            headers["If-Match"] = str(if_match)
        acting_role = getattr(self, "acting_role", None)
        if acting_role:
            headers["X-Kraken-Role"] = str(acting_role)
        try:
            request_arguments: dict[str, Any] = {
                "token": self.auth.access_token,
                "payload": payload,
                "headers": headers,
            }
            if timeout is not None:
                request_arguments["timeout"] = timeout
            return self._http.request(method, path, **request_arguments)
        except RemoteServerError as exc:
            if exc.status == 409 and exc.code == "revision.conflict":
                project_id = ""
                parts = [part for part in path.split("?")[0].split("/") if part]
                if "projects" in parts and parts.index("projects") + 1 < len(parts):
                    project_id = parts[parts.index("projects") + 1]
                wake = ProjectEventWake(
                    project_id=project_id,
                    event_type="revision.conflict",
                    event_id="",
                    revision=(
                        None
                        if exc.problem.get("actual_revision") is None
                        else int(exc.problem["actual_revision"])
                    ),
                    entity_kind=str(exc.problem.get("entity_kind", "")),
                    entity_id=str(exc.problem.get("entity_id", "")),
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
