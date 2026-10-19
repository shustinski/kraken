"""services: list representations."""
from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any

from kraken_server.services_parts.command_context import NotFoundError


class ListRepresentationsMixin:
    def assign_project_role(
        self, project_id: str, principal_id: str, role: str, context: CommandContext
    ) -> dict[str, Any]:
        return self._change_project_role(project_id, principal_id, role, context, revoke=False)

    def revoke_project_role(
        self, project_id: str, principal_id: str, role: str, context: CommandContext
    ) -> dict[str, Any]:
        return self._change_project_role(project_id, principal_id, role, context, revoke=True)

    def list_representations(
        self, project_id: str, layer_id: str, *, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        self.get_project(project_id)
        if layer_id not in self._representations:
            raise NotFoundError(layer_id)
        with self._lock:
            return [
                dict(item)
                for item in self._representations[layer_id]
                if include_archived or item["state"] != "archived"
            ]

if TYPE_CHECKING:
    from kraken_server.services_parts.command_context import CommandContext
