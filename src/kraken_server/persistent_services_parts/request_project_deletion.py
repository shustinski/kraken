"""persistent_services: request project deletion."""
from __future__ import annotations
from typing import Any
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_server.services import NotFoundError

from kraken_server.persistent_services_parts.layer_source_scan import _project_id


class RequestProjectDeletionMixin:
    def request_project_deletion(self, project_id: str, actor_id: str, reason: str = "") -> dict[str, Any]:
        from kraken_manager.domain.identity import Permission

        from kraken_server.project_deletion import DeletionRequests

        try:
            project = self.projections.get_project(_project_id(project_id))
            if project is None:
                raise NotFoundError(project_id)
            actor = self._actor(actor_id)
            AuthorizationPolicy().require(
                principal=actor, storage=self.profiles.get(project.storage_profile),
                permission=Permission.ARCHIVE_PROJECT,
                roles=self.identities.roles_for(project.id, actor.id), gitlab_identity_verified=True,
            )
            return DeletionRequests(self.engine).request(project_id, project.name, actor_id, reason)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def project_deletion_requests(self, project_id: str) -> list[dict[str, Any]]:
        from kraken_server.project_deletion import DeletionRequests

        return DeletionRequests(self.engine).list(project_id)
