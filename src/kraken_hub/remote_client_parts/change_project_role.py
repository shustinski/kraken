"""remote_client: change project role."""
from __future__ import annotations
from collections.abc import Mapping
from kraken_manager.domain.identity import (
    Permission,
    Principal,
    ProjectRole,
    parse_project_role,
    permissions_for_roles,
)
from kraken_manager.domain.project import (
    Layer,
    Project,
    Representation,
)

from kraken_hub.remote_client_parts.extract_review_archive import RemoteServerError


class ChangeProjectRoleMixin:
    def activate_representation(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
        idempotency_key: str,
    ) -> Representation:
        del principal
        return self._update_representation(
            project=project,
            layer=layer,
            representation=representation,
            operation={"active": True},
            idempotency_key=idempotency_key,
        )

    def deactivate_representation(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
        idempotency_key: str,
    ) -> Representation:
        del principal
        return self._update_representation(
            project=project,
            layer=layer,
            representation=representation,
            operation={"active": False},
            idempotency_key=idempotency_key,
        )

    def archive_representation(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
        idempotency_key: str,
    ) -> Representation:
        del principal
        return self._update_representation(
            project=project,
            layer=layer,
            representation=representation,
            operation={"archive": True},
            idempotency_key=idempotency_key,
        )

    def project_permissions(
        self, project_id: object, principal: Principal
    ) -> frozenset[Permission]:
        try:
            payload = self._call(
                "GET",
                f"/api/v1/projects/{project_id}/acl/{principal.id}",
            )
        except RemoteServerError:
            return frozenset()
        roles_raw = payload.get("roles", ()) if isinstance(payload, Mapping) else ()
        roles: set[ProjectRole] = set()
        for role_name in roles_raw:
            try:
                roles.add(parse_project_role(role_name))
            except ValueError:
                continue
        return permissions_for_roles(roles)

    def project_roles(
        self,
        project_id: object,
        principal_id: object,
    ) -> frozenset[ProjectRole]:
        payload = self._call(
            "GET",
            f"/api/v1/projects/{project_id}/acl/{principal_id}",
        )
        raw_roles = payload.get("roles", ()) if isinstance(payload, Mapping) else ()
        return frozenset(parse_project_role(role) for role in raw_roles)

    def project_role_revision(
        self,
        project_id: object,
        principal_id: object,
    ) -> int:
        payload = self._call(
            "GET",
            f"/api/v1/projects/{project_id}/acl/{principal_id}",
        )
        return int(payload.get("revision", 0)) if isinstance(payload, Mapping) else 0

    def request_project_deletion(self, *, project: Project, reason: str = "", **_kwargs):
        return self._call(
            "POST",
            f"/api/v1/projects/{project.id}/deletion-requests",
            payload={"reason": reason},
        )

    def project_deletion_requests(self, project_id: object):
        return self._call("GET", f"/api/v1/projects/{project_id}/deletion-requests").get("items", [])

    def _change_project_role(
        self,
        *,
        method: str,
        project: Project,
        target_principal_id: object,
        role: ProjectRole,
        expected_revision: int,
        idempotency_key: str,
    ) -> frozenset[ProjectRole]:
        payload = self._call(
            method,
            f"/api/v1/projects/{project.id}/acl/{target_principal_id}/{role.value}",
            payload={} if method == "PUT" else None,
            idempotency_key=idempotency_key,
            if_match=expected_revision,
        )
        raw_roles = payload.get("roles", ()) if isinstance(payload, Mapping) else ()
        return frozenset(parse_project_role(value) for value in raw_roles)

    def assign_project_role(
        self,
        *,
        principal: Principal,
        project: Project,
        target_principal_id: object,
        role: ProjectRole,
        expected_revision: int,
        idempotency_key: str,
    ) -> frozenset[ProjectRole]:
        del principal
        return self._change_project_role(
            method="PUT",
            project=project,
            target_principal_id=target_principal_id,
            role=role,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
        )

    def revoke_project_role(
        self,
        *,
        principal: Principal,
        project: Project,
        target_principal_id: object,
        role: ProjectRole,
        expected_revision: int,
        idempotency_key: str,
    ) -> frozenset[ProjectRole]:
        del principal
        return self._change_project_role(
            method="DELETE",
            project=project,
            target_principal_id=target_principal_id,
            role=role,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
        )
