"""services: change project role."""
from __future__ import annotations
from typing import TYPE_CHECKING
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from kraken_server.services_parts.command_context import (
    ConflictError,
    ForbiddenError,
    ValidationError,
)


class ChangeProjectRoleMixin:
    def _change_project_role(
        self,
        project_id: str,
        principal_id: str,
        role: str,
        context: CommandContext,
        *,
        revoke: bool,
    ) -> dict[str, Any]:
        self.get_project(project_id)
        from kraken_manager.domain.roles import CATALOG, cascade_lost_assignments, parse_role_id

        role = parse_role_id(role)
        if role not in CATALOG.roles:
            raise ValidationError("Unsupported project role")
        actor_roles = self._acl.get((project_id, context.actor_id), set())
        if not CATALOG.can_grant(actor_roles, role):
            raise ForbiddenError(
                f"Недостаточно прав: нельзя изменить роль «{CATALOG.roles[role].title}»."
            )
        if revoke and role == "maintainer":
            maintainers = {
                principal
                for (candidate_project, principal), roles in self._acl.items()
                if candidate_project == project_id and "maintainer" in roles
            }
            if principal_id in maintainers and len(maintainers) <= 1:
                raise ForbiddenError("Нельзя отозвать роль последнего сопровождающего проекта.")
        operation = "revoke" if revoke else "assign"
        idempotency = (f"acl:{operation}:{project_id}:{principal_id}", context.idempotency_key)
        key = (project_id, principal_id)
        with self._lock:
            if idempotency in self._idempotency:
                return dict(self._idempotency[idempotency])
            revision = self._acl_revisions.get(key, 0)
            if context.expected_revision != revision:
                raise ConflictError("ACL revision changed")
            roles = self._acl.setdefault(key, set())
            if revoke and role not in roles:
                raise ConflictError("The principal does not have this role")
            if not revoke and role in roles:
                raise ConflictError("The principal already has this role")
            if revoke:
                roles.remove(role)
                self._granted_by.pop((project_id, principal_id, role), None)
                assignments = tuple(
                    (holder, granted_role, grantor)
                    for (candidate_project, holder, granted_role), grantor in self._granted_by.items()
                    if candidate_project == project_id and granted_role in self._acl.get((candidate_project, holder), set())
                )
                lost = cascade_lost_assignments(
                    assignments,
                    principal_id=principal_id,
                    revoked_role=role,
                    remaining_roles=roles,
                    grantor_is_admin=False,
                    catalog=CATALOG,
                )
                for holder, lost_role in lost:
                    holder_roles = self._acl.get((project_id, holder), set())
                    holder_roles.discard(lost_role)
                    self._granted_by.pop((project_id, holder, lost_role), None)
                    self._acl_revisions[(project_id, holder)] = self._acl_revisions.get((project_id, holder), 0) + 1
            else:
                roles.add(role)
                self._granted_by[(project_id, principal_id, role)] = context.actor_id
            revision += 1
            self._acl_revisions[key] = revision
            result = {
                "project_id": project_id,
                "principal_id": principal_id,
                "roles": sorted(roles),
                "revision": revision,
            }
            self._events[project_id].append(
                {
                    "event_id": str(uuid4()),
                    "event_type": "ProjectRoleRevoked" if revoke else "ProjectRoleAssigned",
                    "revision": revision,
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "actor_id": context.actor_id,
                    "payload": {"principal_id": principal_id, "role": role},
                }
            )
            self._idempotency[idempotency] = result
            return dict(result)

if TYPE_CHECKING:
    from kraken_server.services_parts.command_context import CommandContext
