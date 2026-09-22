"""Project ACL commands with an independent optimistic event stream."""

from __future__ import annotations

from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import AssignProjectRoleCommand, RevokeProjectRoleCommand
from kraken_manager.application.errors import AuthorizationError, ConcurrencyError, ConflictError, NotFoundError
from kraken_manager.application.ports import Clock, StorageProfileCatalog, UnitOfWorkFactory
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.common import PrincipalId
from kraken_manager.domain.identity import Permission, ProjectRole, ProjectRoleAssignment, SystemRole
from kraken_manager.domain.roles import CATALOG, cascade_lost_assignments


def _stream(command: AssignProjectRoleCommand | RevokeProjectRoleCommand) -> str:
    return f"acl:{command.project_id}:{command.principal_id}"


class _AclHandler:
    event_type: str

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        storage_profiles: StorageProfileCatalog,
        clock: Clock,
        authorization: AuthorizationPolicy | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._profiles = storage_profiles
        self._clock = clock
        self._authorization = authorization or AuthorizationPolicy()

    def _apply(self, uow: object, command: object, now: object) -> None:
        raise NotImplementedError

    def __call__(
        self, command: AssignProjectRoleCommand | RevokeProjectRoleCommand
    ) -> frozenset[ProjectRole]:
        with self._uow_factory() as uow:
            project = uow.projections.get_project(command.project_id)
            if project is None:
                raise NotFoundError(f"Project {command.project_id} was not found")
            target = uow.identities.get(command.principal_id)
            if target is None or not target.active:
                raise NotFoundError(f"Principal {command.principal_id} was not found or is inactive")
            if any(
                event.event_type == self.event_type
                for event in uow.event_store.find_by_idempotency_key(
                    project.id, command.context.idempotency_key
                )
            ):
                return uow.acl.roles_for(project.id, command.principal_id)
            profile = self._profiles.get(project.storage_profile)
            if profile is None:
                raise NotFoundError(f"Storage profile {project.storage_profile!r} was not found")
            self._authorization.decide(
                principal=command.context.actor,
                storage=profile,
                permission=Permission.MANAGE_ACL,
                roles=uow.acl.roles_for(project.id, command.context.actor.id),
                gitlab_identity_verified=command.context.gitlab_identity_verified,
            ).require()
            stream_id = _stream(command)
            revision = uow.event_store.current_revision(stream_id)
            if revision != command.expected_revision:
                raise ConcurrencyError(
                    f"Expected ACL revision {command.expected_revision}, found {revision}"
                )
            current = uow.acl.roles_for(project.id, command.principal_id)
            if self.event_type == "ProjectRoleAssigned" and command.role in current:
                raise ConflictError("The principal already has this project role")
            if self.event_type == "ProjectRoleRevoked" and command.role not in current:
                raise ConflictError("The principal does not have this project role")
            actor_roles = uow.acl.roles_for(project.id, command.context.actor.id)
            server_admin = SystemRole.SERVER_ADMIN in command.context.actor.system_roles
            if not CATALOG.can_grant(actor_roles, command.role.value, is_server_admin=server_admin):
                raise AuthorizationError(
                    f"Недостаточно прав: нельзя изменить роль «{CATALOG.roles[command.role.value].title}»."
                )
            if self.event_type == "ProjectRoleRevoked" and command.role is ProjectRole.MAINTAINER:
                list_assignments = getattr(uow.acl, "assignments_for", None)
                if list_assignments is not None:
                    maintainers = {
                        assignment.principal_id
                        for assignment in list_assignments(project.id)
                        if assignment.role is ProjectRole.MAINTAINER and assignment.active
                    }
                    actor_is_admin = server_admin or ProjectRole.ADMIN in actor_roles
                    if command.principal_id in maintainers and len(maintainers) <= 1 and not actor_is_admin:
                        raise AuthorizationError("Нельзя отозвать роль последнего сопровождающего проекта.")
            now = self._clock.now()
            event = EventEnvelope.create(
                stream_id=stream_id,
                project_id=project.id,
                revision=revision + 1,
                event_type=self.event_type,
                payload={
                    "principal_id": str(command.principal_id),
                    "role": command.role.value,
                    "assigned_by": str(command.context.actor.id),
                    "changed_at": now.isoformat(),
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=command.context.performer_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(stream_id, expected_revision=revision, events=(event,))
            self._apply(uow, command, now)
            uow.commit()
            return uow.acl.roles_for(project.id, command.principal_id)


class AssignProjectRoleHandler(_AclHandler):
    event_type = "ProjectRoleAssigned"

    def _apply(self, uow: object, command: AssignProjectRoleCommand, now: object) -> None:
        uow.acl.assign(
            ProjectRoleAssignment.create(
                project_id=command.project_id,
                principal_id=command.principal_id,
                role=command.role,
                assigned_by=command.context.actor.id,
                assigned_at=now,
            )
        )


class RevokeProjectRoleHandler(_AclHandler):
    event_type = "ProjectRoleRevoked"

    def _apply(self, uow: object, command: RevokeProjectRoleCommand, now: object) -> None:
        list_assignments = getattr(uow.acl, "assignments_for", None)
        snapshot = () if list_assignments is None else tuple(list_assignments(command.project_id))
        uow.acl.revoke(command.project_id, command.principal_id, command.role)
        if list_assignments is None:
            return
        target = uow.identities.get(command.principal_id)
        remaining = uow.acl.roles_for(command.project_id, command.principal_id)
        lost = cascade_lost_assignments(
            tuple(
                (str(assignment.principal_id), assignment.role.value, str(assignment.assigned_by))
                for assignment in snapshot
                if assignment.active
            ),
            principal_id=str(command.principal_id),
            revoked_role=command.role.value,
            remaining_roles={role.value for role in remaining},
            grantor_is_admin=target is not None and SystemRole.SERVER_ADMIN in target.system_roles,
            catalog=CATALOG,
        )
        for holder_id, role_id in lost:
            holder = PrincipalId(holder_id)
            role = ProjectRole(role_id)
            uow.acl.revoke(command.project_id, holder, role)
            stream_id = f"acl:{command.project_id}:{holder}"
            revision = uow.event_store.current_revision(stream_id)
            event = EventEnvelope.create(
                stream_id=stream_id,
                project_id=command.project_id,
                revision=revision + 1,
                event_type="ProjectRoleRevoked",
                payload={
                    "principal_id": holder_id,
                    "role": role_id,
                    "assigned_by": str(command.context.actor.id),
                    "changed_at": now.isoformat(),
                    "cascade": True,
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=command.context.performer_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=f"{command.context.idempotency_key}:cascade:{holder_id}:{role_id}",
            )
            uow.event_store.append(stream_id, expected_revision=revision, events=(event,))


__all__ = ["AssignProjectRoleHandler", "RevokeProjectRoleHandler"]
