"""Project ACL commands with an independent optimistic event stream."""

from __future__ import annotations

from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import AssignProjectRoleCommand, RevokeProjectRoleCommand
from kraken_manager.application.errors import ConcurrencyError, ConflictError, NotFoundError
from kraken_manager.application.ports import Clock, StorageProfileCatalog, UnitOfWorkFactory
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission, ProjectRole, ProjectRoleAssignment


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
        del now
        uow.acl.revoke(command.project_id, command.principal_id, command.role)


__all__ = ["AssignProjectRoleHandler", "RevokeProjectRoleHandler"]
