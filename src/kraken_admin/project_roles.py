"""Atomic local administration of a project's role assignments."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from kraken_manager.application.acl import AssignProjectRoleHandler, RevokeProjectRoleHandler
from kraken_manager.application.dto import AssignProjectRoleCommand, CommandContext, RevokeProjectRoleCommand
from kraken_manager.application.errors import ConcurrencyError
from kraken_manager.domain.common import PrincipalId, ProjectId
from kraken_manager.domain.identity import Principal, ProjectRole, SystemRole
from kraken_manager.domain.roles import CATALOG, cascade_lost_assignments


def administrator(accounts) -> Principal:
    from .local_admin import _local_administrator

    account = _local_administrator(accounts)
    return replace(
        Principal.local(subject=account.username, display_name=account.display_name, principal_id=account.account_id),
        system_roles=frozenset({SystemRole.SERVER_ADMIN}),
    )


def snapshot(assignments) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted(
        (str(item.principal_id), item.role.value, str(item.assigned_by)) for item in assignments if item.active
    ))


def cascade_preview(assignments, principal: Principal, desired: frozenset[ProjectRole]):
    active = list(assignments)
    current = {role for holder, role, _grantor in active if holder == str(principal.id)}
    # Grants happen first, so replacement of a grant-capable role preserves descendants.
    for role in desired:
        if role.value not in current:
            active.append((str(principal.id), role.value, "local-admin"))
    lost = set()
    for role in sorted(current - {item.value for item in desired}):
        active = [entry for entry in active if entry[:2] != (str(principal.id), role)]
        removed = cascade_lost_assignments(
            active, principal_id=str(principal.id), revoked_role=role,
            remaining_roles={r for holder, r, _ in active if holder == str(principal.id)},
            grantor_is_admin=SystemRole.SERVER_ADMIN in principal.system_roles, catalog=CATALOG,
        )
        lost.update(removed)
        active = [entry for entry in active if entry[:2] not in lost]
    return tuple(sorted(lost))


def set_roles(services, accounts, *, project_id: str, principal: Principal,
              desired: frozenset[ProjectRole], expected_revision: int, expected_snapshot) -> None:
    actor = administrator(accounts)
    identifier = ProjectId(project_id)
    with services.uow_factory() as uow:
        uow.lock_project_acl(identifier)
        if snapshot(uow.acl.assignments_for(identifier)) != expected_snapshot:
            raise ConcurrencyError("Назначения изменились. Обновите таблицу и повторите действие.")
        stream = f"acl:{identifier}:{principal.id}"
        if uow.event_store.current_revision(stream) != expected_revision:
            raise ConcurrencyError("Роли пользователя изменились. Обновите таблицу.")
        if uow.identities.get(principal.id) is None:
            uow.identities.save(principal)
        current = uow.acl.roles_for(identifier, principal.id)
        for enabled, roles in ((True, desired - current), (False, current - desired)):
            handler_type = AssignProjectRoleHandler if enabled else RevokeProjectRoleHandler
            command_type = AssignProjectRoleCommand if enabled else RevokeProjectRoleCommand
            handler = handler_type(services.uow_factory, services.profiles, services.clock, allow_inactive=True)
            for role in sorted(roles, key=lambda item: item.value):
                handler(command_type(
                    context=CommandContext(actor=actor, idempotency_key=str(uuid4())),
                    project_id=identifier, principal_id=PrincipalId(str(principal.id)), role=role,
                    expected_revision=uow.event_store.current_revision(stream),
                ), transaction=uow)
        accounts._record_audit(uow._connection, None, "project.roles_changed", None, {
            "project_id": project_id, "principal_id": str(principal.id),
            "roles": sorted(role.value for role in desired),
        })
        uow.commit()


def all_principals(services, accounts) -> list[Principal]:
    people = {str(item.id): item for item in services.identities.list(include_inactive=True)}
    for account in accounts.list_accounts(include_disabled=True):
        people[str(account.account_id)] = replace(
            Principal.local(subject=account.username, display_name=account.display_name,
                            principal_id=account.account_id), active=account.enabled,
            system_roles=frozenset(SystemRole(role) for role in accounts.global_roles_for(account.account_id)),
        )
    return sorted(people.values(), key=lambda person: (person.display_name.casefold(), str(person.id)))
