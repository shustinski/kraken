"""Extensible project-role catalog, grant rules, and access resolution.

Roles and actions are data. A deployment can register another role or open an
action to every role without editing the authorization policy. Capability
inheritance (``includes``) is separate from the right to assign a role
(``grants``): elementer can do a corrector's work and still cannot appoint one.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

from kraken_manager.domain.identity import Permission

_LEGACY_ROLE_IDS = {
    "owner": "maintainer",
    "manager": "maintainer",
    "contributor": "elementer",
    "reviewer": "corrector",
}

_acting_role: ContextVar[str | None] = ContextVar("kraken_acting_role", default=None)


def set_acting_role(role: str | None) -> Token[str | None]:
    text = str(role or "").strip()
    return _acting_role.set(text or None)


def reset_acting_role(token: Token[str | None]) -> None:
    _acting_role.reset(token)


def current_acting_role() -> str | None:
    return _acting_role.get()


def bind_acting_role(service: object, role: str | None) -> None:
    """Remember the role the desktop session sends and the local policy reads."""

    text = str(role or "").strip() or None
    set_acting_role(text)
    remote = getattr(service, "remote", None)
    if remote is not None:
        remote.acting_role = text


def parse_role_id(value: object) -> str:
    text = str(getattr(value, "value", value) or "").strip().lower()
    return _LEGACY_ROLE_IDS.get(text, text)


@dataclass(frozen=True, slots=True)
class RoleDefinition:
    role_id: str
    title: str
    includes: frozenset[str] = frozenset()
    grants: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ActionDefinition:
    action_id: str
    title: str
    # None means every role. Otherwise only these roles, plus roles that include them.
    roles: frozenset[str] | None = None


@dataclass(frozen=True, slots=True)
class AccessDecision:
    allowed: bool
    message: str


@dataclass(frozen=True, slots=True)
class RoleCatalog:
    roles: dict[str, RoleDefinition]
    actions: dict[str, ActionDefinition]
    direct_permissions: dict[str, frozenset[str]]

    def register_role(self, definition: RoleDefinition) -> None:
        role_id = parse_role_id(definition.role_id)
        self.roles[role_id] = RoleDefinition(
            role_id,
            definition.title,
            frozenset(parse_role_id(item) for item in definition.includes),
            frozenset(parse_role_id(item) for item in definition.grants),
        )

    def register_action(self, definition: ActionDefinition) -> None:
        roles = None if definition.roles is None else frozenset(parse_role_id(item) for item in definition.roles)
        self.actions[definition.action_id] = ActionDefinition(definition.action_id, definition.title, roles)

    def closure(self, role_ids: frozenset[str] | set[str]) -> frozenset[str]:
        pending = [parse_role_id(role_id) for role_id in role_ids]
        seen: set[str] = set()
        while pending:
            role_id = pending.pop()
            if role_id in seen:
                continue
            seen.add(role_id)
            definition = self.roles.get(role_id)
            if definition is not None:
                pending.extend(definition.includes)
        return frozenset(seen)

    def permission_names(self, role_id: str) -> frozenset[str]:
        names: set[str] = set()
        for included in self.closure({role_id}):
            names.update(self.direct_permissions.get(included, ()))
        return frozenset(names)

    def can_grant(
        self,
        held_roles: frozenset[str] | set[str],
        target_role: str,
        *,
        is_server_admin: bool = False,
    ) -> bool:
        target = parse_role_id(target_role)
        if target not in self.roles:
            return False
        if is_server_admin or "admin" in {parse_role_id(role) for role in held_roles}:
            return True
        granted: set[str] = set()
        for role_id in held_roles:
            definition = self.roles.get(parse_role_id(role_id))
            if definition is not None:
                granted.update(definition.grants)
        return target in granted

    def resolve(
        self,
        *,
        held_roles: frozenset[str] | set[str],
        acting_role: str,
        action: str,
        is_server_admin: bool = False,
    ) -> AccessDecision:
        acting = parse_role_id(acting_role)
        held = {parse_role_id(role) for role in held_roles}
        definition = self.roles.get(acting)
        title = definition.title if definition is not None else acting
        if acting not in self.roles:
            return AccessDecision(False, f"Недостаточно прав: неизвестная роль «{acting}».")
        holds_acting = acting in held or (acting == "admin" and is_server_admin)
        if not holds_acting:
            return AccessDecision(
                False,
                f"Недостаточно прав: роль «{title}» вам не назначена.",
            )
        if is_server_admin and acting == "admin" and action == "manage_acl":
            return AccessDecision(True, "Доступ разрешён.")
        action_definition = self.actions.get(action)
        if action_definition is not None and action_definition.roles is None:
            return AccessDecision(True, "Доступ разрешён.")
        if action_definition is not None and action_definition.roles is not None:
            allowed_roles = action_definition.roles
        else:
            allowed_roles = frozenset(
                role_id
                for role_id, names in self.direct_permissions.items()
                if action in names
            )
        if self.closure({acting}) & set(allowed_roles):
            return AccessDecision(True, "Доступ разрешён.")
        action_title = action_definition.title if action_definition is not None else _ACTION_TITLES.get(action, action)
        return AccessDecision(
            False,
            f"Недостаточно прав: роль «{title}» не может выполнить действие «{action_title}».",
        )


def cascade_lost_assignments(
    assignments: tuple[tuple[str, str, str], ...] | list[tuple[str, str, str]],
    *,
    principal_id: str,
    revoked_role: str,
    remaining_roles: frozenset[str] | set[str],
    grantor_is_admin: bool,
    catalog: RoleCatalog,
) -> tuple[tuple[str, str], ...]:
    """Roles that fall when ``revoked_role`` is taken from ``principal_id``.

    ``assignments`` are ``(principal_id, role, assigned_by)`` for one project.
    People appointed by a maintainer lose those roles when that maintainer role
    is revoked and the maintainer has no remaining role that can grant access.
    The same rule repeats down the chain.
    """

    revoked = parse_role_id(revoked_role)
    definition = catalog.roles.get(revoked)
    if definition is None or not definition.grants:
        return ()
    remaining = {parse_role_id(role) for role in remaining_roles}
    if grantor_is_admin or "admin" in remaining or any(catalog.roles.get(role) and catalog.roles[role].grants for role in remaining):
        return ()
    active = {
        (holder, parse_role_id(role), assigned_by)
        for holder, role, assigned_by in assignments
        if (holder, parse_role_id(role)) != (principal_id, revoked)
    }
    removed = {(principal_id, revoked)}
    lost: list[tuple[str, str]] = []
    queue = [principal_id]
    while queue:
        grantor = queue.pop()
        grantor_roles = {role for holder, role, _assigned_by in active if holder == grantor and (holder, role) not in removed}
        if grantor != principal_id and any(catalog.roles.get(role) and catalog.roles[role].grants for role in grantor_roles):
            continue
        for holder, role, assigned_by in list(active):
            if assigned_by != grantor or (holder, role) in removed:
                continue
            removed.add((holder, role))
            lost.append((holder, role))
            queue.append(holder)
    return tuple(lost)


_ACTION_TITLES = {
    "view_project": "просмотр проекта",
    "view_history": "просмотр истории",
    "export_statistics": "экспорт статистики",
    "rename_project": "переименование проекта",
    "archive_project": "архивация проекта",
    "migrate_project": "перенос проекта",
    "manage_acl": "управление ролями",
    "manage_structure": "изменение структуры",
    "assign_work": "назначение работы",
    "manage_review": "ведение проверки",
    "accept_review": "приёмка проверки",
    "import_artifact": "импорт данных",
    "run_plugin": "запуск обработки",
    "add_note": "заметки",
    "return_review": "возврат на доработку",
}


def action_for_request(method: str, path: str) -> str:
    if method.upper() == "POST" and path.rstrip("/").endswith("/projects"):
        return ""
    rules = (
        ("/deletion-requests", "archive_project"),
        ("/acl/", "manage_acl"),
        ("/layers", "manage_structure"),
        ("/representations", "manage_structure"),
        ("/rename", "rename_project"),
        ("/archive", "archive_project"),
        ("/restore", "archive_project"),
        ("/review", "manage_review"),
        ("/plugin", "run_plugin"),
        ("/jobs", "run_plugin"),
        ("/pipeline", "run_plugin"),
        ("/import", "import_artifact"),
        ("/notes", "add_note"),
        ("/artifacts", "import_artifact"),
        ("/analyses", "run_plugin"),
    )
    for fragment, action in rules:
        if fragment in path:
            return action
    return "manage_structure"


def default_catalog() -> RoleCatalog:
    view = frozenset({"view_project", "view_history", "export_statistics"})
    catalog = RoleCatalog(roles={}, actions={}, direct_permissions={})
    catalog.register_role(RoleDefinition("viewer", "Наблюдатель"))
    catalog.register_role(RoleDefinition("corrector", "Корректор", includes=frozenset({"viewer"})))
    catalog.register_role(RoleDefinition("elementer", "Элементщик", includes=frozenset({"corrector"})))
    catalog.register_role(RoleDefinition("sewer", "Сшивальщик", includes=frozenset({"viewer"})))
    catalog.register_role(
        RoleDefinition(
            "maintainer",
            "Сопровождающий",
            includes=frozenset({"elementer", "sewer", "viewer"}),
            grants=frozenset({"maintainer", "sewer", "corrector", "elementer", "viewer"}),
        )
    )
    catalog.register_role(
        RoleDefinition(
            "admin",
            "Администратор",
            includes=frozenset({"maintainer"}),
            grants=frozenset({"admin", "maintainer", "sewer", "corrector", "elementer", "viewer"}),
        )
    )
    catalog.direct_permissions.update(
        {
            "viewer": view,
            "corrector": frozenset({"add_note", "return_review"}),
            "elementer": frozenset({"import_artifact", "run_plugin"}),
            "sewer": frozenset({"import_artifact", "run_plugin", "add_note"}),
            "maintainer": frozenset(
                {
                    "manage_structure",
                    "assign_work",
                    "manage_review",
                    "accept_review",
                    "rename_project",
                    "manage_acl",
                    "archive_project",
                    "migrate_project",
                    "import_artifact",
                    "run_plugin",
                    "add_note",
                }
            ),
            "admin": frozenset(permission.value for permission in Permission),
        }
    )
    return catalog


CATALOG = default_catalog()


__all__ = [
    "CATALOG",
    "AccessDecision",
    "ActionDefinition",
    "RoleCatalog",
    "RoleDefinition",
    "action_for_request",
    "bind_acting_role",
    "cascade_lost_assignments",
    "current_acting_role",
    "default_catalog",
    "parse_role_id",
    "reset_acting_role",
    "set_acting_role",
]
