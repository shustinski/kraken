"""manager_app: manage project participants."""
from __future__ import annotations
from uuid import uuid4
from kraken_manager.domain.identity import ProjectRole
from kraken_manager.presentation.qt.models import ProjectListItem


class ManageProjectParticipantsMixin:
    def manage_project_participants(self, item: ProjectListItem | None) -> None:
        if item is None:
            return
        from kraken_manager.domain.identity import parse_project_role
        from kraken_manager.domain.roles import bind_acting_role
        from kraken_manager.presentation.qt.role_management import RoleManagementDialog

        project = self.service.get_project(item.project_id)
        if project is None:
            self._error("Проект больше не доступен.")
            return
        project_principals = getattr(self.service, "list_role_candidates", None)
        principals = list(
            project_principals(project.id)
            if callable(project_principals)
            else self.service.list_principals()
        )

        def roles_for(principal) -> frozenset[ProjectRole]:
            assigned = self.service.project_roles(project.id, principal.id)
            if isinstance(assigned, dict):
                return frozenset(parse_project_role(role) for role in assigned.get("roles", ()))
            return frozenset(assigned)

        def change(principal, role: ProjectRole, enabled: bool) -> None:
            revision = self.service.project_role_revision(project.id, principal.id)
            if enabled:
                self.service.assign_project_role(
                    principal=self.session.principal,
                    project=project,
                    target_principal_id=principal.id,
                    role=role,
                    expected_revision=revision,
                    idempotency_key=str(uuid4()),
                )
            else:
                self.service.revoke_project_role(
                    principal=self.session.principal,
                    project=project,
                    target_principal_id=principal.id,
                    role=role,
                    expected_revision=revision,
                    idempotency_key=str(uuid4()),
                )

        actor_roles = list(roles_for(self.session.principal))
        rank = {role.value: index for index, role in enumerate(ProjectRole)}
        actor_roles.sort(key=lambda role: rank.get(role.value, 100))

        def remember(role: ProjectRole) -> None:
            bind_acting_role(self.service, role.value)

        if actor_roles:
            remember(actor_roles[0])
        RoleManagementDialog(
            self.shell,
            project_name=project.name,
            principals=principals,
            roles_for=roles_for,
            change_role=change,
            acting_roles=actor_roles,
            on_acting_role=remember,
        ).exec()
