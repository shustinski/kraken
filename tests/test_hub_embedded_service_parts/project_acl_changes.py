"""test_hub_embedded_service: test project acl changes."""
from __future__ import annotations
import tempfile
from pathlib import Path
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.domain.identity import ProjectRole
from kraken_manager.domain.project import GridOrientation


class ProjectAclChangesMixin:
    def test_project_acl_changes_are_optimistic_idempotent_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = EmbeddedProjectService(Path(temporary))
            owner = service.create_initial_account("owner", "Owner", "")
            service.accounts.create_account("worker", "Worker", "")
            worker = service.login("worker", "")
            assert worker is not None
            project = service.create_project(
                principal=owner.principal,
                name="ACL",
                width=1,
                height=1,
                orientation=GridOrientation.Y_DOWN,
                idempotency_key="project",
            )
            roles = service.assign_project_role(
                principal=owner.principal,
                project=project,
                target_principal_id=worker.principal.id,
                role=ProjectRole.ELEMENTER,
                expected_revision=0,
                idempotency_key="assign",
            )
            self.assertEqual(frozenset({ProjectRole.ELEMENTER}), roles)
            self.assertEqual(
                roles,
                service.assign_project_role(
                    principal=owner.principal,
                    project=project,
                    target_principal_id=worker.principal.id,
                    role=ProjectRole.ELEMENTER,
                    expected_revision=0,
                    idempotency_key="assign",
                ),
            )
            roles = service.revoke_project_role(
                principal=owner.principal,
                project=project,
                target_principal_id=worker.principal.id,
                role=ProjectRole.ELEMENTER,
                expected_revision=1,
                idempotency_key="revoke",
            )
            self.assertEqual(frozenset(), roles)
            event_types = {event.event_type for event in service.history(project.id)}
            self.assertTrue({"ProjectRoleAssigned", "ProjectRoleRevoked"}.issubset(event_types))

    def test_last_project_owner_cannot_be_revoked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = EmbeddedProjectService(Path(temporary))
            owner = service.create_initial_account("owner", "Owner", "")
            project = service.create_project(
                principal=owner.principal,
                name="Protected owner",
                width=1,
                height=1,
                orientation=GridOrientation.Y_DOWN,
                idempotency_key="project",
            )

            with self.assertRaisesRegex(ValueError, "последнего сопровождающего"):
                service.revoke_project_role(
                    principal=owner.principal,
                    project=project,
                    target_principal_id=owner.principal.id,
                    role=ProjectRole.MAINTAINER,
                    expected_revision=service.project_role_revision(
                        project.id,
                        owner.principal.id,
                    ),
                    idempotency_key="revoke-last-owner",
                )

            self.assertEqual(
                frozenset({ProjectRole.MAINTAINER}),
                service.project_roles(project.id, owner.principal.id),
            )
