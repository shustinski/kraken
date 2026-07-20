from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.application.dto import CommandContext, CreateProjectCommand
from kraken_manager.application.use_cases import CreateProjectHandler
from kraken_manager.domain.identity import ProjectRole
from kraken_manager.domain.project import GridOrientation
from kraken_manager.infrastructure.filesystem import LocalProjectUnitOfWorkFactory, SQLiteProjectionStore


class EmbeddedProjectServiceTests(unittest.TestCase):
    def test_manual_performer_catalog_is_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = EmbeddedProjectService(Path(directory))
            created = service.create_manual_performer(name="Reviewer", color="#123ABC")
            reopened = EmbeddedProjectService(Path(directory))
            self.assertEqual((created,), reopened.list_performers())

    def test_authenticated_local_project_vertical_slice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = EmbeddedProjectService(Path(temporary))
            service.accounts.create_account("operator", "Operator", "correct horse battery staple")
            session = service.login("operator", "correct horse battery staple")
            self.assertIsNotNone(session)
            assert session is not None
            project = service.create_project(
                principal=session.principal,
                name="Local chip",
                width=20,
                height=10,
                orientation=GridOrientation.Y_UP,
                idempotency_key="create-local-chip",
                layer_template=True,
            )
            self.assertEqual(200, project.frame_count)
            listed = service.list_projects()
            self.assertEqual((project.id,), tuple(item.id for item in listed))
            self.assertEqual(4, len(service.list_layers(project.id)))
            self.assertGreaterEqual(len(service.history(project.id)), 5)
            SQLiteProjectionStore(service.catalog_root, str(project.id)).destroy_cache()
            rebuilt = service.get_project(project.id)
            self.assertIsNotNone(rebuilt)
            self.assertEqual(project.name, rebuilt.name)
            self.assertEqual(4, len(service.list_layers(project.id)))

    def test_local_profile_rejects_more_than_one_hundred_thousand_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = EmbeddedProjectService(Path(temporary))
            account = service.accounts.create_account("operator", "Operator", "correct horse battery staple")
            session = service.login(account.username, "correct horse battery staple")
            assert session is not None
            with self.assertRaises(Exception):
                service.create_project(
                    principal=session.principal,
                    name="Too large",
                    width=1001,
                    height=100,
                    orientation=GridOrientation.Y_DOWN,
                    idempotency_key="too-large",
                )

    def test_event_first_commit_failure_is_recovered_before_next_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = EmbeddedProjectService(Path(temporary))
            account = service.accounts.create_account("operator", "Operator", "correct horse battery staple")
            session = service.login(account.username, "correct horse battery staple")
            assert session is not None

            class FailingAcl:
                def roles_for(self, project_id, principal_id):
                    return service.identities.roles_for(project_id, principal_id)

                def assign(self, assignment):
                    raise OSError("simulated ACL storage failure")

                def revoke(self, project_id, principal_id, role):
                    service.identities.revoke(project_id, principal_id, role)

            command = CreateProjectCommand(
                context=CommandContext(actor=session.principal, idempotency_key="recover-create"),
                name="Recoverable",
                width=3,
                height=3,
                orientation=GridOrientation.Y_DOWN,
                storage_profile_id=service.profile.id,
            )
            factory = LocalProjectUnitOfWorkFactory(
                service.catalog_root,
                str(command.project_id),
                identities=service.identities,
                acl=FailingAcl(),
            )
            with self.assertRaises(OSError):
                CreateProjectHandler(factory, service.profiles, service.clock)(command)
            recovered = service.list_projects()
            self.assertEqual((command.project_id,), tuple(item.id for item in recovered))
            self.assertEqual(
                frozenset({ProjectRole.OWNER}),
                service.identities.roles_for(command.project_id, session.principal.id),
            )


if __name__ == "__main__":
    unittest.main()
