from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.application.dto import CommandContext, CreateProjectCommand
from kraken_manager.application.use_cases import CreateProjectHandler
from kraken_manager.domain.identity import ProjectRole
from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind
from kraken_manager.infrastructure.filesystem import LocalProjectUnitOfWorkFactory, SQLiteProjectionStore


class EmbeddedProjectServiceTests(unittest.TestCase):
    def test_image_and_vector_representations_can_be_added_to_a_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = EmbeddedProjectService(Path(temporary))
            session = service.create_initial_account("operator", "Operator", "")
            project = service.create_project(
                principal=session.principal,
                name="Representations",
                width=2,
                height=2,
                orientation=GridOrientation.Y_DOWN,
                idempotency_key="create-project",
            )
            layer = service.create_layer(
                principal=session.principal,
                project=project,
                name="Metal",
                layer_type=LayerType.METAL,
                order=1,
                idempotency_key="create-layer",
            )

            image = service.create_representation(
                principal=session.principal,
                project=project,
                layer=layer,
                name="Original",
                kind=RepresentationKind.IMAGE,
                idempotency_key="create-image",
                active=True,
            )
            vector = service.create_representation(
                principal=session.principal,
                project=project,
                layer=layer,
                name="Contours",
                kind=RepresentationKind.VECTOR,
                idempotency_key="create-vector",
                active=True,
            )

            representations = service.list_representations(project.id, layer.id)
            self.assertEqual({image.id, vector.id}, {item.id for item in representations})
            self.assertEqual(
                {RepresentationKind.IMAGE, RepresentationKind.VECTOR},
                {item.kind for item in representations},
            )

    def test_initial_account_is_created_with_identity_and_authenticated_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = EmbeddedProjectService(Path(temporary))

            session = service.create_initial_account("operator", "Оператор", "correct horse battery staple")

            self.assertTrue(service.has_accounts)
            self.assertEqual("operator", session.principal.subject)
            self.assertEqual("Оператор", session.principal.display_name)
            self.assertEqual(session.principal, service.resolve_session(session.token))

    def test_initial_account_cannot_be_created_when_an_account_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = EmbeddedProjectService(Path(temporary))
            service.create_initial_account("operator", "Operator", "correct horse battery staple")

            with self.assertRaisesRegex(ValueError, "already exists"):
                service.create_initial_account("second", "Second", "another correct password")

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
