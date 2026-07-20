from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.application.dto import CommandContext, CreateProjectCommand
from kraken_manager.application.use_cases import CreateProjectHandler
from kraken_manager.domain.identity import ProjectRole
from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind
from kraken_manager.domain.artifacts import deterministic_frame_series_id
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

    def test_project_and_layer_lifecycle_is_audited_temporal_and_rebuildable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = EmbeddedProjectService(Path(temporary))
            session = service.create_initial_account("owner", "Owner", "")
            project = service.create_project(
                principal=session.principal,
                name="Original",
                width=4,
                height=3,
                orientation=GridOrientation.Y_DOWN,
                idempotency_key="project",
            )
            layer = service.create_layer(
                principal=session.principal,
                project=project,
                name="Metal",
                layer_type=LayerType.METAL,
                order=1,
                idempotency_key="layer",
            )
            project = service.get_project(project.id)
            assert project is not None
            project = service.rename_project(
                principal=session.principal,
                project=project,
                name="Renamed",
                idempotency_key="rename-project",
            )
            renamed_at = service.history(project.id)[-1].recorded_at
            layer = service.rename_layer(
                principal=session.principal,
                project=project,
                layer=layer,
                name="M1",
                idempotency_key="rename-layer",
            )
            layer = service.reorder_layer(
                principal=session.principal,
                project=project,
                layer=layer,
                order=7,
                idempotency_key="reorder-layer",
            )
            layer = service.archive_layer(
                principal=session.principal,
                project=project,
                layer=layer,
                idempotency_key="archive-layer",
            )
            project = service.archive_project(
                principal=session.principal,
                project=project,
                idempotency_key="archive-project",
            )

            self.assertEqual((), service.list_projects())
            self.assertEqual("Renamed", service.get_project(project.id, as_of=renamed_at).name)
            self.assertEqual("archived", project.state.value)
            self.assertEqual("archived", layer.state.value)

            # The same command key is a no-op even with the stale revision supplied by the caller.
            self.assertEqual(
                project,
                service.archive_project(
                    principal=session.principal,
                    project=project,
                    idempotency_key="archive-project",
                ),
            )
            project = service.restore_project(
                principal=session.principal,
                project=project,
                idempotency_key="restore-project",
            )
            SQLiteProjectionStore(service.catalog_root, str(project.id)).destroy_cache()
            rebuilt = service.get_project(project.id)
            rebuilt_layer = service.list_layers(project.id, include_archived=True)[0]
            self.assertEqual(("Renamed", "active"), (rebuilt.name, rebuilt.state.value))
            self.assertEqual(("M1", 7, "archived"), (rebuilt_layer.name, rebuilt_layer.order, rebuilt_layer.state.value))
            self.assertIn("ProjectArchived", {item.event_type for item in service.activity_records()})

    def test_directory_import_is_preflighted_managed_immutable_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "1_1.png").write_bytes(b"first")
            (source / "2_1.png").write_bytes(b"second")
            service = EmbeddedProjectService(root / "data")
            session = service.create_initial_account("owner", "Owner", "")
            project = service.create_project(
                principal=session.principal,
                name="Import",
                width=2,
                height=2,
                orientation=GridOrientation.Y_DOWN,
                idempotency_key="project",
            )
            layer = service.create_layer(
                principal=session.principal,
                project=project,
                name="Metal",
                layer_type=LayerType.METAL,
                order=1,
                idempotency_key="layer",
            )
            project = service.get_project(project.id)
            assert project is not None
            representation = service.create_representation(
                principal=session.principal,
                project=project,
                layer=layer,
                name="Raw",
                kind=RepresentationKind.IMAGE,
                idempotency_key="representation",
                active=True,
            )
            plan = service.plan_import_directory(project=project, directory=source)
            self.assertTrue(plan.ready)
            self.assertEqual((2, 2, 11), (len(plan.items), plan.missing_coordinates, plan.total_bytes))
            imported = service.commit_managed_import(
                principal=session.principal,
                project=project,
                layer=layer,
                representation=representation,
                plan=plan,
                idempotency_key="import",
            )
            repeated = service.commit_managed_import(
                principal=session.principal,
                project=project,
                layer=layer,
                representation=representation,
                plan=plan,
                idempotency_key="import",
            )
            self.assertEqual(imported.versions, repeated.versions)
            cells = service.frame_cells(project.id, layer.id, representation.id)
            self.assertEqual(
                ((1, 1, "image_ready"), (2, 1, "image_ready")),
                tuple((cell.x, cell.y, cell.status) for cell in cells),
            )
            projection = service._projection(project.id)
            for item, version in zip(plan.items, imported.versions, strict=True):
                series_id = deterministic_frame_series_id(
                    representation.id, project.frame_id_at(item.x, item.y)
                )
                self.assertEqual(version, projection.get_active_artifact_version(series_id))
                self.assertTrue(
                    (
                        service.catalog_root
                        / "projects"
                        / str(project.id)
                        / "objects"
                        / "sha256"
                        / version.sha256[:2]
                        / version.sha256[2:4]
                        / version.sha256
                    ).is_file()
                )
            scan = service.scan_integrity()
            self.assertTrue(scan.valid)
            self.assertEqual(2, scan.blobs)
            backup = root / "backup"
            manifest = service.export_backup(project.id, backup)
            self.assertGreater(manifest.event_count, 0)
            restored_service = EmbeddedProjectService(root / "restored")
            restored = restored_service.import_backup(backup)
            self.assertEqual((project.id, project.name), (restored.id, restored.name))
            self.assertTrue(restored_service.scan_integrity().valid)

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
                role=ProjectRole.CONTRIBUTOR,
                expected_revision=0,
                idempotency_key="assign",
            )
            self.assertEqual(frozenset({ProjectRole.CONTRIBUTOR}), roles)
            self.assertEqual(
                roles,
                service.assign_project_role(
                    principal=owner.principal,
                    project=project,
                    target_principal_id=worker.principal.id,
                    role=ProjectRole.CONTRIBUTOR,
                    expected_revision=0,
                    idempotency_key="assign",
                ),
            )
            roles = service.revoke_project_role(
                principal=owner.principal,
                project=project,
                target_principal_id=worker.principal.id,
                role=ProjectRole.CONTRIBUTOR,
                expected_revision=1,
                idempotency_key="revoke",
            )
            self.assertEqual(frozenset(), roles)
            event_types = {event.event_type for event in service.history(project.id)}
            self.assertTrue({"ProjectRoleAssigned", "ProjectRoleRevoked"}.issubset(event_types))

    def test_representation_lifecycle_switches_active_variant_and_rebuilds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = EmbeddedProjectService(Path(temporary))
            session = service.create_initial_account("owner", "Owner", "")
            project = service.create_project(
                principal=session.principal, name="R", width=1, height=1,
                orientation=GridOrientation.Y_DOWN, idempotency_key="p",
            )
            layer = service.create_layer(
                principal=session.principal, project=project, name="M", layer_type=LayerType.METAL,
                order=1, idempotency_key="l",
            )
            project = service.get_project(project.id)
            assert project is not None
            first = service.create_representation(
                principal=session.principal, project=project, layer=layer, name="A",
                kind=RepresentationKind.IMAGE, idempotency_key="a", active=True,
            )
            layer = service.list_layers(project.id)[0]
            second = service.create_representation(
                principal=session.principal, project=project, layer=layer, name="B",
                kind=RepresentationKind.IMAGE, idempotency_key="b", active=False,
            )
            layer = service.list_layers(project.id)[0]
            first = service.rename_representation(
                principal=session.principal, project=project, layer=layer,
                representation=first, name="Original", idempotency_key="rename",
            )
            layer = service.list_layers(project.id)[0]
            first = service.update_representation_note(
                principal=session.principal, project=project, layer=layer,
                representation=first, note="source scan", idempotency_key="note",
            )
            layer = service.list_layers(project.id)[0]
            second = service.activate_representation(
                principal=session.principal, project=project, layer=layer,
                representation=second, idempotency_key="activate",
            )
            values = service.list_representations(project.id, layer.id)
            first = next(item for item in values if item.id == first.id)
            self.assertFalse(first.active)
            self.assertTrue(second.active)
            layer = service.list_layers(project.id)[0]
            service.archive_representation(
                principal=session.principal, project=project, layer=layer,
                representation=first, idempotency_key="archive",
            )
            SQLiteProjectionStore(service.catalog_root, str(project.id)).destroy_cache()
            rebuilt = service.list_representations(project.id, layer.id)
            self.assertEqual(("B",), tuple(item.name for item in rebuilt))


if __name__ == "__main__":
    unittest.main()
