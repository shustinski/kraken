from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from kraken_core.frame_matrix import StoreNamespace
from kraken_core.plugin_protocol import PluginAsset, PluginResultPublicationV2
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.application.dto import CommandContext, CreateProjectCommand
from kraken_manager.application.use_cases import CreateProjectHandler
from kraken_manager.domain.artifacts import ArtifactScope, deterministic_frame_series_id
from kraken_manager.domain.identity import ProjectRole
from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind
from kraken_manager.infrastructure.filesystem import LocalProjectUnitOfWorkFactory, SQLiteProjectionStore


class EmbeddedProjectServiceTests(unittest.TestCase):
    @staticmethod
    def _memory_secret_store():
        class MemorySecretStore:
            def __init__(self) -> None:
                self.values: dict[str, bytes] = {}

            def get(self, key: str) -> bytes | None:
                return self.values.get(key)

            def set(self, key: str, value: bytes) -> None:
                self.values[key] = value

        return MemorySecretStore()

    def test_export_artifact_version_supports_managed_and_verified_external_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = EmbeddedProjectService(root / "data")
            session = service.create_initial_account("owner", "Owner", "")
            project = service.create_project(
                principal=session.principal,
                name="Files",
                width=1,
                height=1,
                orientation=GridOrientation.Y_DOWN,
                idempotency_key="project",
            )
            managed_source = root / "managed.bin"
            external_source = root / "external.bin"
            managed_source.write_bytes(b"managed")
            external_source.write_bytes(b"external")

            managed_series = service.create_artifact_series(
                principal=session.principal,
                project_id=project.id,
                scope=ArtifactScope.PROJECT_ATTACHMENT,
                name="Managed",
                idempotency_key="managed-series",
            )
            managed = service.add_managed_artifact_version(
                principal=session.principal,
                project_id=project.id,
                series_id=managed_series.id,
                source=managed_source,
                idempotency_key="managed-version",
            )
            external_series = service.create_artifact_series(
                principal=session.principal,
                project_id=project.id,
                scope=ArtifactScope.PROJECT_EXTERNAL_LINK,
                name="External",
                idempotency_key="external-series",
            )
            external = service.add_external_artifact_version(
                principal=session.principal,
                project_id=project.id,
                series_id=external_series.id,
                source=external_source,
                idempotency_key="external-version",
            )

            managed_target = root / "managed-export.bin"
            external_target = root / "external-export.bin"
            service.export_artifact_version(project.id, managed, managed_target)
            service.export_artifact_version(project.id, external, external_target)
            self.assertEqual(b"managed", managed_target.read_bytes())
            self.assertEqual(b"external", external_target.read_bytes())

            external_source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "изменён"):
                service.export_artifact_version(
                    project.id,
                    external,
                    root / "blocked.bin",
                )

    def test_notes_and_artifact_lifecycle_are_rebuildable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = EmbeddedProjectService(root / "data")
            session = service.create_initial_account("owner", "Owner", "")
            project = service.create_project(
                principal=session.principal,
                name="Lifecycle",
                width=1,
                height=1,
                orientation=GridOrientation.Y_DOWN,
                idempotency_key="project",
            )
            source = root / "attachment.txt"
            source.write_text("first", encoding="utf-8")
            series = service.create_artifact_series(
                principal=session.principal,
                project_id=project.id,
                scope=ArtifactScope.PROJECT_ATTACHMENT,
                name="Документ",
                idempotency_key="series",
            )
            first = service.add_managed_artifact_version(
                principal=session.principal,
                project_id=project.id,
                series_id=series.id,
                source=source,
                idempotency_key="version-1",
            )
            source.write_text("second", encoding="utf-8")
            second = service.add_managed_artifact_version(
                principal=session.principal,
                project_id=project.id,
                series_id=series.id,
                source=source,
                parent_version_id=first.id,
                idempotency_key="version-2",
            )
            service.activate_artifact_version(
                principal=session.principal,
                project_id=project.id,
                series_id=series.id,
                version_id=first.id,
                idempotency_key="activate-first",
            )
            series = service.rename_artifact_series(
                principal=session.principal,
                series=service.list_artifact_series(project.id)[0],
                name="Переименованный документ",
                idempotency_key="rename-series",
            )
            service.archive_artifact_series(
                principal=session.principal,
                series=series,
                idempotency_key="archive-series",
            )
            note = service.create_note(
                principal=session.principal,
                project_id=project.id,
                body="Первая редакция",
                idempotency_key="note",
            )
            revised = service.revise_note(
                principal=session.principal,
                note=note,
                body="Вторая редакция",
                idempotency_key="revise-note",
            )

            projection = SQLiteProjectionStore(service.catalog_root, str(project.id))
            projection.destroy_cache()

            rebuilt_series = service.list_artifact_series(
                project.id,
                include_archived=True,
            )
            self.assertEqual(1, len(rebuilt_series))
            self.assertEqual("Переименованный документ", rebuilt_series[0].name)
            self.assertTrue(rebuilt_series[0].archived)
            self.assertEqual(
                {first.id, second.id},
                {
                    version.id
                    for version in service.artifact_versions(project.id, series.id)
                },
            )
            self.assertEqual(
                first.id,
                service.active_artifact_version(project.id, series.id).id,
            )
            rebuilt_notes = service.list_notes(project.id)
            self.assertEqual((revised.note_id,), tuple(item.note_id for item in rebuilt_notes))
            self.assertEqual((2, "Вторая редакция"), (rebuilt_notes[0].revision, rebuilt_notes[0].body))

    def test_delete_project_removes_kraken_cache_but_preserves_workspace_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = EmbeddedProjectService(root / "data")
            session = service.create_initial_account("operator", "Operator", "")
            (root / "source").mkdir()
            (root / "derived").mkdir()
            project = service.create_project(
                principal=session.principal,
                name="Preserved project",
                width=2,
                height=2,
                orientation=GridOrientation.Y_DOWN,
                idempotency_key="create-project",
                source_root=root / "source",
                derived_root=root / "derived",
            )
            binding = service.project_workspace(project.id)
            self.assertIsNotNone(binding)
            assert binding is not None
            source_file = Path(binding.source_project_dir) / "img" / "0.jpg"
            result_file = Path(binding.derived_project_dir) / "result" / "model.bin"
            vector_file = Path(binding.derived_project_dir) / "vector" / "0.cif"
            source_file.write_bytes(b"image")
            result_file.write_bytes(b"model")
            vector_file.write_bytes(b"vector")

            project_cache = service.catalog_root / "projects" / str(project.id)
            (project_cache / "staging" / "temporary.bin").write_bytes(b"cache")
            thumbnail_root = service.data_dir / "cache" / "frame-thumbnails"
            thumbnail_namespace = thumbnail_root / StoreNamespace(
                plugin="matrix",
                project=str(project.id),
                generation="v1",
            ).digest()
            thumbnail_namespace.mkdir(parents=True)
            (thumbnail_namespace / "thumb.sqlite3").write_bytes(b"cache")
            project_staging = service.data_dir / "agent-staging" / f"{project.id}-job"
            project_staging.mkdir(parents=True)
            (project_staging / "job.json").write_text(
                f'{{"project_id":"{project.id}"}}',
                encoding="utf-8",
            )
            other_staging = service.data_dir / "agent-staging" / "other-job"
            other_staging.mkdir()
            (other_staging / "job.json").write_text(
                '{"project_id":"other-project"}',
                encoding="utf-8",
            )

            result = service.delete_project(
                principal=session.principal,
                project=project,
                confirmation_name=project.name,
            )

            self.assertTrue(result.catalog_cache_removed)
            self.assertTrue(result.thumbnail_cache_removed)
            self.assertEqual(1, result.staging_directories_removed)
            self.assertFalse(project_cache.exists())
            self.assertFalse(thumbnail_namespace.exists())
            self.assertFalse(project_staging.exists())
            self.assertTrue(other_staging.exists())
            self.assertTrue(source_file.is_file())
            self.assertTrue(result_file.is_file())
            self.assertTrue(vector_file.is_file())
            self.assertEqual(
                (),
                tuple(item for item in service.list_projects(include_archived=True) if item.id == project.id),
            )
            self.assertEqual(
                frozenset(),
                service.project_roles(project.id, session.principal.id),
            )

    def test_legacy_directory_representation_provides_viewport_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "images"
            source.mkdir()
            for index in range(4):
                (source / f"{index:04d}.jpg").write_bytes(f"image-{index}".encode())

            service = EmbeddedProjectService(root / "data")
            session = service.create_initial_account("operator", "Operator", "")
            project = service.create_project(
                principal=session.principal,
                name="External images",
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
            representation = service.create_representation(
                principal=session.principal,
                project=project,
                layer=layer,
                name="Legacy",
                kind=RepresentationKind.IMAGE,
                idempotency_key="representation",
                source=str(source),
                active=True,
            )

            viewport = service.matrix_viewport(
                project.id,
                layer_id=layer.id,
                representation_ids=(representation.id,),
                x1=1,
                y1=1,
                x2=2,
                y2=2,
            )

            self.assertEqual(4, len(viewport["cells"]))
            self.assertEqual(
                tuple(str(source / f"{index:04d}.jpg") for index in range(4)),
                tuple(cell["asset_path"] for cell in viewport["cells"]),
            )
            self.assertTrue(all(cell["asset_source_key"] for cell in viewport["cells"]))

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
            self.assertEqual(image.id, vector.source_image_representation_id)

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

    def test_manual_performer_can_be_updated_and_archived(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = EmbeddedProjectService(Path(directory))
            created = service.create_manual_performer(name="Reviewer", color="#123ABC")

            updated = service.update_performer(
                performer_id=created.id,
                name="Senior reviewer",
                color="#60A5FA",
            )

            self.assertEqual("Senior reviewer", updated.name)
            self.assertEqual("#60A5FA", updated.color)
            archived = service.archive_performer(created.id)
            self.assertFalse(archived.active)
            self.assertEqual((), service.list_performers())
            self.assertEqual((archived,), service.list_performers(include_archived=True))

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

    def test_local_profile_accepts_more_than_one_million_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = EmbeddedProjectService(Path(temporary))
            account = service.accounts.create_account("operator", "Operator", "correct horse battery staple")
            session = service.login(account.username, "correct horse battery staple")
            assert session is not None
            project = service.create_project(
                principal=session.principal,
                name="Large sparse project",
                width=10_001,
                height=100,
                orientation=GridOrientation.Y_DOWN,
                idempotency_key="large-project",
            )
            self.assertEqual(1_000_100, project.frame_count)

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
            detailed_viewport = service.matrix_viewport(
                project.id,
                layer_id=layer.id,
                representation_ids=(representation.id,),
                x1=1,
                y1=1,
                x2=2,
                y2=2,
                lod=0,
            )
            aggregate_viewport = service.matrix_viewport(
                project.id,
                layer_id=layer.id,
                representation_ids=(representation.id,),
                x1=1,
                y1=1,
                x2=2,
                y2=2,
                lod=1,
            )
            self.assertEqual(2, len(detailed_viewport["cells"]))
            self.assertEqual("image_ready", detailed_viewport["cells"][0]["status"])
            self.assertEqual(
                imported.versions[0].sha256,
                detailed_viewport["cells"][0]["asset_sha256"],
            )
            self.assertEqual(
                {"image_ready": 2},
                aggregate_viewport["aggregates"][0]["status_counts"],
            )
            self.assertTrue(detailed_viewport["revision"])
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

    def test_sparse_managed_vector_keeps_raster_asset_in_viewport(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = EmbeddedProjectService(root / "data")
            session = service.create_initial_account("owner", "Owner", "")
            project = service.create_project(
                principal=session.principal,
                name="Sparse vector",
                width=2,
                height=1,
                orientation=GridOrientation.Y_DOWN,
                idempotency_key="project",
            )
            layer = service.create_layer(
                principal=session.principal,
                project=project,
                name="Metal",
                layer_type=LayerType.METAL,
                order=0,
                idempotency_key="layer",
            )
            images = service.create_representation(
                principal=session.principal,
                project=project,
                layer=layer,
                name="Images",
                kind=RepresentationKind.IMAGE,
                source="managed-import",
                active=True,
                idempotency_key="images",
            )
            vectors = service.create_representation(
                principal=session.principal,
                project=project,
                layer=layer,
                name="CIF",
                kind=RepresentationKind.VECTOR,
                source="managed-import",
                source_image_representation_id=images.id,
                active=True,
                idempotency_key="vectors",
            )
            source = root / "source"
            source.mkdir()
            (source / "1_1.png").write_bytes(b"image")
            image_plan = service.plan_import_directory(project=project, directory=source)
            image_result = service.commit_managed_import(
                principal=session.principal,
                project=project,
                layer=layer,
                representation=images,
                plan=image_plan,
                idempotency_key="image-import",
            )

            viewport = service.matrix_viewport(
                project.id,
                layer_id=layer.id,
                representation_ids=(images.id, vectors.id),
                x1=1,
                y1=1,
                x2=1,
                y2=1,
                lod=0,
            )

            cell = viewport["cells"][0]
            self.assertTrue(cell["missing"])
            self.assertEqual("error", cell["status"])
            self.assertEqual(image_result.versions[0].sha256, cell["asset_sha256"])
            self.assertEqual(
                (str(vectors.id),),
                cell["missing_representation_ids"],
            )
            backup = root / "backup"
            manifest = service.export_backup(project.id, backup)
            self.assertGreater(manifest.event_count, 0)
            restored_service = EmbeddedProjectService(root / "restored")
            restored = restored_service.import_backup(backup)
            self.assertEqual((project.id, project.name), (restored.id, restored.name))
            self.assertTrue(restored_service.scan_integrity().valid)

    def test_managed_vector_keeps_external_image_thumbnail_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = EmbeddedProjectService(root / "data")
            session = service.create_initial_account("owner", "Owner", "")
            project = service.create_project(
                principal=session.principal,
                name="External image and CIF",
                width=1,
                height=1,
                orientation=GridOrientation.Y_DOWN,
                idempotency_key="project",
            )
            layer = service.create_layer(
                principal=session.principal,
                project=project,
                name="Metal",
                layer_type=LayerType.METAL,
                order=0,
                idempotency_key="layer",
            )
            image_directory = root / "images"
            image_directory.mkdir()
            image_path = image_directory / "0001.png"
            image_path.write_bytes(b"image")
            images = service.create_representation(
                principal=session.principal,
                project=project,
                layer=layer,
                name="Images",
                kind=RepresentationKind.IMAGE,
                source=str(image_directory),
                active=True,
                idempotency_key="images",
            )
            vectors = service.create_representation(
                principal=session.principal,
                project=project,
                layer=layer,
                name="CIF",
                kind=RepresentationKind.VECTOR,
                source="managed-import",
                source_image_representation_id=images.id,
                active=True,
                idempotency_key="vectors",
            )
            vector_directory = root / "vectors"
            vector_directory.mkdir()
            (vector_directory / "1_1.cif").write_text("CIF", encoding="utf-8")
            vector_plan = service.plan_import_directory(
                project=project,
                directory=vector_directory,
            )
            vector_result = service.commit_managed_import(
                principal=session.principal,
                project=project,
                layer=layer,
                representation=vectors,
                plan=vector_plan,
                idempotency_key="vector-import",
            )

            viewport = service.matrix_viewport(
                project.id,
                layer_id=layer.id,
                representation_ids=(images.id, vectors.id),
                x1=1,
                y1=1,
                x2=1,
                y2=1,
                lod=0,
            )

            cell = viewport["cells"][0]
            self.assertEqual("vectorized", cell["status"])
            self.assertEqual(vector_result.versions[0].sha256, cell["sha256"])
            self.assertEqual(str(image_path), cell["asset_path"])
            self.assertTrue(cell["asset_source_key"])
            self.assertEqual("image/png", cell["asset_media_type"])

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

            with self.assertRaisesRegex(ValueError, "последнего владельца"):
                service.revoke_project_role(
                    principal=owner.principal,
                    project=project,
                    target_principal_id=owner.principal.id,
                    role=ProjectRole.OWNER,
                    expected_revision=service.project_role_revision(
                        project.id,
                        owner.principal.id,
                    ),
                    idempotency_key="revoke-last-owner",
                )

            self.assertEqual(
                frozenset({ProjectRole.OWNER}),
                service.project_roles(project.id, owner.principal.id),
            )

    def test_desktop_review_package_round_trip_uses_manifest_routing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = EmbeddedProjectService(
                root / "data",
                secret_store=self._memory_secret_store(),
            )
            owner = service.create_initial_account("owner", "Owner", "")
            performer = service.create_manual_performer(
                name="Reviewer",
                color="#336699",
            )
            project = service.create_project(
                principal=owner.principal,
                name="Review",
                width=1,
                height=1,
                orientation=GridOrientation.Y_DOWN,
                idempotency_key="project",
            )
            layer = service.create_layer(
                principal=owner.principal,
                project=project,
                name="Metal",
                layer_type=LayerType.METAL,
                order=0,
                idempotency_key="layer",
            )
            images = service.create_representation(
                principal=owner.principal,
                project=service.get_project(project.id),
                layer=layer,
                name="Images",
                kind=RepresentationKind.IMAGE,
                active=True,
                idempotency_key="images",
            )
            layer = service.list_layers(project.id)[0]
            vectors = service.create_representation(
                principal=owner.principal,
                project=service.get_project(project.id),
                layer=layer,
                name="CIF",
                kind=RepresentationKind.VECTOR,
                source_image_representation_id=images.id,
                active=True,
                idempotency_key="vectors",
            )
            source_images = root / "images"
            source_vectors = root / "vectors"
            source_images.mkdir()
            source_vectors.mkdir()
            (source_images / "1_1.png").write_bytes(b"image")
            (source_vectors / "1_1.cif").write_text("DS 1 1 1;\nDF;\nE\n", encoding="utf-8")
            service.commit_managed_import(
                principal=owner.principal,
                project=service.get_project(project.id),
                layer=layer,
                representation=images,
                plan=service.plan_import_directory(
                    project=project,
                    directory=source_images,
                ),
                idempotency_key="import-images",
            )
            service.commit_managed_import(
                principal=owner.principal,
                project=service.get_project(project.id),
                layer=layer,
                representation=vectors,
                plan=service.plan_import_directory(
                    project=project,
                    directory=source_vectors,
                ),
                idempotency_key="import-vectors",
            )
            batch = service.create_review_batch(
                principal=owner.principal,
                project_id=project.id,
                layer_id=layer.id,
                image_representation_id=images.id,
                vector_representation_id=vectors.id,
                coordinates=((1, 1),),
                assignee_id=performer.id,
                instructions="Check",
                due_at=None,
                idempotency_key="review",
            )
            package = root / "review-package"
            issued = service.export_review_batch(
                principal=owner.principal,
                batch=batch,
                destination=package,
                idempotency_key="export",
            )
            returned_cif = next(package.rglob("*.cif"))
            returned_cif.write_text("DS 1 1 1;\nP 0 0 1 0 1 1;\nDF;\nE\n", encoding="utf-8")

            matched_batch, preflight = service.review_return_preflight(
                principal=owner.principal,
                source=package,
                idempotency_key="preflight",
            )
            self.assertEqual(issued.id, matched_batch.id)
            self.assertTrue(preflight.report.can_commit)
            committed = service.commit_review_return(
                principal=owner.principal,
                batch=matched_batch,
                source=package,
                idempotency_key="commit",
            )
            self.assertEqual(1, len(committed.candidate_versions))
            accepted = service.accept_review(
                principal=owner.principal,
                batch=committed.batch,
                candidate_version_ids=tuple(
                    version.id for version in committed.candidate_versions
                ),
                idempotency_key="accept",
            )
            self.assertEqual("completed", accepted.state.value)

    def test_backup_restore_requires_explicit_ownership_on_new_workstation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_service = EmbeddedProjectService(root / "source-data")
            source_owner = source_service.create_initial_account(
                "source-owner",
                "Source Owner",
                "",
            )
            project = source_service.create_project(
                principal=source_owner.principal,
                name="Portable",
                width=1,
                height=1,
                orientation=GridOrientation.Y_DOWN,
                idempotency_key="project",
            )
            bundle = root / "bundle"
            source_service.export_backup(
                project.id,
                bundle,
                principal=source_owner.principal,
            )

            restored_service = EmbeddedProjectService(root / "restored-data")
            new_owner = restored_service.create_initial_account(
                "new-owner",
                "New Owner",
                "",
            )
            with self.assertRaisesRegex(ValueError, "принятие владения"):
                restored_service.import_backup(
                    bundle,
                    principal=new_owner.principal,
                    take_ownership=False,
                )

            restored = restored_service.import_backup(
                bundle,
                principal=new_owner.principal,
                take_ownership=True,
            )
            self.assertEqual(project.id, restored.id)
            self.assertEqual(
                frozenset({ProjectRole.OWNER}),
                restored_service.project_roles(
                    restored.id,
                    new_owner.principal.id,
                ),
            )

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

    def test_v2_publications_are_merged_before_domain_import(self) -> None:
        service = object.__new__(EmbeddedProjectService)
        captured = {}

        def import_result(**kwargs):
            captured.update(kwargs)
            return "imported"

        service.import_agent_result = import_result
        first = PluginResultPublicationV2(
            job_id="job-1",
            publication_id="publication-1",
            sequence=1,
            plugin_id="plugin",
            plugin_version="2",
            outputs=(
                PluginAsset(
                    asset_id="dataset-1",
                    role="dataset",
                    scope="layer",
                    relative_path="outputs/dataset.zip",
                    sha256=hashlib.sha256(b"dataset").hexdigest(),
                    media_type="application/zip",
                ),
            ),
            applied_parameters={"threshold": 0.5},
            frame_values={"frame-1": {"confidence": 0.7}},
        )
        final = PluginResultPublicationV2(
            job_id="job-1",
            publication_id="publication-2",
            sequence=2,
            plugin_id="plugin",
            plugin_version="2",
            outputs=(
                PluginAsset(
                    asset_id="model-1",
                    role="model",
                    scope="layer",
                    relative_path="outputs/model.bin",
                    sha256=hashlib.sha256(b"model").hexdigest(),
                    media_type="application/octet-stream",
                ),
            ),
            frame_values={"frame-1": {"quality": 0.9}},
            final=True,
        )

        result = service.import_agent_publications(
            principal=object(),
            publications=(final.to_dict(), first.to_dict()),
            staging_root=Path("."),
        )

        self.assertEqual("imported", result)
        merged = captured["result_payload"]
        self.assertEqual(
            ["dataset-1", "model-1"],
            [item["asset_id"] for item in merged["outputs"]],
        )
        self.assertEqual(
            {"confidence": 0.7, "quality": 0.9},
            merged["frame_values"]["frame-1"],
        )
        self.assertEqual(
            ["publication-1", "publication-2"],
            merged["applied_parameters"]["v2_publication_ids"],
        )


if __name__ == "__main__":
    unittest.main()
