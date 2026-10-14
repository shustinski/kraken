"""test_hub_embedded_service: test backup restore requires."""
from __future__ import annotations
import tempfile
from pathlib import Path
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.domain.identity import ProjectRole
from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind
from kraken_manager.infrastructure.filesystem import SQLiteProjectionStore


class BackupRestoreRequiresMixin:
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
            with self.assertRaisesRegex(ValueError, "принятие проекта"):
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
                frozenset({ProjectRole.MAINTAINER}),
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
