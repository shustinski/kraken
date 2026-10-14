"""test_hub_embedded_service: test project layer lifecycle."""
from __future__ import annotations
import tempfile
from pathlib import Path
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.domain.project import GridOrientation, LayerType
from kraken_manager.infrastructure.filesystem import SQLiteProjectionStore


class ProjectLayerLifecycleMixin:
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
