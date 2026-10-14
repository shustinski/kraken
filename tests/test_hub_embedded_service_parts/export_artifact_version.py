"""test_hub_embedded_service: test export artifact version."""
from __future__ import annotations
import tempfile
from pathlib import Path
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.domain.artifacts import ArtifactScope
from kraken_manager.domain.project import GridOrientation


class ExportArtifactVersionMixin:
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
