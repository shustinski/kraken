"""test_hub_embedded_service: test notes artifact lifecycle."""
from __future__ import annotations
import tempfile
from pathlib import Path
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.domain.artifacts import ArtifactScope
from kraken_manager.domain.project import GridOrientation
from kraken_manager.infrastructure.filesystem import SQLiteProjectionStore


class NotesArtifactLifecycleMixin:
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
