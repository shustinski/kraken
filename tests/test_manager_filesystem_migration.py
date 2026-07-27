from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import PrincipalProvider
from kraken_manager.infrastructure.blob import FilesystemBlobStore
from kraken_manager.infrastructure.filesystem import FileProjectLayout, FilesystemEventStore
from kraken_manager.infrastructure.migration import (
    BundleVerifier,
    CanonicalBundleExporter,
    CanonicalBundleImporter,
    MigrationBundleError,
    UnsafeBundlePath,
    safe_join,
)


def _event(project_id: str, stream_id: str, revision: int, event_type: str) -> EventEnvelope:
    return EventEnvelope.create(
        project_id=project_id,
        stream_id=stream_id,
        revision=revision,
        event_type=event_type,
        payload={"number": revision},
        actor=ActorSnapshot(
            principal_id=str(uuid4()),
            provider=PrincipalProvider.LOCAL,
            subject="local:migration",
            display_name="Migration User",
        ),
        recorded_at=datetime.now(timezone.utc),
    )


class FilesystemMigrationTests(unittest.TestCase):
    def test_bundle_round_trip_is_idempotent_and_fully_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            target_root = root / "target"
            project_id = str(uuid4())
            source_layout = FileProjectLayout(source_root, project_id)
            source_layout.initialize({"schema_version": 1, "project_id": project_id, "name": "Migrated"})
            source_events = FilesystemEventStore(source_root, project_id)
            project_stream = f"project:{project_id}"
            layer_stream = f"layer:{uuid4()}"
            source_events.append(
                project_stream,
                expected_revision=0,
                events=(_event(project_id, project_stream, 1, "ProjectCreated"),),
            )
            source_events.append(
                layer_stream,
                expected_revision=0,
                events=(_event(project_id, layer_stream, 1, "LayerCreated"),),
            )
            source_blobs = FilesystemBlobStore.for_project(source_root, project_id)
            blob = source_blobs.put((b"managed", b" content")).blob
            (source_layout.snapshots_dir / "state.json").write_text('{"position":2}', encoding="utf-8")

            bundle_root = root / "bundle"
            exporter = CanonicalBundleExporter(source_events, source_blobs)
            plan = exporter.plan(bundle_root, external_references=({"uri": "file:///offline.bdt"},))
            self.assertEqual((plan.event_count, plan.blob_count, plan.snapshot_count), (2, 1, 1))
            self.assertEqual(plan.external_reference_count, 1)
            manifest = exporter.export(
                bundle_root,
                external_references=({"uri": "file:///offline.bdt", "sha256": "unknown"},),
            )
            report = BundleVerifier().verify(bundle_root)
            self.assertTrue(report.valid, report.errors)
            self.assertEqual(manifest.event_count, 2)

            target_events = FilesystemEventStore(target_root, project_id)
            target_blobs = FilesystemBlobStore.for_project(target_root, project_id)
            importer = CanonicalBundleImporter(target_events, target_blobs)
            result = importer.import_bundle(bundle_root)
            self.assertEqual((result.events_imported, result.blobs_imported, result.snapshots_imported), (2, 1, 1))
            self.assertEqual(
                [event.to_dict(include_storage_metadata=True) for event in target_events.iter_project()],
                [event.to_dict(include_storage_metadata=True) for event in source_events.iter_project()],
            )
            self.assertEqual(target_blobs.read(blob), b"managed content")

            repeated = importer.import_bundle(bundle_root)
            self.assertEqual(repeated.events_imported, 0)
            self.assertEqual(repeated.blobs_imported, 0)

            bundled_blob = next(entry for entry in manifest.entries if entry.kind == "blob")
            (bundle_root / Path(bundled_blob.path)).write_bytes(b"corrupt")
            self.assertFalse(BundleVerifier().verify(bundle_root).valid)

    def test_bundle_paths_reject_traversal_and_non_empty_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for unsafe in ("../escape", "/absolute", "C:/drive", "folder\\file", "CON/file"):
                with self.subTest(path=unsafe), self.assertRaises(UnsafeBundlePath):
                    safe_join(root, unsafe)

            project_id = str(uuid4())
            events = FilesystemEventStore(root / "catalog", project_id)
            blobs = FilesystemBlobStore.for_project(root / "catalog", project_id)
            destination = root / "not-empty"
            destination.mkdir()
            (destination / "foreign.txt").write_text("data", encoding="utf-8")
            with self.assertRaises(MigrationBundleError):
                CanonicalBundleExporter(events, blobs).export(destination)


if __name__ == "__main__":
    unittest.main()
