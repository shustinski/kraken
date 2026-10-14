"""test_hub_embedded_service: test directory import preflighted."""
from __future__ import annotations
import tempfile
from pathlib import Path
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.domain.artifacts import deterministic_frame_series_id
from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind


class DirectoryImportPreflightedMixin:
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
