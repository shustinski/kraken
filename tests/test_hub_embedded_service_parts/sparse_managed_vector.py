"""test_hub_embedded_service: test sparse managed vector."""
from __future__ import annotations
import tempfile
from pathlib import Path
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind


class SparseManagedVectorMixin:
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
