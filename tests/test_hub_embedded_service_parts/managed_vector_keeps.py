"""test_hub_embedded_service: test managed vector keeps."""
from __future__ import annotations
import tempfile
from pathlib import Path
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind


class ManagedVectorKeepsMixin:
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
