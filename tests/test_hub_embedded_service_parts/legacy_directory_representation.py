"""test_hub_embedded_service: test legacy directory representation."""
from __future__ import annotations
import tempfile
from pathlib import Path
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind


class LegacyDirectoryRepresentationMixin:
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
