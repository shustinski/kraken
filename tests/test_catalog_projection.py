"""Catalog tables expose project names, grids and files without reading JSON."""

from __future__ import annotations

import hashlib
import unittest
from datetime import UTC, datetime

from sqlalchemy import create_engine, select

from kraken_manager.domain.artifacts import ArtifactSeries, ArtifactVersion, ExternalReference
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.project import (
    GridOrientation,
    Layer,
    LayerType,
    Project,
    Representation,
    RepresentationKind,
)
from kraken_manager.infrastructure.postgres.catalog import CatalogProjection, catalog_tables


NOW = datetime(2026, 9, 24, tzinfo=UTC)


class CatalogProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        self.tables = catalog_tables()
        for table in self.tables.values():
            table.create(self.engine)
        self.catalog = CatalogProjection(self.tables)
        self.project = Project.create(
            name="kraken_test_project",
            width=10,
            height=10,
            orientation=GridOrientation.Y_DOWN,
            storage_profile="server-postgres",
            created_at=NOW,
        )
        self.layer = Layer.create(
            project_id=self.project.id,
            name="l1",
            type=LayerType.METAL,
            order=0,
            created_at=NOW,
        )
        self.representation = Representation.create(
            project_id=self.project.id,
            layer_id=self.layer.id,
            name="Исходные изображения",
            kind=RepresentationKind.IMAGE,
            source=r"D:\storage\kraken_test_project\img\l1",
            active=True,
            created_at=NOW,
        )

    def test_project_layer_and_frame_are_readable_by_name(self) -> None:
        with self.engine.begin() as connection:
            self.catalog.sync(connection, "project", self.project, {"updated_at": NOW})
            self.catalog.sync(connection, "layer", self.layer, {"updated_at": NOW})
            self.catalog.sync(
                connection, "representation", self.representation, {"active": True, "updated_at": NOW}
            )
            series = ArtifactSeries.for_frame(
                project_id=self.project.id,
                layer_id=self.layer.id,
                representation_id=self.representation.id,
                frame_id=self.project.frame_id_at(9, 1),
                name="0008.jpg",
            )
            self.catalog.sync(connection, "artifact_series", series, {"updated_at": NOW})
            digest = hashlib.sha256(b"frame").hexdigest()
            version = ArtifactVersion.external_link(
                series_id=series.id,
                reference=ExternalReference(
                    uri="file:///D:/storage/kraken_test_project/img/l1/0008.jpg",
                    fingerprint_sha256=digest,
                    observed_size_bytes=5,
                ),
                media_type="image/jpeg",
                filename="0008.jpg",
                author_principal_id=Principal.local(subject="reader", display_name="Reader").id,
                created_at=NOW,
            )
            self.catalog.sync(
                connection,
                "artifact_version",
                version,
                {"active": True, "updated_at": NOW, "project_id": str(self.project.id)},
            )
            project = connection.execute(
                select(
                    self.tables["projects"].c.name,
                    self.tables["projects"].c.width,
                    self.tables["projects"].c.height,
                    self.tables["projects"].c.source_directory,
                )
            ).one()
            frame = connection.execute(
                select(
                    self.tables["images"].c.x,
                    self.tables["images"].c.y,
                    self.tables["images"].c.filename,
                    self.tables["images"].c.media_type,
                    self.tables["images"].c.size_bytes,
                )
            ).one()
            vector_count = connection.execute(
                select(self.tables["vectors"].c.filename)
            ).all()

        self.assertEqual(project.name, "kraken_test_project")
        self.assertEqual((project.width, project.height), (10, 10))
        self.assertEqual(project.source_directory, r"D:\storage\kraken_test_project")
        self.assertEqual((frame.x, frame.y, frame.filename), (9, 1, "0008.jpg"))
        self.assertEqual(frame.media_type, "image/jpeg")
        self.assertEqual(frame.size_bytes, 5)
        self.assertEqual(vector_count, [])

    def test_renaming_a_project_updates_child_rows(self) -> None:
        with self.engine.begin() as connection:
            self.catalog.sync(connection, "project", self.project, {"updated_at": NOW})
            self.catalog.sync(connection, "layer", self.layer, {"updated_at": NOW})
            renamed = self.project.rename("Новое имя")
            self.catalog.sync(connection, "project", renamed, {"updated_at": NOW})
            layer_name = connection.execute(select(self.tables["layers"].c.project_name)).scalar_one()
            directory = connection.execute(
                select(self.tables["projects"].c.source_directory)
            ).scalar_one()
        self.assertEqual(layer_name, "Новое имя")
        self.assertIsNone(directory)

    def test_vector_files_are_stored_apart_from_images(self) -> None:
        vector = Representation.create(
            project_id=self.project.id,
            layer_id=self.layer.id,
            name="Эталон CIF",
            kind=RepresentationKind.VECTOR,
            created_at=NOW,
        )
        with self.engine.begin() as connection:
            self.catalog.sync(connection, "project", self.project, {"updated_at": NOW})
            self.catalog.sync(connection, "layer", self.layer, {"updated_at": NOW})
            self.catalog.sync(connection, "representation", vector, {"active": False, "updated_at": NOW})
            series = ArtifactSeries.for_frame(
                project_id=self.project.id,
                layer_id=self.layer.id,
                representation_id=vector.id,
                frame_id=self.project.frame_id_at(1, 1),
                name="0000.cif",
            )
            self.catalog.sync(connection, "artifact_series", series, {"updated_at": NOW})
            image_count = connection.execute(select(self.tables["images"].c.filename)).all()
            stored = connection.execute(select(self.tables["vectors"].c.filename)).scalar_one()
        self.assertEqual(image_count, [])
        self.assertEqual(stored, "0000.cif")


if __name__ == "__main__":
    unittest.main()
