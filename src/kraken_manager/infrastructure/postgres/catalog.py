"""Readable project tables kept next to the JSON projection."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .event_store import _sqlalchemy


_FRAME_NUMBER = re.compile(r"\d+")


def frame_coordinates(filename: str, width: int) -> tuple[int, int] | None:
    """Map a zero-based frame number in a file name onto the project grid."""

    if width < 1:
        return None
    numbers = _FRAME_NUMBER.findall(Path(filename).stem)
    if not numbers:
        return None
    index = int(numbers[-1])
    return (index % width) + 1, (index // width) + 1


def project_directory_from_images(image_directory: str) -> str | None:
    path = Path(image_directory)
    if path.parent.name.casefold() != "img":
        return None
    return str(path.parent.parent)


def catalog_tables(metadata: Any | None = None) -> dict[str, Any]:
    sa, _ = _sqlalchemy()
    metadata = metadata or sa.MetaData()
    projects = sa.Table(
        "projects",
        metadata,
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("width", sa.Integer, nullable=False),
        sa.Column("height", sa.Integer, nullable=False),
        sa.Column("orientation", sa.Text, nullable=False),
        sa.Column("state", sa.Text, nullable=False),
        sa.Column("source_directory", sa.Text),
        sa.Column("derived_directory", sa.Text),
        sa.Column("revision", sa.BigInteger, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("project_id", sa.Uuid(as_uuid=False), primary_key=True),
    )
    layers = sa.Table(
        "layers",
        metadata,
        sa.Column("project_name", sa.Text, nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("layer_type", sa.Text, nullable=False),
        sa.Column("sort_order", sa.Integer, nullable=False),
        sa.Column("state", sa.Text, nullable=False),
        sa.Column("image_directory", sa.Text),
        sa.Column("revision", sa.BigInteger, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("layer_id", sa.Uuid(as_uuid=False), primary_key=True),
    )
    representations = sa.Table(
        "representations",
        metadata,
        sa.Column("project_name", sa.Text, nullable=False),
        sa.Column("layer_name", sa.Text, nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("purpose", sa.Text, nullable=False),
        sa.Column("state", sa.Text, nullable=False),
        sa.Column("source", sa.Text),
        sa.Column("active", sa.Boolean, nullable=False),
        sa.Column("revision", sa.BigInteger, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("layer_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("representation_id", sa.Uuid(as_uuid=False), primary_key=True),
    )
    images = _file_table(metadata, "images")
    vectors = _file_table(metadata, "vectors")
    return {
        "projects": projects,
        "layers": layers,
        "representations": representations,
        "images": images,
        "vectors": vectors,
    }


def _file_table(metadata: Any, name: str) -> Any:
    sa, _ = _sqlalchemy()
    return sa.Table(
        name,
        metadata,
        sa.Column("project_name", sa.Text, nullable=False),
        sa.Column("layer_name", sa.Text, nullable=False),
        sa.Column("representation_name", sa.Text, nullable=False),
        sa.Column("x", sa.Integer),
        sa.Column("y", sa.Integer),
        sa.Column("filename", sa.Text, nullable=False),
        sa.Column("file_uri", sa.Text),
        sa.Column("media_type", sa.Text),
        sa.Column("size_bytes", sa.BigInteger),
        sa.Column("series_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("layer_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("representation_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("frame_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.PrimaryKeyConstraint("representation_id", "frame_id"),
    )


class CatalogProjection:
    """Upserts the columns a person reads in a database client."""

    def __init__(self, tables: dict[str, Any] | None = None) -> None:
        self.tables = tables or catalog_tables()

    def sync(self, connection: Any, kind: str, model: Any, values: dict[str, Any]) -> None:
        if kind == "project":
            self._project(connection, model, values)
        elif kind == "layer":
            self._layer(connection, model, values)
        elif kind == "representation":
            self._representation(connection, model, values)
        elif kind == "artifact_series":
            self._frame(connection, model)
        elif kind == "artifact_version":
            self._version(connection, model, values)

    def record_directories(
        self,
        connection: Any,
        project_id: str,
        *,
        source_directory: str,
        derived_directory: str,
    ) -> None:
        sa, _ = _sqlalchemy()
        projects = self.tables["projects"]
        connection.execute(
            sa.update(projects)
            .where(projects.c.project_id == project_id)
            .values(source_directory=source_directory, derived_directory=derived_directory)
        )

    def _upsert(
        self,
        connection: Any,
        table: Any,
        primary_key: str,
        row: dict[str, Any],
        *,
        preserve: tuple[str, ...] = (),
    ) -> None:
        insert = _dialect_insert(connection)
        payload = {
            key: value
            for key, value in row.items()
            if key != primary_key and key not in preserve
        }
        connection.execute(
            insert(table)
            .values(**row)
            .on_conflict_do_update(index_elements=[table.c[primary_key]], set_=payload)
        )

    def _project_name(self, connection: Any, project_id: str) -> str:
        sa, _ = _sqlalchemy()
        projects = self.tables["projects"]
        name = connection.execute(
            sa.select(projects.c.name).where(projects.c.project_id == project_id)
        ).scalar_one_or_none()
        return "" if name is None else str(name)

    def _project_width(self, connection: Any, project_id: str) -> int:
        sa, _ = _sqlalchemy()
        projects = self.tables["projects"]
        width = connection.execute(
            sa.select(projects.c.width).where(projects.c.project_id == project_id)
        ).scalar_one_or_none()
        return 0 if width is None else int(width)

    def _layer_context(self, connection: Any, layer_id: str) -> tuple[str, str]:
        sa, _ = _sqlalchemy()
        layers = self.tables["layers"]
        row = connection.execute(
            sa.select(layers.c.project_name, layers.c.name).where(layers.c.layer_id == layer_id)
        ).one_or_none()
        if row is None:
            return "", ""
        return str(row.project_name), str(row.name)

    def _representation_context(
        self, connection: Any, representation_id: str
    ) -> tuple[str, str, str, str, str, str]:
        sa, _ = _sqlalchemy()
        table = self.tables["representations"]
        row = connection.execute(
            sa.select(
                table.c.project_name,
                table.c.layer_name,
                table.c.name,
                table.c.kind,
                table.c.project_id,
                table.c.layer_id,
            ).where(table.c.representation_id == representation_id)
        ).one_or_none()
        if row is None:
            return "", "", "", "", "", ""
        return tuple(str(item) for item in row)

    def _file_table(self, kind: str) -> Any | None:
        if kind == "image":
            return self.tables["images"]
        if kind == "vector":
            return self.tables["vectors"]
        return None

    def _project(self, connection: Any, model: Any, values: dict[str, Any]) -> None:
        sa, _ = _sqlalchemy()
        project_id = str(model.id)
        self._upsert(
            connection,
            self.tables["projects"],
            "project_id",
            {
                "name": model.name,
                "width": int(model.width),
                "height": int(model.height),
                "orientation": model.orientation.value,
                "state": model.state.value,
                "source_directory": None,
                "derived_directory": None,
                "revision": int(model.revision),
                "created_at": model.created_at,
                "updated_at": values["updated_at"],
                "project_id": project_id,
            },
            preserve=("source_directory", "derived_directory"),
        )
        for table_name in ("layers", "representations", "images", "vectors"):
            table = self.tables[table_name]
            if "project_id" not in table.c:
                continue
            connection.execute(
                sa.update(table).where(table.c.project_id == project_id).values(project_name=model.name)
            )

    def _layer(self, connection: Any, model: Any, values: dict[str, Any]) -> None:
        sa, _ = _sqlalchemy()
        layer_id = str(model.id)
        project_id = str(model.project_id)
        self._upsert(
            connection,
            self.tables["layers"],
            "layer_id",
            {
                "project_name": self._project_name(connection, project_id),
                "name": model.name,
                "layer_type": model.type.value,
                "sort_order": int(model.order),
                "state": model.state.value,
                "image_directory": None,
                "revision": int(model.revision),
                "created_at": model.created_at,
                "project_id": project_id,
                "layer_id": layer_id,
            },
            preserve=("image_directory",),
        )
        for table_name in ("representations", "images", "vectors"):
            table = self.tables[table_name]
            connection.execute(
                sa.update(table).where(table.c.layer_id == layer_id).values(layer_name=model.name)
            )

    def _representation(self, connection: Any, model: Any, values: dict[str, Any]) -> None:
        sa, _ = _sqlalchemy()
        project_name, layer_name = self._layer_context(connection, str(model.layer_id))
        self._upsert(
            connection,
            self.tables["representations"],
            "representation_id",
            {
                "project_name": project_name,
                "layer_name": layer_name,
                "name": model.name,
                "kind": model.kind.value,
                "purpose": model.purpose.value,
                "state": model.state.value,
                "source": model.source,
                "active": bool(values["active"]),
                "revision": int(model.revision),
                "created_at": model.created_at,
                "project_id": str(model.project_id),
                "layer_id": str(model.layer_id),
                "representation_id": str(model.id),
            },
        )
        files = self._file_table(model.kind.value)
        if files is not None:
            connection.execute(
                sa.update(files)
                .where(files.c.representation_id == str(model.id))
                .values(representation_name=model.name)
            )
        if model.purpose.value == "source" and model.source:
            layers = self.tables["layers"]
            connection.execute(
                sa.update(layers)
                .where(layers.c.layer_id == str(model.layer_id))
                .values(image_directory=model.source)
            )
            directory = project_directory_from_images(model.source)
            if directory is not None:
                projects = self.tables["projects"]
                connection.execute(
                    sa.update(projects)
                    .where(
                        projects.c.project_id == str(model.project_id),
                        projects.c.source_directory.is_(None),
                    )
                    .values(source_directory=directory)
                )

    def _frame(self, connection: Any, model: Any) -> None:
        if getattr(model, "frame_id", None) is None or model.representation_id is None:
            return
        project_name, layer_name, representation_name, kind, project_id, layer_id = self._representation_context(
            connection, str(model.representation_id)
        )
        files = self._file_table(kind)
        if files is None:
            return
        coordinates = frame_coordinates(model.name, self._project_width(connection, str(model.project_id)))
        x, y = (None, None) if coordinates is None else coordinates
        insert = _dialect_insert(connection)
        connection.execute(
            insert(files)
            .values(
                project_name=project_name,
                layer_name=layer_name,
                representation_name=representation_name,
                x=x,
                y=y,
                filename=model.name,
                file_uri=None,
                media_type=None,
                size_bytes=None,
                series_id=str(model.id),
                project_id=project_id or str(model.project_id),
                layer_id=layer_id or str(model.layer_id),
                representation_id=str(model.representation_id),
                frame_id=str(model.frame_id),
            )
            .on_conflict_do_update(
                index_elements=[files.c.representation_id, files.c.frame_id],
                set_={
                    "filename": model.name,
                    "x": x,
                    "y": y,
                    "representation_name": representation_name,
                    "series_id": str(model.id),
                },
            )
        )

    def _version(self, connection: Any, model: Any, values: dict[str, Any]) -> None:
        if not bool(values.get("active", True)):
            return
        sa, _ = _sqlalchemy()
        series_id = str(model.series_id)
        located = None
        for table in (self.tables["images"], self.tables["vectors"]):
            row = connection.execute(
                sa.select(table.c.representation_id, table.c.frame_id).where(table.c.series_id == series_id)
            ).first()
            if row is not None:
                located = (table, row)
                break
        if located is None:
            return
        table, row = located
        external = getattr(model, "external", None)
        connection.execute(
            sa.update(table)
            .where(
                table.c.representation_id == str(row.representation_id),
                table.c.frame_id == str(row.frame_id),
            )
            .values(
                filename=model.filename,
                media_type=model.media_type,
                size_bytes=int(model.size_bytes),
                file_uri=None if external is None else external.uri,
            )
        )


def _dialect_insert(connection: Any) -> Any:
    if connection.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    return insert
