"""Current and recorded-time temporal PostgreSQL projections."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Iterator

from kraken_manager.infrastructure.filesystem._codec import decode_model, encode_model

from .event_store import _sqlalchemy


def _projection_tables() -> tuple[Any, Any, Any]:
    sa, _ = _sqlalchemy()
    metadata = sa.MetaData()
    columns = (
        sa.Column("kind", sa.Text, primary_key=True),
        sa.Column("entity_id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("layer_id", sa.Uuid(as_uuid=False)),
        sa.Column("frame_id", sa.Uuid(as_uuid=False)),
        sa.Column("parent_id", sa.Uuid(as_uuid=False)),
        sa.Column("revision", sa.BigInteger, nullable=False),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("archived", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("sort_order", sa.Integer, nullable=False, server_default="0"),
        sa.Column("payload", sa.JSON, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    current = sa.Table("projections_current", metadata, *columns)
    temporal = sa.Table(
        "projections_temporal",
        metadata,
        sa.Column("sequence", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("entity_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("revision", sa.BigInteger, nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_to", sa.DateTime(timezone=True)),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("layer_id", sa.Uuid(as_uuid=False)),
        sa.Column("frame_id", sa.Uuid(as_uuid=False)),
        sa.Column("parent_id", sa.Uuid(as_uuid=False)),
        sa.Column("active", sa.Boolean, nullable=False),
        sa.Column("archived", sa.Boolean, nullable=False),
        sa.Column("sort_order", sa.Integer, nullable=False),
        sa.Column("payload", sa.JSON, nullable=False),
    )
    return metadata, current, temporal


class PostgresProjectionStore:
    def __init__(self, engine: Any, *, connection: Any | None = None, create_schema_for_tests: bool = False) -> None:
        self.engine = engine
        self.connection = connection
        self.metadata, self.current, self.temporal = _projection_tables()
        if create_schema_for_tests:
            self.metadata.create_all(engine)

    @contextmanager
    def _scope(self, *, write: bool = False) -> Iterator[Any]:
        if self.connection is not None:
            yield self.connection
            return
        context = self.engine.begin() if write else self.engine.connect()
        with context as connection:
            yield connection

    @staticmethod
    def _class(kind: str) -> type[Any]:
        from kraken_manager.domain.artifacts import ArtifactSeries, ArtifactVersion
        from kraken_manager.domain.project import Layer, Project, Representation
        from kraken_manager.domain.workflows import PluginJob, ReviewBatch

        return {
            "project": Project,
            "layer": Layer,
            "representation": Representation,
            "artifact_series": ArtifactSeries,
            "artifact_version": ArtifactVersion,
            "plugin_job": PluginJob,
            "review_batch": ReviewBatch,
        }[kind]

    @staticmethod
    def _archived(model: Any) -> bool:
        if bool(getattr(model, "archived", False)):
            return True
        state = getattr(model, "state", None)
        return getattr(state, "value", state) == "archived"

    def _metadata(self, kind: str, model: Any, *, active: bool | None, recorded_at: datetime) -> dict[str, Any]:
        project_id = getattr(model, "project_id", None)
        if kind == "project":
            project_id = model.id
        if project_id is None and kind == "artifact_version":
            series = self.get_artifact_series(model.series_id)
            if series is None:
                raise ValueError("Artifact series projection must exist before its versions")
            project_id = series.project_id
            layer_id = series.layer_id
        else:
            layer_id = getattr(model, "layer_id", None)
        if project_id is None:
            raise ValueError(f"{kind} projection has no project identity")
        return {
            "kind": kind,
            "entity_id": str(model.id),
            "project_id": str(project_id),
            "layer_id": None if layer_id is None else str(layer_id),
            "frame_id": None if getattr(model, "frame_id", None) is None else str(model.frame_id),
            "parent_id": None if getattr(model, "series_id", None) is None else str(model.series_id),
            "revision": int(getattr(model, "revision", 0)),
            "active": bool(getattr(model, "active", False) if active is None else active),
            "archived": self._archived(model),
            "sort_order": int(getattr(model, "order", 0)),
            "payload": encode_model(model),
            "updated_at": recorded_at,
        }

    def _save(self, kind: str, model: Any, *, active: bool | None = None, recorded_at: datetime | None = None) -> None:
        sa, pg_insert = _sqlalchemy()
        timestamp = recorded_at or datetime.now(UTC)
        values = self._metadata(kind, model, active=active, recorded_at=timestamp)
        with self._scope(write=True) as connection:
            if kind == "artifact_version" and values["active"]:
                previous_ids = connection.execute(
                    sa.select(self.current.c.entity_id).where(
                        self.current.c.kind == kind,
                        self.current.c.parent_id == values["parent_id"],
                        self.current.c.active.is_(True),
                        self.current.c.entity_id != values["entity_id"],
                    )
                ).scalars().all()
                if previous_ids:
                    connection.execute(
                        sa.update(self.temporal)
                        .where(
                            self.temporal.c.kind == kind,
                            self.temporal.c.entity_id.in_(previous_ids),
                            self.temporal.c.valid_to.is_(None),
                        )
                        .values(valid_to=timestamp)
                    )
                connection.execute(
                    sa.update(self.current)
                    .where(self.current.c.kind == kind, self.current.c.parent_id == values["parent_id"])
                    .values(active=False)
                )
            connection.execute(
                sa.update(self.temporal)
                .where(
                    self.temporal.c.kind == kind,
                    self.temporal.c.entity_id == values["entity_id"],
                    self.temporal.c.valid_to.is_(None),
                )
                .values(valid_to=timestamp)
            )
            temporal_values = {key: value for key, value in values.items() if key != "updated_at"}
            temporal_values["valid_from"] = timestamp
            temporal_values["valid_to"] = None
            connection.execute(sa.insert(self.temporal), temporal_values)
            update_values = {key: value for key, value in values.items() if key not in {"kind", "entity_id"}}
            connection.execute(
                pg_insert(self.current)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=[self.current.c.kind, self.current.c.entity_id],
                    set_=update_values,
                    where=self.current.c.revision <= values["revision"],
                )
            )

    def _get(self, kind: str, entity_id: Any, *, as_of: datetime | None = None) -> Any | None:
        sa, _ = _sqlalchemy()
        if as_of is None:
            statement = sa.select(self.current.c.payload).where(
                self.current.c.kind == kind, self.current.c.entity_id == str(entity_id)
            )
        else:
            statement = (
                sa.select(self.temporal.c.payload)
                .where(
                    self.temporal.c.kind == kind,
                    self.temporal.c.entity_id == str(entity_id),
                    self.temporal.c.valid_from <= as_of,
                    sa.or_(self.temporal.c.valid_to.is_(None), self.temporal.c.valid_to > as_of),
                )
                .order_by(self.temporal.c.revision.desc())
                .limit(1)
            )
        with self._scope() as connection:
            payload = connection.execute(statement).scalar_one_or_none()
        return None if payload is None else decode_model(self._class(kind), payload)

    def _list(
        self,
        kind: str,
        field: str,
        value: Any,
        *,
        include_archived: bool = False,
        as_of: datetime | None = None,
    ) -> tuple[Any, ...]:
        if field not in {"project_id", "layer_id", "parent_id"}:
            raise ValueError("Unsupported projection list field")
        sa, _ = _sqlalchemy()
        table = self.current if as_of is None else self.temporal
        statement = sa.select(table.c.payload).where(
            table.c.kind == kind,
            getattr(table.c, field) == str(value),
        )
        if as_of is not None:
            statement = statement.where(
                table.c.valid_from <= as_of,
                sa.or_(table.c.valid_to.is_(None), table.c.valid_to > as_of),
            )
        if not include_archived:
            statement = statement.where(table.c.archived.is_(False))
        statement = statement.order_by(table.c.sort_order, table.c.entity_id)
        with self._scope() as connection:
            payloads = connection.execute(statement).scalars().all()
        model_class = self._class(kind)
        return tuple(decode_model(model_class, payload) for payload in payloads)

    def get_project(self, project_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get("project", project_id, as_of=as_of)

    def list_projects(self, *, include_archived: bool = False) -> tuple[Any, ...]:
        sa, _ = _sqlalchemy()
        statement = sa.select(self.current.c.payload).where(self.current.c.kind == "project")
        if not include_archived:
            statement = statement.where(self.current.c.archived.is_(False))
        statement = statement.order_by(self.current.c.entity_id)
        with self._scope() as connection:
            payloads = connection.execute(statement).scalars().all()
        model_class = self._class("project")
        projects = [decode_model(model_class, payload) for payload in payloads]
        return tuple(sorted(projects, key=lambda project: (project.name.casefold(), str(project.id))))

    def save_project(self, project: Any) -> None:
        self._save("project", project)

    def get_layer(self, layer_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get("layer", layer_id, as_of=as_of)

    def list_layers(
        self, project_id: Any, *, include_archived: bool = False, as_of: datetime | None = None
    ) -> tuple[Any, ...]:
        return self._list(
            "layer", "project_id", project_id, include_archived=include_archived, as_of=as_of
        )

    def save_layer(self, layer: Any) -> None:
        self._save("layer", layer)

    def get_representation(self, representation_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get("representation", representation_id, as_of=as_of)

    def list_representations(
        self, layer_id: Any, *, include_archived: bool = False, as_of: datetime | None = None
    ) -> tuple[Any, ...]:
        return self._list(
            "representation", "layer_id", layer_id, include_archived=include_archived, as_of=as_of
        )

    def save_representation(self, representation: Any) -> None:
        self._save("representation", representation)

    def get_artifact_series(self, series_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get("artifact_series", series_id, as_of=as_of)

    def save_artifact_series(self, series: Any) -> None:
        self._save("artifact_series", series)

    def get_artifact_version(self, version_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get("artifact_version", version_id, as_of=as_of)

    def get_active_artifact_version(self, series_id: Any, *, as_of: datetime | None = None) -> Any | None:
        sa, _ = _sqlalchemy()
        table = self.current if as_of is None else self.temporal
        statement = sa.select(table.c.payload).where(
            table.c.kind == "artifact_version",
            table.c.parent_id == str(series_id),
            table.c.active.is_(True),
        )
        if as_of is not None:
            statement = (
                statement.where(
                    table.c.valid_from <= as_of,
                    sa.or_(table.c.valid_to.is_(None), table.c.valid_to > as_of),
                )
                .order_by(table.c.valid_from.desc())
                .limit(1)
            )
        with self._scope() as connection:
            payload = connection.execute(statement).scalar_one_or_none()
        return None if payload is None else decode_model(self._class("artifact_version"), payload)

    def save_artifact_version(self, version: Any, *, activate: bool) -> None:
        self._save("artifact_version", version, active=activate)

    def save_plugin_job(self, job: Any) -> None:
        self._save("plugin_job", job)

    def get_plugin_job(self, job_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get("plugin_job", job_id, as_of=as_of)

    def get_review_batch(self, batch_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get("review_batch", batch_id, as_of=as_of)

    def save_review_batch(self, batch: Any) -> None:
        self._save("review_batch", batch)

    def list_active_review_batches(
        self, project_id: Any, layer_id: Any, *, as_of: datetime | None = None
    ) -> tuple[Any, ...]:
        terminal = {"completed", "cancelled"}
        return tuple(
            batch
            for batch in self._list(
                "review_batch", "project_id", project_id, include_archived=True, as_of=as_of
            )
            if str(batch.layer_id) == str(layer_id) and batch.state.value not in terminal
        )


__all__ = ["PostgresProjectionStore"]
