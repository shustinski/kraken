from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._atomic import fsync_directory
from ._codec import decode_model, encode_model
from .event_store import StoredEvent, _json_compatible
from .layout import FileProjectLayout


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc_value, traceback))
        finally:
            self.close()


@dataclass(frozen=True, slots=True)
class ProjectionMutation:
    namespace: str
    key: str
    value: Mapping[str, Any] | None

    @classmethod
    def delete(cls, namespace: str, key: str) -> ProjectionMutation:
        return cls(namespace=namespace, key=key, value=None)


class SQLiteProjectionStore:
    """Disposable current/temporal projection cache for a file project."""

    SCHEMA_VERSION = 1

    def __init__(self, catalog_root: str | Path, project_id: str) -> None:
        self.layout = FileProjectLayout(Path(catalog_root), project_id)
        self.layout.ensure_directories()
        self.path = self.layout.index_path
        self._ensure_schema()

    @staticmethod
    def _connect_path(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(path, timeout=30.0, factory=_ClosingConnection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _connect(self) -> sqlite3.Connection:
        return self._connect_path(self.path)

    @classmethod
    def _create_schema(cls, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE IF NOT EXISTS indexed_events (
                global_position INTEGER PRIMARY KEY,
                event_id TEXT NOT NULL UNIQUE,
                stream_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                event_type TEXT,
                recorded_at TEXT,
                event_json TEXT NOT NULL,
                UNIQUE(stream_id, revision)
            );
            CREATE INDEX IF NOT EXISTS ix_indexed_events_stream
                ON indexed_events(stream_id, revision);
            CREATE INDEX IF NOT EXISTS ix_indexed_events_time
                ON indexed_events(recorded_at, global_position);
            CREATE INDEX IF NOT EXISTS ix_indexed_events_type
                ON indexed_events(event_type, global_position);

            CREATE TABLE IF NOT EXISTS projection_history (
                namespace TEXT NOT NULL,
                entity_key TEXT NOT NULL,
                event_position INTEGER NOT NULL,
                recorded_at TEXT,
                value_json TEXT,
                is_deleted INTEGER NOT NULL CHECK(is_deleted IN (0, 1)),
                PRIMARY KEY(namespace, entity_key, event_position),
                FOREIGN KEY(event_position) REFERENCES indexed_events(global_position)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS ix_projection_history_position
                ON projection_history(namespace, event_position);
            CREATE INDEX IF NOT EXISTS ix_projection_history_time
                ON projection_history(namespace, recorded_at, entity_key);

            CREATE TABLE IF NOT EXISTS current_projections (
                namespace TEXT NOT NULL,
                entity_key TEXT NOT NULL,
                event_position INTEGER NOT NULL,
                recorded_at TEXT,
                value_json TEXT NOT NULL,
                PRIMARY KEY(namespace, entity_key),
                FOREIGN KEY(event_position) REFERENCES indexed_events(global_position)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS ix_current_projections_position
                ON current_projections(namespace, event_position);

            CREATE TABLE IF NOT EXISTS typed_model_history (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                project_id TEXT,
                layer_id TEXT,
                parent_id TEXT,
                archived INTEGER NOT NULL DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 0,
                sort_order INTEGER NOT NULL DEFAULT 0,
                recorded_at TEXT NOT NULL,
                value_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_typed_model_history_temporal
                ON typed_model_history(model_type, entity_id, recorded_at, sequence);

            CREATE TABLE IF NOT EXISTS typed_current_models (
                model_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                project_id TEXT,
                layer_id TEXT,
                parent_id TEXT,
                archived INTEGER NOT NULL DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 0,
                sort_order INTEGER NOT NULL DEFAULT 0,
                recorded_at TEXT NOT NULL,
                value_json TEXT NOT NULL,
                PRIMARY KEY(model_type, entity_id)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS ix_typed_current_project
                ON typed_current_models(model_type, project_id, archived, sort_order, entity_id);
            CREATE INDEX IF NOT EXISTS ix_typed_current_layer
                ON typed_current_models(model_type, layer_id, archived, sort_order, entity_id);
            CREATE INDEX IF NOT EXISTS ix_typed_current_parent
                ON typed_current_models(model_type, parent_id, active, entity_id);
            """
        )
        connection.execute(
            "INSERT OR REPLACE INTO index_metadata(key, value) VALUES ('schema_version', ?)",
            (str(cls.SCHEMA_VERSION),),
        )

    def _ensure_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._create_schema(connection)

    @staticmethod
    def _event_parts(event: StoredEvent | Mapping[str, Any]) -> tuple[dict[str, Any], int]:
        if isinstance(event, StoredEvent):
            return event.to_dict(), event.global_position
        value = dict(event)
        try:
            position = int(value.pop("global_position"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("indexed event requires a positive global_position") from exc
        if position <= 0:
            raise ValueError("global_position must be positive")
        return value, position

    @staticmethod
    def _canonical_json(value: Any) -> str:
        return json.dumps(_json_compatible(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _apply(
        cls,
        connection: sqlite3.Connection,
        event: StoredEvent | Mapping[str, Any],
        mutations: Iterable[ProjectionMutation],
    ) -> None:
        value, position = cls._event_parts(event)
        try:
            event_id = str(value["event_id"])
            stream_id = str(value["stream_id"])
            revision = int(value["revision"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("indexed event has an incomplete envelope") from exc
        event_type = value.get("event_type", value.get("type"))
        recorded_at = value.get("recorded_at")
        if isinstance(recorded_at, datetime):
            recorded_at = recorded_at.isoformat()
        event_json = cls._canonical_json(value)

        existing = connection.execute(
            "SELECT event_json FROM indexed_events WHERE global_position = ? OR event_id = ?",
            (position, event_id),
        ).fetchone()
        if existing is not None:
            if existing["event_json"] != event_json:
                raise ValueError(f"event index conflict at position {position}")
            return

        connection.execute(
            """
            INSERT INTO indexed_events(
                global_position, event_id, stream_id, revision, event_type, recorded_at, event_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (position, event_id, stream_id, revision, event_type, recorded_at, event_json),
        )

        for mutation in mutations:
            if not mutation.namespace or not mutation.key:
                raise ValueError("projection namespace and key must be non-empty")
            encoded = None if mutation.value is None else cls._canonical_json(mutation.value)
            connection.execute(
                """
                INSERT INTO projection_history(
                    namespace, entity_key, event_position, recorded_at, value_json, is_deleted
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    mutation.namespace,
                    mutation.key,
                    position,
                    recorded_at,
                    encoded,
                    int(mutation.value is None),
                ),
            )
            if mutation.value is None:
                connection.execute(
                    "DELETE FROM current_projections WHERE namespace = ? AND entity_key = ?",
                    (mutation.namespace, mutation.key),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO current_projections(
                        namespace, entity_key, event_position, recorded_at, value_json
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(namespace, entity_key) DO UPDATE SET
                        event_position = excluded.event_position,
                        recorded_at = excluded.recorded_at,
                        value_json = excluded.value_json
                    WHERE excluded.event_position >= current_projections.event_position
                    """,
                    (mutation.namespace, mutation.key, position, recorded_at, encoded),
                )
        connection.execute(
            "INSERT OR REPLACE INTO index_metadata(key, value) VALUES ('checkpoint', ?)",
            (str(position),),
        )

    def apply(
        self,
        event: StoredEvent | Mapping[str, Any],
        mutations: Iterable[ProjectionMutation] = (),
    ) -> None:
        with self._connect() as connection:
            self._apply(connection, event, mutations)

    def get(
        self,
        namespace: str,
        key: str,
        *,
        as_of_position: int | None = None,
        as_of: datetime | str | None = None,
    ) -> Mapping[str, Any] | None:
        if as_of_position is not None and as_of is not None:
            raise ValueError("choose either as_of_position or as_of")
        with self._connect() as connection:
            if as_of_position is None and as_of is None:
                row = connection.execute(
                    "SELECT value_json FROM current_projections WHERE namespace = ? AND entity_key = ?",
                    (namespace, key),
                ).fetchone()
                return None if row is None else json.loads(row["value_json"])

            if as_of_position is not None:
                row = connection.execute(
                    """
                    SELECT value_json, is_deleted FROM projection_history
                    WHERE namespace = ? AND entity_key = ? AND event_position <= ?
                    ORDER BY event_position DESC LIMIT 1
                    """,
                    (namespace, key, as_of_position),
                ).fetchone()
            else:
                timestamp = as_of.isoformat() if isinstance(as_of, datetime) else str(as_of)
                row = connection.execute(
                    """
                    SELECT value_json, is_deleted FROM projection_history
                    WHERE namespace = ? AND entity_key = ? AND recorded_at <= ?
                    ORDER BY recorded_at DESC, event_position DESC LIMIT 1
                    """,
                    (namespace, key, timestamp),
                ).fetchone()
            if row is None or row["is_deleted"]:
                return None
            return json.loads(row["value_json"])

    def scan(
        self,
        namespace: str,
        *,
        prefix: str = "",
        after_key: str | None = None,
        limit: int = 1000,
    ) -> list[tuple[str, Mapping[str, Any]]]:
        if limit <= 0:
            return []
        lower = max(prefix, after_key or "")
        upper = prefix + "\U0010ffff"
        operator = ">" if after_key is not None else ">="
        query = f"""
            SELECT entity_key, value_json FROM current_projections
            WHERE namespace = ? AND entity_key {operator} ? AND entity_key < ?
            ORDER BY entity_key LIMIT ?
        """
        with self._connect() as connection:
            rows = connection.execute(query, (namespace, lower, upper, limit)).fetchall()
        return [(row["entity_key"], json.loads(row["value_json"])) for row in rows]

    def history(
        self,
        *,
        after_position: int = 0,
        limit: int = 1000,
        stream_id: str | None = None,
    ) -> list[StoredEvent]:
        clauses = ["global_position > ?"]
        parameters: list[Any] = [after_position]
        if stream_id is not None:
            clauses.append("stream_id = ?")
            parameters.append(stream_id)
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT global_position, event_json FROM indexed_events
                WHERE {' AND '.join(clauses)}
                ORDER BY global_position LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [StoredEvent(json.loads(row["event_json"]), row["global_position"]) for row in rows]

    @staticmethod
    def _model_class(model_type: str) -> type[Any]:
        from kraken_manager.domain.artifacts import ArtifactSeries, ArtifactVersion
        from kraken_manager.domain.project import Layer, Project, Representation
        from kraken_manager.domain.workflows import PluginJob, ReviewBatch

        classes = {
            "project": Project,
            "layer": Layer,
            "representation": Representation,
            "artifact_series": ArtifactSeries,
            "artifact_version": ArtifactVersion,
            "plugin_job": PluginJob,
            "review_batch": ReviewBatch,
        }
        return classes[model_type]

    @staticmethod
    def _is_archived(model: Any) -> bool:
        if bool(getattr(model, "archived", False)):
            return True
        state = getattr(model, "state", None)
        return getattr(state, "value", state) == "archived"

    def _save_typed(
        self,
        model_type: str,
        model: Any,
        *,
        active: bool | None = None,
        recorded_at: datetime | None = None,
    ) -> None:
        entity_id = str(model.id)
        project_id = getattr(model, "project_id", None)
        layer_id = getattr(model, "layer_id", None)
        parent_id = getattr(model, "series_id", None)
        archived = self._is_archived(model)
        active_value = bool(getattr(model, "active", False) if active is None else active)
        sort_order = int(getattr(model, "order", 0))
        recorded_at_text = (recorded_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        value_json = self._canonical_json(encode_model(model))
        parameters = (
            model_type,
            entity_id,
            None if project_id is None else str(project_id),
            None if layer_id is None else str(layer_id),
            None if parent_id is None else str(parent_id),
            int(archived),
            int(active_value),
            sort_order,
            recorded_at_text,
            value_json,
        )
        with self._connect() as connection:
            if model_type == "artifact_version" and active_value:
                connection.execute(
                    "UPDATE typed_current_models SET active = 0 WHERE model_type = ? AND parent_id = ?",
                    (model_type, str(parent_id)),
                )
            connection.execute(
                """
                INSERT INTO typed_model_history(
                    model_type, entity_id, project_id, layer_id, parent_id,
                    archived, active, sort_order, recorded_at, value_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                parameters,
            )
            connection.execute(
                """
                INSERT INTO typed_current_models(
                    model_type, entity_id, project_id, layer_id, parent_id,
                    archived, active, sort_order, recorded_at, value_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_type, entity_id) DO UPDATE SET
                    project_id = excluded.project_id,
                    layer_id = excluded.layer_id,
                    parent_id = excluded.parent_id,
                    archived = excluded.archived,
                    active = excluded.active,
                    sort_order = excluded.sort_order,
                    recorded_at = excluded.recorded_at,
                    value_json = excluded.value_json
                """,
                parameters,
            )

    def _get_typed(
        self,
        model_type: str,
        entity_id: Any,
        *,
        as_of: datetime | None = None,
    ) -> Any | None:
        with self._connect() as connection:
            if as_of is None:
                row = connection.execute(
                    "SELECT value_json FROM typed_current_models WHERE model_type = ? AND entity_id = ?",
                    (model_type, str(entity_id)),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT value_json FROM typed_model_history
                    WHERE model_type = ? AND entity_id = ? AND recorded_at <= ?
                    ORDER BY recorded_at DESC, sequence DESC LIMIT 1
                    """,
                    (model_type, str(entity_id), as_of.isoformat()),
                ).fetchone()
        if row is None:
            return None
        return decode_model(self._model_class(model_type), json.loads(row["value_json"]))

    def _list_typed(
        self,
        model_type: str,
        field: str,
        value: Any,
        *,
        include_archived: bool,
        as_of: datetime | None = None,
    ) -> tuple[Any, ...]:
        if field not in {"project_id", "layer_id", "parent_id"}:
            raise ValueError("unsupported typed projection filter")
        archived_clause = "" if include_archived else "AND archived = 0"
        with self._connect() as connection:
            if as_of is None:
                rows = connection.execute(
                    f"""
                    SELECT value_json FROM typed_current_models
                    WHERE model_type = ? AND {field} = ? {archived_clause}
                    ORDER BY sort_order, entity_id
                    """,
                    (model_type, str(value)),
                ).fetchall()
            else:
                rows = connection.execute(
                    f"""
                    WITH ranked AS (
                        SELECT value_json, archived, sort_order, entity_id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY entity_id
                                   ORDER BY recorded_at DESC, sequence DESC
                               ) AS rank
                        FROM typed_model_history
                        WHERE model_type = ? AND {field} = ? AND recorded_at <= ?
                    )
                    SELECT value_json FROM ranked
                    WHERE rank = 1 {archived_clause}
                    ORDER BY sort_order, entity_id
                    """,
                    (model_type, str(value), as_of.isoformat()),
                ).fetchall()
        model_class = self._model_class(model_type)
        return tuple(decode_model(model_class, json.loads(row["value_json"])) for row in rows)

    # Typed ProjectionStore port -------------------------------------------------
    def get_project(self, project_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get_typed("project", project_id, as_of=as_of)

    def save_project(self, project: Any) -> None:
        self._save_typed("project", project)

    def get_layer(self, layer_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get_typed("layer", layer_id, as_of=as_of)

    def list_layers(
        self, project_id: Any, *, include_archived: bool = False, as_of: datetime | None = None
    ) -> tuple[Any, ...]:
        return self._list_typed(
            "layer", "project_id", project_id, include_archived=include_archived, as_of=as_of
        )

    def save_layer(self, layer: Any) -> None:
        self._save_typed("layer", layer)

    def get_representation(self, representation_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get_typed("representation", representation_id, as_of=as_of)

    def list_representations(
        self, layer_id: Any, *, include_archived: bool = False, as_of: datetime | None = None
    ) -> tuple[Any, ...]:
        return self._list_typed(
            "representation", "layer_id", layer_id, include_archived=include_archived, as_of=as_of
        )

    def save_representation(self, representation: Any) -> None:
        self._save_typed("representation", representation)

    def get_artifact_series(self, series_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get_typed("artifact_series", series_id, as_of=as_of)

    def list_artifact_series(
        self,
        project_id: Any,
        *,
        layer_id: Any | None = None,
        representation_id: Any | None = None,
        include_archived: bool = False,
        as_of: datetime | None = None,
    ) -> tuple[Any, ...]:
        values = self._list_typed(
            "artifact_series",
            "project_id" if layer_id is None else "layer_id",
            project_id if layer_id is None else layer_id,
            include_archived=include_archived,
            as_of=as_of,
        )
        return tuple(
            item
            for item in values
            if representation_id is None or item.representation_id == representation_id
        )

    def save_artifact_series(self, series: Any) -> None:
        self._save_typed("artifact_series", series)

    def get_artifact_version(self, version_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get_typed("artifact_version", version_id, as_of=as_of)

    def get_active_artifact_version(self, series_id: Any, *, as_of: datetime | None = None) -> Any | None:
        with self._connect() as connection:
            table = "typed_current_models" if as_of is None else "typed_model_history"
            time_clause = "" if as_of is None else "AND recorded_at <= ?"
            parameters = (str(series_id),) if as_of is None else (str(series_id), as_of.isoformat())
            row = connection.execute(
                f"""
                SELECT value_json FROM {table}
                WHERE model_type = 'artifact_version' AND parent_id = ? AND active = 1 {time_clause}
                ORDER BY recorded_at DESC{', sequence DESC' if as_of is not None else ''} LIMIT 1
                """,
                parameters,
            ).fetchone()
        if row is None:
            return None
        return decode_model(self._model_class("artifact_version"), json.loads(row["value_json"]))

    def save_artifact_version(self, version: Any, *, activate: bool) -> None:
        self._save_typed("artifact_version", version, active=activate)

    def save_plugin_job(self, job: Any) -> None:
        self._save_typed("plugin_job", job)

    def get_plugin_job(self, job_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get_typed("plugin_job", job_id, as_of=as_of)

    def get_review_batch(self, batch_id: Any, *, as_of: datetime | None = None) -> Any | None:
        return self._get_typed("review_batch", batch_id, as_of=as_of)

    def save_review_batch(self, batch: Any) -> None:
        self._save_typed("review_batch", batch)

    def list_active_review_batches(
        self, project_id: Any, layer_id: Any, *, as_of: datetime | None = None
    ) -> tuple[Any, ...]:
        terminal_states = {"completed", "cancelled"}
        return tuple(
            batch
            for batch in self._list_typed(
                "review_batch", "project_id", project_id, include_archived=True, as_of=as_of
            )
            if str(batch.layer_id) == str(layer_id)
            and getattr(batch.state, "value", batch.state) not in terminal_states
        )

    @property
    def checkpoint(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM index_metadata WHERE key = 'checkpoint'"
            ).fetchone()
        return 0 if row is None else int(row["value"])

    def rebuild(
        self,
        events: Iterable[StoredEvent | Mapping[str, Any]],
        projector: Callable[[StoredEvent | Mapping[str, Any]], Iterable[ProjectionMutation]] | None = None,
    ) -> int:
        """Build a new index beside the old one and atomically swap it in."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix="read-rebuild-", suffix=".sqlite3", dir=self.path.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        count = 0
        try:
            connection = self._connect_path(temporary)
            try:
                connection.execute("PRAGMA journal_mode = DELETE")
                self._create_schema(connection)
                connection.commit()
                connection.execute("BEGIN IMMEDIATE")
                for event in events:
                    mutations = () if projector is None else projector(event)
                    self._apply(connection, event, mutations)
                    count += 1
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            fsync_directory(self.path.parent)
            return count
        finally:
            temporary.unlink(missing_ok=True)
            Path(f"{temporary}-journal").unlink(missing_ok=True)

    def destroy_cache(self) -> None:
        """Delete only the rebuildable index; authoritative events are untouched."""

        for suffix in ("", "-wal", "-shm"):
            Path(f"{self.path}{suffix}").unlink(missing_ok=True)
