"""Disposable spatial projection, derived entirely from current typed models."""
from __future__ import annotations

import sqlite3


_KEY = "matrix_coordinates_v1"
SCHEMA = """
CREATE TABLE IF NOT EXISTS matrix_coordinates (
    series_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    layer_id TEXT NOT NULL,
    representation_id TEXT NOT NULL,
    frame_id TEXT NOT NULL,
    version_id TEXT NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_matrix_coordinates_region
    ON matrix_coordinates(project_id, layer_id, representation_id, y, x);
"""

_VALID = """
    json_type(cursor_json, '$.parameters.x') = 'integer'
    AND json_type(cursor_json, '$.parameters.y') = 'integer'
    AND json_extract(cursor_json, '$.parameters.x') BETWEEN 1 AND width
    AND json_extract(cursor_json, '$.parameters.y') BETWEEN 1 AND height
"""

_REBUILD = f"""WITH RECURSIVE ancestry(
    series_id, project_id, layer_id, representation_id, frame_id, version_id,
    cursor_json, visited, width, height
) AS (
    SELECT s.entity_id, s.project_id, s.layer_id,
           json_extract(s.value_json, '$.representation_id'),
           json_extract(s.value_json, '$.frame_id'), v.entity_id,
           v.value_json, '|' || v.entity_id || '|',
           json_extract(p.value_json, '$.width'), json_extract(p.value_json, '$.height')
    FROM typed_current_models AS s
    JOIN typed_current_models AS p
      ON p.model_type = 'project' AND p.entity_id = s.project_id
    JOIN typed_current_models AS v INDEXED BY ix_typed_current_parent
      ON v.model_type = 'artifact_version' AND v.parent_id = s.entity_id AND v.active = 1
    WHERE s.model_type = 'artifact_series' AND s.archived = 0
      AND json_extract(s.value_json, '$.frame_id') IS NOT NULL
      AND json_extract(s.value_json, '$.representation_id') IS NOT NULL
    UNION ALL
    SELECT a.series_id, a.project_id, a.layer_id, a.representation_id,
           a.frame_id, a.version_id, parent.value_json,
           a.visited || parent.entity_id || '|', a.width, a.height
    FROM ancestry AS a
    JOIN typed_current_models AS parent
      ON parent.model_type = 'artifact_version'
     AND parent.entity_id = json_extract(a.cursor_json, '$.parent_version_id')
    WHERE NOT COALESCE(({_VALID}), 0)
      AND instr(a.visited, '|' || parent.entity_id || '|') = 0
), resolved AS (
    SELECT series_id, project_id, layer_id, representation_id, frame_id, version_id,
           json_extract(cursor_json, '$.parameters.x') AS x,
           json_extract(cursor_json, '$.parameters.y') AS y,
           ROW_NUMBER() OVER (
               PARTITION BY project_id, layer_id, representation_id, frame_id
               ORDER BY series_id DESC
           ) AS winner
    FROM ancestry WHERE COALESCE(({_VALID}), 0)
)
INSERT INTO matrix_coordinates(
    series_id, project_id, layer_id, representation_id, frame_id, version_id, x, y
)
SELECT series_id, project_id, layer_id, representation_id, frame_id, version_id, x, y
FROM resolved WHERE winner = 1
"""


def invalidate(connection: sqlite3.Connection) -> None:
    connection.execute("DELETE FROM index_metadata WHERE key = ?", (_KEY,))


def create_schema(connection: sqlite3.Connection) -> None:
    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'matrix_coordinates'"
    ).fetchone() is None:
        invalidate(connection)
    connection.executescript(SCHEMA)


def _current(connection: sqlite3.Connection) -> tuple[str, str | None]:
    values = dict(connection.execute(
        "SELECT key, value FROM index_metadata WHERE key IN ('checkpoint', ?)", (_KEY,)
    ).fetchall())
    return values.get("checkpoint", "0"), values.get(_KEY)


def ensure_snapshot(connection: sqlite3.Connection) -> None:
    """Keep subsequent coordinate/model reads in the same SQLite snapshot.

    Warm reads need no writer lock. A stale cache is rebuilt atomically after
    acquiring the writer slot and rechecking another reader's completed rebuild.
    """
    connection.execute("BEGIN")
    checkpoint, cached = _current(connection)
    if checkpoint == cached:
        return
    connection.rollback()
    connection.execute("BEGIN IMMEDIATE")
    checkpoint, cached = _current(connection)
    if checkpoint != cached:
        connection.execute("DELETE FROM matrix_coordinates")
        connection.execute(_REBUILD)
        connection.execute("INSERT OR REPLACE INTO index_metadata(key, value) VALUES (?, ?)", (_KEY, checkpoint))


def rows(connection: sqlite3.Connection, project_id, layer_id, representation_id, bounds):
    ensure_snapshot(connection)
    return connection.execute(
        """
        SELECT s.value_json AS series_json, v.value_json AS version_json, c.x, c.y
        FROM matrix_coordinates AS c INDEXED BY ix_matrix_coordinates_region
        JOIN typed_current_models AS s ON s.model_type = 'artifact_series' AND s.entity_id = c.series_id
        JOIN typed_current_models AS v ON v.model_type = 'artifact_version' AND v.entity_id = c.version_id
        WHERE c.project_id = ? AND c.layer_id = ? AND c.representation_id = ?
          AND c.x BETWEEN ? AND ? AND c.y BETWEEN ? AND ?
        ORDER BY c.series_id
        """,
        (str(project_id), str(layer_id), str(representation_id), bounds[0], bounds[2], bounds[1], bounds[3]),
    ).fetchall()


def has_frames(connection: sqlite3.Connection, project_id, layer_id, representation_id) -> bool:
    ensure_snapshot(connection)
    return connection.execute(
        """SELECT 1 FROM matrix_coordinates
           WHERE project_id = ? AND layer_id = ? AND representation_id = ? LIMIT 1""",
        (str(project_id), str(layer_id), str(representation_id)),
    ).fetchone() is not None
