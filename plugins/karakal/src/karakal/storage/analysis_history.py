"""Private SQLite history for standalone Karakal analysis runs."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kraken_core.analysis_protocol import AnalysisFrameResult, AnalysisScaleDefinition
from kraken_core.analysis_run_protocol import AnalysisPartitionJobManifest, AnalysisRunManifest


SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def default_history_database() -> Path:
    override = str(os.environ.get("KARAKAL_DATA_DIR", "")).strip()
    if override:
        root = Path(override).expanduser()
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"]) / "Karakal" / "data"
    else:
        root = Path.home() / ".local" / "share" / "karakal"
    return root / "analysis.sqlite3"


@dataclass(frozen=True, slots=True)
class StoredAnalysisRun:
    run_id: str
    project_id: str
    fingerprint: str
    state: str
    total_frames: int
    completed_frames: int
    failed_frames: int
    created_at: str
    updated_at: str
    manifest: AnalysisRunManifest


@dataclass(frozen=True, slots=True)
class StoredAnalysisPartition:
    partition_id: str
    run_id: str
    partition_index: int
    state: str
    attempt: int
    completed_frames: int
    failed_frames: int
    manifest: AnalysisPartitionJobManifest


_MIGRATION_V1 = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    state TEXT NOT NULL,
    total_frames INTEGER NOT NULL,
    completed_frames INTEGER NOT NULL DEFAULT 0,
    failed_frames INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_analysis_runs_fingerprint ON analysis_runs(fingerprint);
CREATE TABLE IF NOT EXISTS analysis_sources (
    run_id TEXT NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    binding_key TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_version TEXT NOT NULL,
    display_name TEXT NOT NULL,
    PRIMARY KEY (run_id, binding_key)
);
CREATE TABLE IF NOT EXISTS analysis_partitions (
    partition_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    partition_index INTEGER NOT NULL,
    job_id TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    state TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 0,
    completed_frames INTEGER NOT NULL DEFAULT 0,
    failed_frames INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    UNIQUE(run_id, partition_index)
);
CREATE TABLE IF NOT EXISTS analysis_frame_results (
    run_id TEXT NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    partition_id TEXT NOT NULL REFERENCES analysis_partitions(partition_id) ON DELETE CASCADE,
    frame_id TEXT NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    status TEXT NOT NULL,
    message TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    PRIMARY KEY (run_id, frame_id)
);
CREATE TABLE IF NOT EXISTS analysis_metric_values (
    run_id TEXT NOT NULL,
    frame_id TEXT NOT NULL,
    metric_key TEXT NOT NULL,
    raw_value REAL NOT NULL,
    goodness REAL NOT NULL,
    percentile REAL,
    unit TEXT NOT NULL,
    higher_is_better INTEGER NOT NULL,
    PRIMARY KEY (run_id, frame_id, metric_key),
    FOREIGN KEY (run_id, frame_id) REFERENCES analysis_frame_results(run_id, frame_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_analysis_metric_run_key ON analysis_metric_values(run_id, metric_key);
CREATE TABLE IF NOT EXISTS analysis_metric_scales (
    run_id TEXT NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    metric_key TEXT NOT NULL,
    mode TEXT NOT NULL,
    low REAL NOT NULL,
    high REAL NOT NULL,
    p05 REAL,
    p50 REAL,
    p95 REAL,
    clipped_low INTEGER NOT NULL,
    clipped_high INTEGER NOT NULL,
    PRIMARY KEY (run_id, metric_key)
);
CREATE TABLE IF NOT EXISTS analysis_artifacts (
    artifact_key TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    frame_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    recipe_fingerprint TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class AnalysisHistoryStore:
    """Transaction-safe standalone history; never stores source image bytes."""

    def __init__(self, database: str | Path | None = None) -> None:
        self.database = Path(database or default_history_database()).expanduser().resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def migrate(self) -> None:
        with self._connect() as connection:
            connection.executescript(_MIGRATION_V1)
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, _utc_now()),
            )

    def schema_version(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT MAX(version) AS version FROM schema_migrations").fetchone()
        return int(row["version"] or 0)

    def create_run(self, manifest: AnalysisRunManifest, *, state: str = "queued") -> None:
        now = _utc_now()
        payload = json.dumps(manifest.to_payload(), ensure_ascii=False, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO analysis_runs(
                    run_id, project_id, fingerprint, manifest_json, state, total_frames,
                    completed_frames, failed_frames, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?)
                """,
                (
                    manifest.run_id,
                    manifest.project_id,
                    manifest.fingerprint,
                    payload,
                    state,
                    len(manifest.frame_ids),
                    now,
                    now,
                ),
            )
            connection.executemany(
                """
                INSERT INTO analysis_sources(
                    run_id, binding_key, source_id, source_version, display_name
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        manifest.run_id,
                        source.binding_key,
                        source.source_id,
                        source.source_version,
                        source.display_name,
                    )
                    for source in manifest.source_bindings
                ),
            )

    def save_partition(self, manifest: AnalysisPartitionJobManifest, *, state: str = "queued") -> None:
        payload = json.dumps(manifest.to_payload(), ensure_ascii=False, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO analysis_partitions(
                    partition_id, run_id, partition_index, job_id, manifest_json, state, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(partition_id) DO UPDATE SET
                    manifest_json=excluded.manifest_json,
                    state=CASE WHEN analysis_partitions.state='imported' THEN 'imported' ELSE excluded.state END,
                    updated_at=excluded.updated_at
                """,
                (
                    manifest.partition_id,
                    manifest.run_id,
                    manifest.partition_index,
                    manifest.job_id,
                    payload,
                    state,
                    _utc_now(),
                ),
            )

    def mark_partition_running(self, partition_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE analysis_partitions
                SET state='running', attempt=attempt+1, updated_at=?
                WHERE partition_id=? AND state!='imported'
                """,
                (_utc_now(), partition_id),
            )

    def import_partition(
        self,
        partition_id: str,
        frames: Iterable[AnalysisFrameResult],
        *,
        terminal_state: str = "imported",
    ) -> bool:
        """Import once and return False when this partition was already imported."""

        frame_results = tuple(frames)
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            partition = connection.execute(
                "SELECT run_id, state FROM analysis_partitions WHERE partition_id=?", (partition_id,)
            ).fetchone()
            if partition is None:
                raise KeyError(f"Unknown analysis partition: {partition_id}")
            if partition["state"] == "imported":
                return False
            run_id = str(partition["run_id"])
            completed = 0
            failed = 0
            for frame in frame_results:
                completed += int(frame.status == "ready")
                failed += int(frame.status != "ready")
                connection.execute(
                    """
                    INSERT INTO analysis_frame_results(
                        run_id, partition_id, frame_id, x, y, status, message, imported_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, frame_id) DO UPDATE SET
                        partition_id=excluded.partition_id,
                        x=excluded.x,
                        y=excluded.y,
                        status=excluded.status,
                        message=excluded.message,
                        imported_at=excluded.imported_at
                    """,
                    (run_id, partition_id, frame.frame_id, frame.x, frame.y, frame.status, frame.message, now),
                )
                connection.execute(
                    "DELETE FROM analysis_metric_values WHERE run_id=? AND frame_id=?", (run_id, frame.frame_id)
                )
                connection.executemany(
                    """
                    INSERT INTO analysis_metric_values(
                        run_id, frame_id, metric_key, raw_value, goodness, percentile, unit, higher_is_better
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            run_id,
                            frame.frame_id,
                            metric.key,
                            metric.raw_value,
                            metric.goodness,
                            metric.percentile,
                            metric.unit,
                            int(metric.higher_is_better),
                        )
                        for metric in frame.metrics
                    ),
                )
            connection.execute(
                """
                UPDATE analysis_partitions
                SET state=?, completed_frames=?, failed_frames=?, updated_at=?
                WHERE partition_id=?
                """,
                (terminal_state, completed, failed, now, partition_id),
            )
            self._refresh_run_counts(connection, run_id, now)
        return True

    @staticmethod
    def _refresh_run_counts(connection: sqlite3.Connection, run_id: str, now: str) -> None:
        counts = connection.execute(
            """
            SELECT
                COALESCE(SUM(completed_frames), 0) AS completed,
                COALESCE(SUM(failed_frames), 0) AS failed,
                SUM(CASE WHEN state='imported' THEN 1 ELSE 0 END) AS imported,
                COUNT(*) AS total
            FROM analysis_partitions WHERE run_id=?
            """,
            (run_id,),
        ).fetchone()
        imported = int(counts["imported"] or 0)
        partition_total = int(counts["total"] or 0)
        failed = int(counts["failed"] or 0)
        if partition_total and imported == partition_total:
            state = "partial" if failed else "completed"
        elif imported:
            state = "running"
        else:
            state = "queued"
        connection.execute(
            """
            UPDATE analysis_runs
            SET state=?, completed_frames=?, failed_frames=?, updated_at=? WHERE run_id=?
            """,
            (state, int(counts["completed"] or 0), failed, now, run_id),
        )

    def save_scales(self, run_id: str, scales: Iterable[AnalysisScaleDefinition]) -> None:
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO analysis_metric_scales(
                    run_id, metric_key, mode, low, high, p05, p50, p95, clipped_low, clipped_high
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, metric_key) DO UPDATE SET
                    mode=excluded.mode, low=excluded.low, high=excluded.high,
                    p05=excluded.p05, p50=excluded.p50, p95=excluded.p95,
                    clipped_low=excluded.clipped_low, clipped_high=excluded.clipped_high
                """,
                (
                    (
                        run_id,
                        scale.metric_key,
                        scale.mode.value,
                        scale.low,
                        scale.high,
                        scale.p05,
                        scale.p50,
                        scale.p95,
                        scale.clipped_low,
                        scale.clipped_high,
                    )
                    for scale in scales
                ),
            )

    def list_runs(self, *, limit: int = 100) -> tuple[StoredAnalysisRun, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM analysis_runs ORDER BY created_at DESC LIMIT ?", (max(1, int(limit)),)
            ).fetchall()
        return tuple(self._stored_run(row) for row in rows)

    def get_run(self, run_id: str) -> StoredAnalysisRun | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM analysis_runs WHERE run_id=?", (run_id,)).fetchone()
        return None if row is None else self._stored_run(row)

    @staticmethod
    def _stored_run(row: sqlite3.Row) -> StoredAnalysisRun:
        return StoredAnalysisRun(
            run_id=str(row["run_id"]),
            project_id=str(row["project_id"]),
            fingerprint=str(row["fingerprint"]),
            state=str(row["state"]),
            total_frames=int(row["total_frames"]),
            completed_frames=int(row["completed_frames"]),
            failed_frames=int(row["failed_frames"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            manifest=AnalysisRunManifest.from_payload(json.loads(row["manifest_json"])),
        )

    def incomplete_partitions(self, run_id: str) -> tuple[StoredAnalysisPartition, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM analysis_partitions
                WHERE run_id=? AND state!='imported' ORDER BY partition_index
                """,
                (run_id,),
            ).fetchall()
        return tuple(
            StoredAnalysisPartition(
                partition_id=str(row["partition_id"]),
                run_id=str(row["run_id"]),
                partition_index=int(row["partition_index"]),
                state=str(row["state"]),
                attempt=int(row["attempt"]),
                completed_frames=int(row["completed_frames"]),
                failed_frames=int(row["failed_frames"]),
                manifest=AnalysisPartitionJobManifest.from_payload(json.loads(row["manifest_json"])),
            )
            for row in rows
        )

    def frame_results(self, run_id: str, *, metric_key: str | None = None) -> tuple[sqlite3.Row, ...]:
        query = """
            SELECT f.frame_id, f.x, f.y, f.status, f.message,
                   m.metric_key, m.raw_value, m.goodness, m.percentile
            FROM analysis_frame_results AS f
            LEFT JOIN analysis_metric_values AS m
              ON m.run_id=f.run_id AND m.frame_id=f.frame_id
            WHERE f.run_id=?
        """
        parameters: list[object] = [run_id]
        if metric_key is not None:
            query += " AND m.metric_key=?"
            parameters.append(metric_key)
        query += " ORDER BY f.y, f.x, f.frame_id"
        with self._connect() as connection:
            return tuple(connection.execute(query, parameters).fetchall())

    def delete_run(self, run_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM analysis_runs WHERE run_id=?", (run_id,))
        return cursor.rowcount > 0


__all__ = [
    "AnalysisHistoryStore",
    "SCHEMA_VERSION",
    "StoredAnalysisPartition",
    "StoredAnalysisRun",
    "default_history_database",
]
