"""Event-backed SQLite projections for local Kraken analysis runs."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kraken_core.analysis_bundle import stream_bundle_records
from kraken_core.analysis_protocol import AnalysisFrameResult, AnalysisScaleMode
from kraken_core.analysis_run_protocol import (
    AnalysisPartitionJobManifest,
    AnalysisPartitionResultManifest,
    AnalysisRunManifest,
    canonical_json,
)
from kraken_manager.infrastructure.blob import FilesystemBlobStore
from kraken_manager.infrastructure.filesystem.event_store import FilesystemEventStore
from kraken_manager.infrastructure.filesystem.layout import FileProjectLayout


@dataclass(frozen=True, slots=True)
class KrakenAnalysisRun:
    run_id: str
    project_id: str
    fingerprint: str
    state: str
    total_frames: int
    completed_frames: int
    failed_frames: int
    imported_partitions: int
    total_partitions: int
    manifest: AnalysisRunManifest


_SCHEMA = """
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
CREATE INDEX IF NOT EXISTS ix_kraken_analysis_runs_fingerprint ON analysis_runs(fingerprint);
CREATE TABLE IF NOT EXISTS analysis_sources (
    run_id TEXT NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    binding_key TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_version TEXT NOT NULL,
    display_name TEXT NOT NULL,
    PRIMARY KEY(run_id, binding_key)
);
CREATE TABLE IF NOT EXISTS analysis_partitions (
    partition_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    partition_index INTEGER NOT NULL,
    job_id TEXT NOT NULL UNIQUE,
    manifest_json TEXT NOT NULL,
    state TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 0,
    completed_frames INTEGER NOT NULL DEFAULT 0,
    failed_frames INTEGER NOT NULL DEFAULT 0,
    bundle_sha256 TEXT,
    result_json TEXT,
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
    PRIMARY KEY(run_id, frame_id)
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
    PRIMARY KEY(run_id, frame_id, metric_key),
    FOREIGN KEY(run_id, frame_id) REFERENCES analysis_frame_results(run_id, frame_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_kraken_analysis_metric_run_key
    ON analysis_metric_values(run_id, metric_key, goodness);
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
    PRIMARY KEY(run_id, metric_key)
);
CREATE TABLE IF NOT EXISTS analysis_artifacts (
    artifact_key TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    frame_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    recipe_fingerprint TEXT NOT NULL,
    blob_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


class FilesystemAnalysisStore:
    """Canonical events + disposable SQLite projection + immutable result blobs."""

    def __init__(self, catalog_root: str | Path, project_id: str) -> None:
        self.catalog_root = Path(catalog_root).resolve()
        self.project_id = str(project_id)
        self.layout = FileProjectLayout(self.catalog_root, self.project_id)
        self.layout.ensure_directories()
        self.events = FilesystemEventStore(self.catalog_root, self.project_id)
        self.blobs = FilesystemBlobStore.for_project(self.catalog_root, self.project_id)
        self.database = self.layout.index_path
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def _append_event(self, run_id: str, event_type: str, payload: dict[str, object]) -> None:
        stream_id = f"analysis-run:{run_id}"
        self.events.append(
            stream_id,
            expected_revision=self.events.current_revision(stream_id),
            events=(
                {
                    "event_type": event_type,
                    "payload": payload,
                    "recorded_at": _now(),
                    "idempotency_key": payload.get("idempotency_key"),
                },
            ),
        )

    def create_run(
        self,
        manifest: AnalysisRunManifest,
        partitions: tuple[AnalysisPartitionJobManifest, ...],
    ) -> KrakenAnalysisRun:
        if manifest.project_id != self.project_id:
            raise ValueError("Analysis run belongs to another Kraken project")
        expected_frame_ids = tuple(item for partition in partitions for item in (frame.frame_id for frame in partition.frames))
        if expected_frame_ids != manifest.frame_ids:
            raise ValueError("Analysis partitions do not reproduce the ordered run frame selection")
        if len(partitions) != manifest.partition_count:
            raise ValueError("Analysis partition count does not match the run manifest")
        with self._connect() as connection:
            existing = connection.execute("SELECT fingerprint FROM analysis_runs WHERE run_id=?", (manifest.run_id,)).fetchone()
        if existing is not None:
            if existing["fingerprint"] != manifest.fingerprint:
                raise ValueError("Analysis run id already belongs to another manifest")
            result = self.get_run(manifest.run_id)
            assert result is not None
            return result

        payload = {
            "idempotency_key": f"analysis-create:{manifest.run_id}:{manifest.fingerprint}",
            "manifest": manifest.to_payload(),
            "partitions": [partition.to_payload() for partition in partitions],
        }
        self._append_event(manifest.run_id, "analysis_run.created", payload)
        self._project_run_created(manifest, partitions)
        result = self.get_run(manifest.run_id)
        assert result is not None
        return result

    def _project_run_created(
        self,
        manifest: AnalysisRunManifest,
        partitions: tuple[AnalysisPartitionJobManifest, ...],
    ) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT OR IGNORE INTO analysis_runs(
                    run_id, project_id, fingerprint, manifest_json, state, total_frames,
                    completed_frames, failed_frames, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'queued', ?, 0, 0, ?, ?)
                """,
                (
                    manifest.run_id,
                    manifest.project_id,
                    manifest.fingerprint,
                    canonical_json(manifest.to_payload()),
                    len(manifest.frame_ids),
                    now,
                    now,
                ),
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO analysis_sources(
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
            connection.executemany(
                """
                INSERT OR IGNORE INTO analysis_partitions(
                    partition_id, run_id, partition_index, job_id, manifest_json, state, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'queued', ?)
                """,
                (
                    (
                        partition.partition_id,
                        partition.run_id,
                        partition.partition_index,
                        partition.job_id,
                        canonical_json(partition.to_payload()),
                        now,
                    )
                    for partition in partitions
                ),
            )

    def import_partition(self, result: AnalysisPartitionResultManifest, bundle_path: str | Path) -> bool:
        with self._connect() as connection:
            partition = connection.execute(
                "SELECT * FROM analysis_partitions WHERE partition_id=?", (result.partition_id,)
            ).fetchone()
        if partition is None:
            raise KeyError(f"Unknown Kraken analysis partition: {result.partition_id}")
        job = AnalysisPartitionJobManifest.from_payload(json.loads(partition["manifest_json"]))
        if (result.job_id, result.run_id, result.project_id) != (job.job_id, job.run_id, job.project_id):
            raise ValueError("Analysis partition result does not match its submitted job")
        if result.bundle is None:
            raise ValueError("Kraken can only import an analysis result that contains frame records")
        if partition["state"] == "imported":
            if partition["bundle_sha256"] != result.bundle.sha256:
                raise ValueError("Imported partition was repeated with different content")
            return False

        stored = self.blobs.put_file(bundle_path, expected_sha256=result.bundle.sha256)
        with self.blobs.open(stored.blob) as stream:
            frames = tuple(
                stream_bundle_records(
                    stream,
                    result.bundle,
                    expected_frame_ids=(frame.frame_id for frame in job.frames),
                )
            )
        self._append_event(
            result.run_id,
            "analysis_partition.imported",
            {
                "idempotency_key": f"analysis-import:{result.partition_id}:{result.bundle.sha256}",
                "result": result.to_payload(),
                "blob_sha256": stored.blob.sha256,
            },
        )
        self._project_partition_imported(result, stored.blob.sha256, frames)
        return True

    def _project_partition_imported(
        self,
        result: AnalysisPartitionResultManifest,
        blob_sha256: str,
        frames: tuple[AnalysisFrameResult, ...],
    ) -> None:
        completed = sum(frame.status == "ready" for frame in frames)
        failed = len(frames) - completed
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for frame in frames:
                connection.execute(
                    """
                    INSERT INTO analysis_frame_results(
                        run_id, partition_id, frame_id, x, y, status, message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, frame_id) DO UPDATE SET
                        partition_id=excluded.partition_id, x=excluded.x, y=excluded.y,
                        status=excluded.status, message=excluded.message
                    """,
                    (result.run_id, result.partition_id, frame.frame_id, frame.x, frame.y, frame.status, frame.message),
                )
                connection.execute(
                    "DELETE FROM analysis_metric_values WHERE run_id=? AND frame_id=?",
                    (result.run_id, frame.frame_id),
                )
                connection.executemany(
                    """
                    INSERT INTO analysis_metric_values(
                        run_id, frame_id, metric_key, raw_value, goodness, percentile, unit, higher_is_better
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            result.run_id,
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
                SET state='imported', completed_frames=?, failed_frames=?, bundle_sha256=?, result_json=?, updated_at=?
                WHERE partition_id=?
                """,
                (completed, failed, blob_sha256, canonical_json(result.to_payload()), _now(), result.partition_id),
            )
            complete = self._refresh_run(connection, result.run_id)
            if complete:
                self._finalize_scales(connection, result.run_id)

    @staticmethod
    def _refresh_run(connection: sqlite3.Connection, run_id: str) -> bool:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN state='imported' THEN 1 ELSE 0 END) AS imported,
                   COALESCE(SUM(completed_frames), 0) AS completed,
                   COALESCE(SUM(failed_frames), 0) AS failed
            FROM analysis_partitions WHERE run_id=?
            """,
            (run_id,),
        ).fetchone()
        total = int(row["total"] or 0)
        imported = int(row["imported"] or 0)
        failed = int(row["failed"] or 0)
        state = "partial" if total and imported == total and failed else "completed" if total and imported == total else "running"
        connection.execute(
            """
            UPDATE analysis_runs SET state=?, completed_frames=?, failed_frames=?, updated_at=? WHERE run_id=?
            """,
            (state, int(row["completed"] or 0), failed, _now(), run_id),
        )
        return bool(total and imported == total)

    @staticmethod
    def _finalize_scales(connection: sqlite3.Connection, run_id: str) -> None:
        metric_keys = [
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT metric_key FROM analysis_metric_values WHERE run_id=?", (run_id,)
            )
        ]
        for metric_key in metric_keys:
            rows = connection.execute(
                """
                SELECT frame_id, goodness FROM analysis_metric_values
                WHERE run_id=? AND metric_key=? ORDER BY goodness, frame_id
                """,
                (run_id, metric_key),
            ).fetchall()
            values = [float(row["goodness"]) for row in rows]
            p05, p50, p95 = (_percentile(values, value) for value in (5.0, 50.0, 95.0))
            low, high = p05, p95
            if high - low < 0.01:
                midpoint = (high + low) / 2.0
                low, high = max(0.0, midpoint - 0.005), min(1.0, midpoint + 0.005)
                if high <= low:
                    low, high = 0.0, 1.0
            connection.execute(
                """
                INSERT OR REPLACE INTO analysis_metric_scales(
                    run_id, metric_key, mode, low, high, p05, p50, p95, clipped_low, clipped_high
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    metric_key,
                    AnalysisScaleMode.WITHIN_RUN.value,
                    low,
                    high,
                    p05,
                    p50,
                    p95,
                    sum(value < low for value in values),
                    sum(value > high for value in values),
                ),
            )
            denominator = max(1, len(rows) - 1)
            connection.executemany(
                """
                UPDATE analysis_metric_values SET percentile=?
                WHERE run_id=? AND frame_id=? AND metric_key=?
                """,
                (
                    (100.0 * index / denominator, run_id, str(row["frame_id"]), metric_key)
                    for index, row in enumerate(rows)
                ),
            )

    def get_run(self, run_id: str) -> KrakenAnalysisRun | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT r.*,
                       COUNT(p.partition_id) AS total_partitions,
                       SUM(CASE WHEN p.state='imported' THEN 1 ELSE 0 END) AS imported_partitions
                FROM analysis_runs AS r
                LEFT JOIN analysis_partitions AS p ON p.run_id=r.run_id
                WHERE r.run_id=? GROUP BY r.run_id
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return KrakenAnalysisRun(
            run_id=str(row["run_id"]),
            project_id=str(row["project_id"]),
            fingerprint=str(row["fingerprint"]),
            state=str(row["state"]),
            total_frames=int(row["total_frames"]),
            completed_frames=int(row["completed_frames"]),
            failed_frames=int(row["failed_frames"]),
            imported_partitions=int(row["imported_partitions"] or 0),
            total_partitions=int(row["total_partitions"] or 0),
            manifest=AnalysisRunManifest.from_payload(json.loads(row["manifest_json"])),
        )

    def list_runs(self) -> tuple[KrakenAnalysisRun, ...]:
        with self._connect() as connection:
            ids = [str(row[0]) for row in connection.execute("SELECT run_id FROM analysis_runs ORDER BY created_at DESC")]
        return tuple(run for run_id in ids if (run := self.get_run(run_id)) is not None)

    def frame_results(self, run_id: str, metric_key: str) -> tuple[sqlite3.Row, ...]:
        with self._connect() as connection:
            return tuple(
                connection.execute(
                    """
                    SELECT f.frame_id, f.x, f.y, f.status, f.message,
                           m.raw_value, m.goodness, m.percentile
                    FROM analysis_frame_results AS f
                    LEFT JOIN analysis_metric_values AS m
                      ON m.run_id=f.run_id AND m.frame_id=f.frame_id AND m.metric_key=?
                    WHERE f.run_id=? ORDER BY f.y, f.x, f.frame_id
                    """,
                    (metric_key, run_id),
                ).fetchall()
            )

    def retryable_partitions(self, run_id: str) -> tuple[AnalysisPartitionJobManifest, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT manifest_json FROM analysis_partitions
                WHERE run_id=? AND state!='imported' ORDER BY partition_index
                """,
                (run_id,),
            ).fetchall()
        return tuple(AnalysisPartitionJobManifest.from_payload(json.loads(row[0])) for row in rows)

    def rebuild(self) -> None:
        events = tuple(self.events.iter_project())
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=OFF")
            for table in (
                "analysis_artifacts",
                "analysis_metric_scales",
                "analysis_metric_values",
                "analysis_frame_results",
                "analysis_partitions",
                "analysis_sources",
                "analysis_runs",
            ):
                connection.execute(f"DELETE FROM {table}")
            connection.execute("PRAGMA foreign_keys=ON")
        for event in events:
            event_type = str(event.data.get("event_type", ""))
            payload = event.data.get("payload")
            if not isinstance(payload, dict):
                continue
            if event_type == "analysis_run.created":
                raw_manifest = payload.get("manifest")
                raw_partitions = payload.get("partitions")
                if isinstance(raw_manifest, dict) and isinstance(raw_partitions, list):
                    self._project_run_created(
                        AnalysisRunManifest.from_payload(raw_manifest),
                        tuple(
                            AnalysisPartitionJobManifest.from_payload(item)
                            for item in raw_partitions
                            if isinstance(item, dict)
                        ),
                    )
            elif event_type == "analysis_partition.imported":
                raw_result = payload.get("result")
                blob_sha256 = str(payload.get("blob_sha256", ""))
                if not isinstance(raw_result, dict):
                    continue
                result = AnalysisPartitionResultManifest.from_payload(raw_result)
                with self._connect() as connection:
                    row = connection.execute(
                        "SELECT manifest_json FROM analysis_partitions WHERE partition_id=?", (result.partition_id,)
                    ).fetchone()
                if row is None or result.bundle is None:
                    continue
                job = AnalysisPartitionJobManifest.from_payload(json.loads(row[0]))
                with self.blobs.open(blob_sha256) as stream:
                    frames = tuple(
                        stream_bundle_records(
                            stream,
                            result.bundle,
                            expected_frame_ids=(frame.frame_id for frame in job.frames),
                        )
                    )
                self._project_partition_imported(result, blob_sha256, frames)


__all__ = ["FilesystemAnalysisStore", "KrakenAnalysisRun"]
