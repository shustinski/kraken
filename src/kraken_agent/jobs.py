"""SQLite-backed job state and staging management for Kraken Agent.

The agent owns only disposable staging data.  Importing a successful result
into an authoritative project is an application-service responsibility.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from contextlib import contextmanager
from typing import Iterable, Iterator

from kraken_core.plugin_protocol import (
    PluginJobManifest,
    PluginJobManifestV2,
    PluginResultManifest,
    PluginResultPublicationV2,
    parse_plugin_job_json,
    parse_plugin_result_json,
    safe_relative_path,
)
from kraken_core.safe_files import (
    UnsafeFilesystemPath,
    contained_path,
    ensure_regular_directory,
    is_link_or_reparse,
    make_contained_directories,
    open_exclusive_write,
    open_regular_read,
)


class AgentJobState(StrEnum):
    QUEUED = "queued"
    STAGING = "staging"
    RUNNING = "running"
    WAITING_FOR_USER = "waiting_for_user"
    IMPORTING = "importing"
    PARTIAL = "partial"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    AWAITING_AUTHORIZATION = "awaiting_authorization"
    RECOVERY_REQUIRED = "recovery_required"


TERMINAL_STATES = {
    AgentJobState.SUCCEEDED,
    AgentJobState.FAILED,
    AgentJobState.CANCELLED,
}

_TRANSITIONS: dict[AgentJobState, frozenset[AgentJobState]] = {
    AgentJobState.QUEUED: frozenset({AgentJobState.STAGING, AgentJobState.CANCELLED, AgentJobState.FAILED}),
    AgentJobState.STAGING: frozenset({AgentJobState.RUNNING, AgentJobState.CANCELLED, AgentJobState.FAILED}),
    AgentJobState.RUNNING: frozenset(
        {
            AgentJobState.WAITING_FOR_USER,
            AgentJobState.IMPORTING,
            AgentJobState.PARTIAL,
            AgentJobState.CANCELLED,
            AgentJobState.FAILED,
            AgentJobState.RECOVERY_REQUIRED,
        }
    ),
    AgentJobState.WAITING_FOR_USER: frozenset(
        {AgentJobState.RUNNING, AgentJobState.IMPORTING, AgentJobState.CANCELLED, AgentJobState.RECOVERY_REQUIRED}
    ),
    AgentJobState.IMPORTING: frozenset(
        {
            AgentJobState.SUCCEEDED,
            AgentJobState.PARTIAL,
            AgentJobState.FAILED,
            AgentJobState.AWAITING_AUTHORIZATION,
            AgentJobState.RECOVERY_REQUIRED,
        }
    ),
    AgentJobState.PARTIAL: frozenset(
        {AgentJobState.IMPORTING, AgentJobState.CANCELLED, AgentJobState.FAILED}
    ),
    AgentJobState.AWAITING_AUTHORIZATION: frozenset(
        {AgentJobState.IMPORTING, AgentJobState.CANCELLED, AgentJobState.FAILED}
    ),
    AgentJobState.RECOVERY_REQUIRED: frozenset(
        {AgentJobState.QUEUED, AgentJobState.IMPORTING, AgentJobState.CANCELLED, AgentJobState.FAILED}
    ),
    AgentJobState.SUCCEEDED: frozenset(),
    AgentJobState.FAILED: frozenset(),
    AgentJobState.CANCELLED: frozenset(),
}


class JobStateError(RuntimeError):
    """Raised when a durable job transition is not valid."""


class DuplicateCallbackError(RuntimeError):
    """Raised when a callback key was reused with different content."""


@dataclass(frozen=True, slots=True)
class AgentJob:
    job_id: str
    state: AgentJobState
    manifest_json: str
    result_json: str | None
    created_at: str
    updated_at: str
    revision: int
    error: str | None = None

    @property
    def manifest(self) -> PluginJobManifest | PluginJobManifestV2:
        return parse_plugin_job_json(self.manifest_json)

    @property
    def result(self) -> PluginResultManifest | PluginResultPublicationV2 | None:
        return None if self.result_json is None else parse_plugin_result_json(self.result_json)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class DurableJobStore:
    """Small transactional store designed to survive UI and agent restarts."""

    def __init__(self, database: Path | str) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS ix_jobs_state_updated
                    ON jobs(state, updated_at);
                CREATE TABLE IF NOT EXISTS callbacks (
                    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                    callback_key TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    PRIMARY KEY(job_id, callback_key)
                );
                CREATE TABLE IF NOT EXISTS publications (
                    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                    publication_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    PRIMARY KEY(job_id, publication_id),
                    UNIQUE(job_id, sequence)
                );
                """
            )

    @staticmethod
    def _row(row: sqlite3.Row) -> AgentJob:
        return AgentJob(
            job_id=row["job_id"],
            state=AgentJobState(row["state"]),
            manifest_json=row["manifest_json"],
            result_json=row["result_json"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            revision=int(row["revision"]),
            error=row["error"],
        )

    def enqueue(self, manifest: PluginJobManifest | PluginJobManifestV2) -> AgentJob:
        now = _utc_now()
        canonical = manifest.to_json()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT * FROM jobs WHERE job_id=?", (manifest.job_id,)).fetchone()
            if existing is not None:
                if existing["manifest_json"] != canonical:
                    connection.rollback()
                    raise DuplicateCallbackError(f"Job {manifest.job_id} already exists with another manifest")
                connection.commit()
                return self._row(existing)
            connection.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, NULL, ?, ?, 0, NULL)",
                (manifest.job_id, AgentJobState.QUEUED.value, canonical, now, now),
            )
            connection.commit()
        return self.get(manifest.job_id)

    def get(self, job_id: str) -> AgentJob:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._row(row)

    def list(self, *, states: Iterable[AgentJobState] | None = None, limit: int = 100) -> tuple[AgentJob, ...]:
        if limit < 1 or limit > 10_000:
            raise ValueError("limit must be between 1 and 10000")
        values = tuple(state.value for state in states or ())
        query = "SELECT * FROM jobs"
        arguments: tuple[object, ...] = ()
        if values:
            placeholders = ",".join("?" for _ in values)
            query += f" WHERE state IN ({placeholders})"
            arguments = values
        query += " ORDER BY updated_at DESC LIMIT ?"
        arguments += (limit,)
        with self._connect() as connection:
            rows = connection.execute(query, arguments).fetchall()
        return tuple(self._row(row) for row in rows)

    def transition(
        self,
        job_id: str,
        target: AgentJobState,
        *,
        expected_revision: int,
        error: str | None = None,
    ) -> AgentJob:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(job_id)
            current = AgentJobState(row["state"])
            revision = int(row["revision"])
            if revision != expected_revision:
                connection.rollback()
                raise JobStateError(f"Expected revision {expected_revision}, found {revision}")
            if target not in _TRANSITIONS[current]:
                connection.rollback()
                raise JobStateError(f"Cannot transition {current.value} to {target.value}")
            connection.execute(
                "UPDATE jobs SET state=?, updated_at=?, revision=revision+1, error=? WHERE job_id=? AND revision=?",
                (target.value, _utc_now(), error, job_id, revision),
            )
            connection.commit()
        return self.get(job_id)

    def record_result(
        self,
        result: PluginResultManifest | PluginResultPublicationV2,
        *,
        callback_key: str,
        expected_revision: int,
    ) -> tuple[AgentJob, bool]:
        """Record a callback once; return ``(job, was_duplicate)``."""

        payload = result.to_json()
        payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (result.job_id,)).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(result.job_id)
            callback = connection.execute(
                "SELECT payload_sha256 FROM callbacks WHERE job_id=? AND callback_key=?",
                (result.job_id, callback_key),
            ).fetchone()
            if callback is not None:
                if callback["payload_sha256"] != payload_hash:
                    connection.rollback()
                    raise DuplicateCallbackError("Callback key was reused with a different result")
                connection.commit()
                return self._row(row), True
            if int(row["revision"]) != expected_revision:
                connection.rollback()
                raise JobStateError(f"Expected revision {expected_revision}, found {row['revision']}")
            if isinstance(result, PluginResultPublicationV2):
                existing_sequence = connection.execute(
                    "SELECT publication_id FROM publications WHERE job_id=? AND sequence=?",
                    (result.job_id, result.sequence),
                ).fetchone()
                if existing_sequence is not None:
                    connection.rollback()
                    raise DuplicateCallbackError(
                        "Publication sequence was reused with another publication ID"
                    )
            connection.execute(
                "INSERT INTO callbacks VALUES (?, ?, ?, ?)",
                (result.job_id, callback_key, payload_hash, _utc_now()),
            )
            if isinstance(result, PluginResultPublicationV2):
                connection.execute(
                    "INSERT INTO publications VALUES (?, ?, ?, ?, ?)",
                    (
                        result.job_id,
                        result.publication_id,
                        result.sequence,
                        payload,
                        _utc_now(),
                    ),
                )
            connection.execute(
                "UPDATE jobs SET result_json=?, updated_at=?, revision=revision+1 WHERE job_id=?",
                (payload, _utc_now(), result.job_id),
            )
            connection.commit()
        return self.get(result.job_id), False

    def list_publications(self, job_id: str) -> tuple[PluginResultPublicationV2, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM publications WHERE job_id=? ORDER BY sequence",
                (job_id,),
            ).fetchall()
        return tuple(
            PluginResultPublicationV2.from_json(str(row["payload_json"]))
            for row in rows
        )

    def recover_interrupted(self) -> int:
        """Mark jobs that cannot safely continue without operator inspection."""

        recoverable = (
            AgentJobState.STAGING.value,
            AgentJobState.RUNNING.value,
            AgentJobState.WAITING_FOR_USER.value,
            AgentJobState.IMPORTING.value,
        )
        with self._lock, self._connect() as connection:
            placeholders = ",".join("?" for _ in recoverable)
            cursor = connection.execute(
                f"UPDATE jobs SET state=?, updated_at=?, revision=revision+1 "
                f"WHERE state IN ({placeholders})",
                (AgentJobState.RECOVERY_REQUIRED.value, _utc_now(), *recoverable),
            )
            return cursor.rowcount


class StagingWorkspace:
    """Per-job staging with containment and checksum verification."""

    def __init__(self, root: Path | str, job_id: str) -> None:
        safe_id = "".join(char for char in job_id if char.isalnum() or char in "-_")
        if safe_id != job_id or not safe_id:
            raise ValueError("job_id is not safe for a workspace name")
        # ``resolve(strict=False)`` canonicalizes an existing staging-root
        # symlink once, but the job path itself is deliberately not resolved:
        # resolving it would hide a malicious symlink/reparse point.
        supplied_root = Path(root)
        if supplied_root.exists() and is_link_or_reparse(supplied_root):
            raise UnsafeFilesystemPath("Staging root cannot be a link or reparse point")
        self.root = supplied_root.resolve(strict=False)
        self.path = self.root / safe_id

    def create(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        ensure_regular_directory(self.root)
        make_contained_directories(self.root, (self.path.name,))
        make_contained_directories(self.root, (self.path.name, "inputs"))
        make_contained_directories(self.root, (self.path.name, "outputs"))
        return self.path

    def resolve(self, relative_path: str) -> Path:
        normalized = safe_relative_path(relative_path)
        ensure_regular_directory(self.path)
        return contained_path(self.path, tuple(normalized.split("/")))

    def _make_parent(self, relative_path: str) -> Path:
        normalized = safe_relative_path(relative_path)
        parts = tuple(normalized.split("/"))
        make_contained_directories(self.path, parts[:-1])
        return contained_path(self.path, parts)

    def digest(self, relative_path: str) -> str:
        candidate = self.resolve(relative_path)
        digest = hashlib.sha256()
        with open_regular_read(candidate, root=self.path) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def write_manifest(self, manifest: PluginJobManifest | PluginJobManifestV2) -> Path:
        self.create()
        destination = self.path / "job.json"
        temporary = destination.with_suffix(".tmp")
        with open_exclusive_write(temporary) as stream:
            stream.write(manifest.to_json().encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        return destination

    def stage_file(self, source: Path | str, relative_path: str, *, expected_sha256: str) -> Path:
        normalized = safe_relative_path(relative_path)
        if not normalized.startswith("inputs/"):
            raise ValueError("Staged inputs must be placed below inputs/")
        source_path = Path(source)
        destination = self._make_parent(normalized)
        temporary = destination.with_name(destination.name + ".part")
        digest = hashlib.sha256()
        with open_regular_read(source_path) as reader, open_exclusive_write(temporary) as writer:
            while chunk := reader.read(1024 * 1024):
                digest.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if digest.hexdigest() != expected_sha256.lower():
            temporary.unlink(missing_ok=True)
            raise ValueError("Staged input checksum does not match its manifest")
        os.replace(temporary, destination)
        contained_path(self.path, tuple(normalized.split("/")))
        return destination

    def verify_result(self, result: PluginResultManifest | PluginResultPublicationV2) -> None:
        seen: set[Path] = set()
        for output in result.outputs:
            normalized = safe_relative_path(output.relative_path)
            if not normalized.startswith("outputs/"):
                raise ValueError("Plugin outputs must be placed below outputs/")
            candidate = self.resolve(output.relative_path)
            if candidate in seen:
                raise ValueError("Result references the same output path twice")
            seen.add(candidate)
            try:
                digest = self.digest(output.relative_path)
            except (FileNotFoundError, UnsafeFilesystemPath) as exc:
                raise ValueError(f"Result output does not exist safely: {output.relative_path}") from exc
            if digest != output.sha256:
                raise ValueError(f"Result checksum mismatch: {output.relative_path}")

    def remove(self) -> None:
        ensure_regular_directory(self.root)
        contained_path(self.root, (self.path.name,))
        shutil.rmtree(self.path, ignore_errors=False)


__all__ = [
    "AgentJob",
    "AgentJobState",
    "DuplicateCallbackError",
    "DurableJobStore",
    "JobStateError",
    "StagingWorkspace",
    "TERMINAL_STATES",
]
