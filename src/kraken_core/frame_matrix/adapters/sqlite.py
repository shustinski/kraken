"""Sharded SQLite implementation of the storage-neutral thumbnail port."""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
from collections.abc import Iterable, Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

from ..storage import (
    CorruptEntry,
    InvalidationSelector,
    StoreCapability,
    StoreFull,
    StoreNamespace,
    StorePolicy,
    StoreStats,
    StoreUnavailable,
    ThumbnailKey,
    ThumbnailRecord,
)


_SCHEMA_VERSION = 1
_SHARD_COUNT = 64


def _key_payload(key: ThumbnailKey) -> dict[str, Any]:
    return {
        "source_key": key.source_key,
        "source_revision": key.source_revision,
        "lod": key.lod,
        "width": key.width,
        "height": key.height,
        "codec": key.codec,
        "renderer_fingerprint": key.renderer_fingerprint,
        "device_pixel_ratio": key.device_pixel_ratio,
        "fit_mode": key.fit_mode,
        "color_mode": key.color_mode,
    }


def _key_from_payload(payload: Mapping[str, Any]) -> ThumbnailKey:
    return ThumbnailKey(
        source_key=str(payload["source_key"]),
        source_revision=str(payload["source_revision"]),
        lod=int(payload["lod"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        codec=str(payload.get("codec", "png")),
        renderer_fingerprint=str(payload.get("renderer_fingerprint", "")),
        device_pixel_ratio=float(payload.get("device_pixel_ratio", 1.0)),
        fit_mode=str(payload.get("fit_mode", "cover")),
        color_mode=str(payload.get("color_mode", "source")),
    )


class ShardedSQLiteThumbnailStore:
    capabilities = (
        StoreCapability.BATCH_READ
        | StoreCapability.BATCH_WRITE
        | StoreCapability.ATOMIC_REPLACE
        | StoreCapability.MULTIPROCESS
        | StoreCapability.ACCESS_METADATA
        | StoreCapability.NAMESPACE_DELETE
        | StoreCapability.COMPACTION
        | StoreCapability.INTEGRITY_CHECK
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._namespace: StoreNamespace | None = None
        self._namespace_root: Path | None = None
        self._policy = StorePolicy()
        self._write_queue: queue.Queue[object] = queue.Queue()
        self._writer: threading.Thread | None = None
        self._stop_marker = object()
        self._maintenance_lock = threading.RLock()
        self._access_lock = threading.Lock()
        self._pending_accesses: dict[int, set[bytes]] = {}
        self._hits = self._misses = self._writes = self._errors = 0

    def open(self, namespace: StoreNamespace, policy: StorePolicy | None = None) -> None:
        self.close()
        self._namespace = namespace
        self._policy = policy or StorePolicy()
        self.root.mkdir(parents=True, exist_ok=True)
        namespace_root = (self.root / namespace.digest()).resolve()
        if not namespace_root.is_relative_to(self.root):
            raise StoreUnavailable("thumbnail namespace escapes store root")
        namespace_root.mkdir(parents=True, exist_ok=True)
        self._namespace_root = namespace_root
        self._ensure_catalog()
        self._write_queue = queue.Queue()
        self._pending_accesses = {}
        self._writer = threading.Thread(
            target=self._writer_loop,
            name=f"kraken-thumbnail-writer-{namespace.digest()[:8]}",
            daemon=True,
        )
        self._writer.start()

    def _require_open(self) -> Path:
        if self._namespace_root is None:
            raise StoreUnavailable("thumbnail store is not open")
        return self._namespace_root

    def _catalog_path(self) -> Path:
        return self._require_open() / "catalog.sqlite3"

    def _shard_path(self, shard: int) -> Path:
        return self._require_open() / f"thumb-{int(shard):02x}.sqlite3"

    @staticmethod
    def _shard_for(key: ThumbnailKey) -> int:
        return key.digest()[0] >> 2

    def _connect(self, path: Path, *, initialize: bool = False) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 30000")
            if initialize:
                connection.execute("PRAGMA page_size = 16384")
                connection.execute("PRAGMA auto_vacuum = INCREMENTAL")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(f"PRAGMA synchronous = {'FULL' if self._policy.durable else 'NORMAL'}")
            return connection
        except sqlite3.Error as exc:
            self._errors += 1
            raise StoreUnavailable(str(exc)) from exc

    def _ensure_catalog(self) -> None:
        path = self._catalog_path()
        initialize = not path.exists()
        with closing(self._connect(path, initialize=initialize)) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS store_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                ) WITHOUT ROWID;
                """
            )
            values = {
                "schema_version": str(_SCHEMA_VERSION),
                "namespace": self._namespace.canonical() if self._namespace is not None else "",
                "shard_count": str(_SHARD_COUNT),
            }
            connection.executemany(
                "INSERT OR REPLACE INTO store_metadata(key, value) VALUES (?, ?)",
                values.items(),
            )

    def _connect_shard(self, shard: int) -> sqlite3.Connection:
        path = self._shard_path(shard)
        initialize = not path.exists()
        connection = self._connect(path, initialize=initialize)
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS thumbnails (
                    key_hash BLOB PRIMARY KEY,
                    key_json TEXT NOT NULL,
                    source_key TEXT NOT NULL,
                    source_revision TEXT NOT NULL,
                    renderer_fingerprint TEXT NOT NULL,
                    codec TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    checksum TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    last_access REAL NOT NULL,
                    access_count INTEGER NOT NULL
                ) WITHOUT ROWID;
                CREATE INDEX IF NOT EXISTS ix_thumbnails_source
                    ON thumbnails(source_key, source_revision);
                CREATE INDEX IF NOT EXISTS ix_thumbnails_renderer
                    ON thumbnails(renderer_fingerprint);
                CREATE INDEX IF NOT EXISTS ix_thumbnails_access
                    ON thumbnails(last_access);
                """
            )
        except BaseException:
            connection.close()
            raise
        return connection

    def get(self, key: ThumbnailKey) -> ThumbnailRecord | None:
        shard = self._shard_for(key)
        path = self._shard_path(shard)
        if not path.exists():
            self._misses += 1
            return None
        try:
            with closing(self._connect_shard(shard)) as connection:
                row = connection.execute(
                    "SELECT payload, codec, checksum, metadata_json FROM thumbnails WHERE key_hash = ?",
                    (key.digest(),),
                ).fetchone()
                if row is None:
                    self._misses += 1
                    return None
        except sqlite3.DatabaseError as exc:
            self._errors += 1
            self._quarantine_shard(shard)
            raise CorruptEntry(f"thumbnail shard is corrupt: {exc}") from exc
        try:
            metadata = json.loads(str(row["metadata_json"]))
            record = ThumbnailRecord(
                key=key,
                payload=bytes(row["payload"]),
                codec=str(row["codec"]),
                checksum=str(row["checksum"]),
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        except (TypeError, ValueError, json.JSONDecodeError, CorruptEntry) as exc:
            self._errors += 1
            self.delete(key)
            raise CorruptEntry(str(exc)) from exc
        self._hits += 1
        self._record_access(shard, key.digest())
        return record

    def _record_access(self, shard: int, key_hash: bytes) -> None:
        should_flush = False
        with self._access_lock:
            self._pending_accesses.setdefault(int(shard), set()).add(bytes(key_hash))
            should_flush = sum(len(values) for values in self._pending_accesses.values()) >= 256
        if should_flush:
            self._flush_accesses()

    def _flush_accesses(self) -> None:
        if self._namespace_root is None:
            return
        with self._access_lock:
            pending = self._pending_accesses
            self._pending_accesses = {}
        if not pending:
            return
        now = time.time()
        try:
            with self._maintenance_lock:
                for shard, hashes in pending.items():
                    if not self._shard_path(shard).exists():
                        continue
                    with closing(self._connect_shard(shard)) as connection:
                        connection.execute("BEGIN IMMEDIATE")
                        connection.executemany(
                            """
                            UPDATE thumbnails
                            SET last_access = ?, access_count = access_count + 1
                            WHERE key_hash = ?
                            """,
                            ((now, key_hash) for key_hash in hashes),
                        )
                        connection.execute("COMMIT")
        except (sqlite3.Error, StoreUnavailable):
            with self._access_lock:
                for shard, hashes in pending.items():
                    self._pending_accesses.setdefault(shard, set()).update(hashes)

    def get_many(self, keys: Iterable[ThumbnailKey]) -> Mapping[ThumbnailKey, ThumbnailRecord]:
        return {key: record for key in keys if (record := self.get(key)) is not None}

    def put(self, record: ThumbnailRecord) -> None:
        self.put_many((record,))

    def put_many(self, records: Iterable[ThumbnailRecord]) -> None:
        prepared = tuple(records)
        if not prepared:
            return
        for record in prepared:
            if len(record.payload) > self._policy.max_entry_bytes:
                self._errors += 1
                raise StoreFull("thumbnail exceeds max_entry_bytes")
        if self._writer is None or not self._writer.is_alive():
            raise StoreUnavailable("thumbnail writer is not running")
        completed = threading.Event()
        errors: list[BaseException] = []
        self._write_queue.put((prepared, completed, errors))
        completed.wait()
        if errors:
            error = errors[0]
            if isinstance(error, (StoreFull, StoreUnavailable)):
                raise error
            raise StoreUnavailable(str(error)) from error

    def _writer_loop(self) -> None:
        while True:
            item = self._write_queue.get()
            if item is self._stop_marker:
                return
            submissions = [item]
            record_count = len(item[0])  # type: ignore[index]
            deadline = time.monotonic() + max(0, self._policy.batch_delay_ms) / 1000.0
            while record_count < self._policy.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    candidate = self._write_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if candidate is self._stop_marker:
                    self._write_queue.put(candidate)
                    break
                submissions.append(candidate)
                record_count += len(candidate[0])  # type: ignore[index]
            records = tuple(record for submission in submissions for record in submission[0])  # type: ignore[index]
            errors: list[BaseException] = []
            try:
                self._write_records(records)
            except BaseException as exc:
                errors.append(exc)
                self._errors += 1
            for _records, completed, submission_errors in submissions:  # type: ignore[misc]
                submission_errors.extend(errors)
                completed.set()

    def _write_records(self, records: tuple[ThumbnailRecord, ...]) -> None:
        grouped: dict[int, list[ThumbnailRecord]] = {}
        for record in records:
            grouped.setdefault(self._shard_for(record.key), []).append(record)
        now = time.time()
        with self._maintenance_lock:
            for shard, shard_records in grouped.items():
                try:
                    with closing(self._connect_shard(shard)) as connection:
                        connection.execute("BEGIN IMMEDIATE")
                        connection.executemany(
                            """
                            INSERT OR REPLACE INTO thumbnails(
                                key_hash, key_json, source_key, source_revision,
                                renderer_fingerprint, codec, payload, checksum,
                                metadata_json, byte_size, created_at, last_access, access_count
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            [
                                (
                                    record.key.digest(),
                                    json.dumps(_key_payload(record.key), separators=(",", ":"), sort_keys=True),
                                    record.key.source_key,
                                    record.key.source_revision,
                                    record.key.renderer_fingerprint,
                                    record.codec,
                                    record.payload,
                                    record.checksum,
                                    json.dumps(dict(record.metadata), separators=(",", ":"), sort_keys=True),
                                    len(record.payload),
                                    now,
                                    now,
                                    0,
                                )
                                for record in shard_records
                            ],
                        )
                        connection.execute("COMMIT")
                except sqlite3.OperationalError as exc:
                    if "full" in str(exc).lower():
                        raise StoreFull(str(exc)) from exc
                    raise StoreUnavailable(str(exc)) from exc
                self._writes += len(shard_records)

    def delete(self, key: ThumbnailKey) -> bool:
        shard = self._shard_for(key)
        if not self._shard_path(shard).exists():
            return False
        with self._maintenance_lock, closing(self._connect_shard(shard)) as connection:
            cursor = connection.execute("DELETE FROM thumbnails WHERE key_hash = ?", (key.digest(),))
            return cursor.rowcount > 0

    def invalidate(self, selector: InvalidationSelector) -> int:
        clauses: list[str] = []
        parameters: list[str] = []
        for column, value in (
            ("source_key", selector.source_key),
            ("source_revision", selector.source_revision),
            ("renderer_fingerprint", selector.renderer_fingerprint),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        where = " AND ".join(clauses) if clauses else "1 = 1"
        deleted = 0
        with self._maintenance_lock:
            for shard in self._existing_shards():
                with closing(self._connect_shard(shard)) as connection:
                    cursor = connection.execute(f"DELETE FROM thumbnails WHERE {where}", parameters)
                    deleted += max(0, cursor.rowcount)
        return deleted

    def _existing_shards(self) -> list[int]:
        root = self._require_open()
        result: list[int] = []
        for path in root.glob("thumb-??.sqlite3"):
            try:
                shard = int(path.stem.split("-")[1], 16)
            except (IndexError, ValueError):
                continue
            if 0 <= shard < _SHARD_COUNT:
                result.append(shard)
        return sorted(set(result))

    def clear_namespace(self) -> None:
        with self._maintenance_lock:
            for shard in self._existing_shards():
                path = self._shard_path(shard)
                for suffix in ("-wal", "-shm", ""):
                    try:
                        Path(f"{path}{suffix}").unlink()
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        raise StoreUnavailable(str(exc)) from exc

    def stats(self) -> StoreStats:
        entries = byte_count = 0
        for shard in self._existing_shards():
            try:
                with closing(self._connect_shard(shard)) as connection:
                    row = connection.execute(
                        "SELECT COUNT(*) AS entries, COALESCE(SUM(byte_size), 0) AS bytes FROM thumbnails"
                    ).fetchone()
                    entries += int(row["entries"])
                    byte_count += int(row["bytes"])
            except sqlite3.Error:
                self._errors += 1
        return StoreStats(
            entries=entries,
            bytes=byte_count,
            hits=self._hits,
            misses=self._misses,
            writes=self._writes,
            errors=self._errors,
        )

    def compact(self) -> None:
        self._flush_accesses()
        with self._maintenance_lock:
            for shard in self._existing_shards():
                with closing(self._connect_shard(shard)) as connection:
                    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    connection.execute("PRAGMA incremental_vacuum")

    def integrity_check(self) -> Mapping[int, str]:
        results: dict[int, str] = {}
        for shard in self._existing_shards():
            try:
                with closing(self._connect_shard(shard)) as connection:
                    row = connection.execute("PRAGMA quick_check").fetchone()
                    results[shard] = str(row[0]) if row else "unknown"
            except sqlite3.Error as exc:
                results[shard] = str(exc)
        return results

    def _quarantine_shard(self, shard: int) -> None:
        path = self._shard_path(shard)
        stamp = int(time.time())
        with self._maintenance_lock:
            for suffix in ("-wal", "-shm", ""):
                candidate = Path(f"{path}{suffix}")
                if not candidate.exists():
                    continue
                quarantine = candidate.with_name(f"{candidate.name}.corrupt-{stamp}")
                try:
                    os.replace(candidate, quarantine)
                except OSError:
                    pass

    def close(self) -> None:
        writer = self._writer
        if writer is not None and writer.is_alive():
            self._write_queue.put(self._stop_marker)
            writer.join(timeout=5.0)
        self._flush_accesses()
        self._writer = None
        self._namespace = None
        self._namespace_root = None


__all__ = ["ShardedSQLiteThumbnailStore"]
