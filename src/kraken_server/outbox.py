"""Transactional outbox fan-out to authenticated WebSocket subscribers."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class ConnectionHub:
    """In-process registry of WebSocket subscriptions."""

    _project_subs: dict[Any, set[str]] = field(default_factory=dict)
    _catalog_subs: set[Any] = field(default_factory=set)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def register(self, websocket: Any) -> None:
        with self._lock:
            self._project_subs.setdefault(websocket, set())

    def unregister(self, websocket: Any) -> None:
        with self._lock:
            self._project_subs.pop(websocket, None)
            self._catalog_subs.discard(websocket)

    def subscribe(self, websocket: Any, *, project_id: str | None = None, catalog: bool = False) -> None:
        with self._lock:
            self._project_subs.setdefault(websocket, set())
            if project_id:
                self._project_subs[websocket].add(str(project_id))
            if catalog:
                self._catalog_subs.add(websocket)

    def unsubscribe(self, websocket: Any, *, project_id: str | None = None, catalog: bool = False) -> None:
        with self._lock:
            if project_id and websocket in self._project_subs:
                self._project_subs[websocket].discard(str(project_id))
            if catalog:
                self._catalog_subs.discard(websocket)

    def publish(self, envelope: Mapping[str, object]) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self._fanout(dict(envelope)), loop)

    async def _fanout(self, envelope: dict[str, object]) -> None:
        project_id = str(envelope.get("project_id", ""))
        with self._lock:
            targets = [
                websocket
                for websocket, projects in self._project_subs.items()
                if project_id and project_id in projects
            ]
            if envelope.get("type") in {"project_event", "catalog_changed"}:
                targets.extend(list(self._catalog_subs))
            # Deduplicate while preserving order.
            seen: set[int] = set()
            unique = []
            for websocket in targets:
                key = id(websocket)
                if key in seen:
                    continue
                seen.add(key)
                unique.append(websocket)
        for websocket in unique:
            try:
                await websocket.send_json(envelope)
            except Exception:  # noqa: BLE001 - transports expose backend-specific disconnect errors
                self.unregister(websocket)


class OutboxPublisher:
    """Poll unpublished outbox rows and push wake envelopes to ConnectionHub."""

    def __init__(
        self,
        engine: Any,
        hub: ConnectionHub,
        *,
        interval_seconds: float = 0.5,
        batch_size: int = 100,
    ) -> None:
        self.engine = engine
        self.hub = hub
        self.interval_seconds = interval_seconds
        self.batch_size = batch_size
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="kraken-outbox-publisher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def publish_once(self) -> int:
        """Claim and publish one batch. Returns number of published rows."""
        from sqlalchemy import text

        published = 0
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT o.outbox_id, o.event_id, o.project_id, o.event_type, o.payload,
                           e.position, e.revision, e.stream_id
                    FROM transactional_outbox AS o
                    LEFT JOIN domain_events AS e ON e.event_id = o.event_id
                    WHERE o.published_at IS NULL
                    ORDER BY o.outbox_id
                    LIMIT :limit
                    FOR UPDATE OF o SKIP LOCKED
                    """
                ),
                {"limit": self.batch_size},
            ).mappings().all()
            now = datetime.now(UTC)
            for row in rows:
                stream_id = "" if row["stream_id"] is None else str(row["stream_id"])
                entity_kind, separator, entity_id = stream_id.partition(":")
                envelope = {
                    "type": "project_event",
                    "project_id": str(row["project_id"]),
                    "event_type": str(row["event_type"]),
                    "event_id": str(row["event_id"]),
                    "position": None if row["position"] is None else int(row["position"]),
                    "revision": None if row["revision"] is None else int(row["revision"]),
                    "stream_id": stream_id,
                    "entity_kind": entity_kind if separator else "",
                    "entity_id": entity_id if separator else "",
                }
                self.hub.publish(envelope)
                if str(row["event_type"]).startswith("Project"):
                    self.hub.publish(
                        {
                            "type": "catalog_changed",
                            "project_id": str(row["project_id"]),
                            "event_type": str(row["event_type"]),
                            "event_id": str(row["event_id"]),
                        }
                    )
                connection.execute(
                    text(
                        """
                        UPDATE transactional_outbox
                        SET published_at = :published_at
                        WHERE outbox_id = :outbox_id
                        """
                    ),
                    {"published_at": now, "outbox_id": int(row["outbox_id"])},
                )
                published += 1
        return published

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.publish_once()
            except Exception:
                LOGGER.exception("Failed to publish Kraken transactional outbox batch")
            if self._stop.wait(self.interval_seconds):
                break


__all__ = ["ConnectionHub", "OutboxPublisher"]
