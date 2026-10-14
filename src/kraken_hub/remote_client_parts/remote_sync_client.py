"""remote_client: remote sync client."""
from __future__ import annotations
import json
import threading
from collections.abc import Callable, Mapping
from typing import Any
from kraken_hub.workspace_service import (
    ProjectEventWake,
)

from kraken_hub.remote_client_parts.module_state import LOGGER

class RemoteSyncClient:
    """Background WebSocket subscriber that emits ProjectEventWake callbacks."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str,
        on_event: Callable[[ProjectEventWake], None],
        on_catalog: Callable[[], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self._ws_url = self._to_ws_url(base_url)
        self._token = token
        self._on_event = on_event
        self._on_catalog = on_catalog
        self._on_status = on_status
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._send_lock = threading.Lock()
        self._ws: Any = None
        self._subscribed: set[str] = set()
        self._catalog = False

    @staticmethod
    def _to_ws_url(base_url: str) -> str:
        if base_url.startswith("https://"):
            return (
                "wss://" + base_url.removeprefix("https://").rstrip("/") + "/api/v1/ws"
            )
        if base_url.startswith("http://"):
            return "ws://" + base_url.removeprefix("http://").rstrip("/") + "/api/v1/ws"
        return base_url.rstrip("/") + "/api/v1/ws"

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._status("reconnecting")
        self._thread = threading.Thread(
            target=self._run, name="kraken-remote-ws", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except (OSError, RuntimeError):
                LOGGER.exception("Failed to close Kraken remote WebSocket")
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._status("offline")

    def _status(self, value: str) -> None:
        if self._on_status is not None:
            self._on_status(value)

    def subscribe_project(self, project_id: str) -> None:
        self._subscribed.add(str(project_id))
        self._send({"type": "subscribe", "project_id": str(project_id)})

    def unsubscribe_project(self, project_id: str) -> None:
        self._subscribed.discard(str(project_id))
        self._send({"type": "unsubscribe", "project_id": str(project_id)})

    def subscribe_catalog(self) -> None:
        self._catalog = True
        self._send({"type": "subscribe", "catalog": True})

    def _send(self, message: Mapping[str, object]) -> None:
        ws = self._ws
        if ws is None:
            return
        with self._send_lock:
            try:
                ws.send(json.dumps(dict(message)))
            except (OSError, RuntimeError):
                LOGGER.exception("Failed to send Kraken remote subscription")

    def _run(self) -> None:
        try:
            from websocket import WebSocketApp  # type: ignore
        except ImportError:
            self._status("offline")
            return

        def on_message(_ws: Any, message: str) -> None:
            try:
                payload = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                return
            kind = str(payload.get("type", ""))
            if kind == "project_event":
                wake = ProjectEventWake(
                    project_id=str(payload.get("project_id", "")),
                    event_type=str(payload.get("event_type", "")),
                    event_id=str(payload.get("event_id", "")),
                    position=(
                        None
                        if payload.get("position") is None
                        else int(payload["position"])
                    ),
                    revision=(
                        None
                        if payload.get("revision") is None
                        else int(payload["revision"])
                    ),
                    stream_id=str(payload.get("stream_id", "")),
                    entity_kind=str(payload.get("entity_kind", "")),
                    entity_id=str(payload.get("entity_id", "")),
                )
                if wake.project_id:
                    self._on_event(wake)
            elif kind == "catalog_changed" and self._on_catalog is not None:
                self._on_catalog()

        def on_open(ws: Any) -> None:
            self._ws = ws
            if self._catalog:
                self._send({"type": "subscribe", "catalog": True})
            for project_id in list(self._subscribed):
                self._send({"type": "subscribe", "project_id": project_id})
            self._status("synchronized")
            # A reconnect may have missed wake messages. REST snapshots are
            # authoritative, so force a catalog/project refresh on every open.
            if self._on_catalog is not None:
                self._on_catalog()

        def on_close(_ws: Any, *_args: object) -> None:
            self._ws = None
            if not self._stop.is_set():
                self._status("reconnecting")

        while not self._stop.is_set():
            app = WebSocketApp(
                self._ws_url,
                header=[f"Authorization: Bearer {self._token}"],
                on_message=on_message,
                on_open=on_open,
                on_close=on_close,
            )
            try:
                app.run_forever(ping_interval=20, ping_timeout=10)
            except (OSError, RuntimeError):
                LOGGER.exception("Kraken remote WebSocket loop failed")
            if self._stop.wait(2.0):
                break
