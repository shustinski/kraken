from __future__ import annotations

import json
import socket
import socketserver
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionRequest:
    action: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionResponse:
    ok: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"ok": self.ok, "message": self.message, "data": self.data}, ensure_ascii=False)


class ActionRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[dict[str, Any]], ActionResponse | Mapping[str, Any] | None]] = {}
        self.register("get_capabilities", lambda _payload: ActionResponse(True, data={"actions": self.actions()}))

    def register(self, name: str, handler: Callable[[dict[str, Any]], ActionResponse | Mapping[str, Any] | None]) -> None:
        self._handlers[str(name)] = handler

    def actions(self) -> list[str]:
        return sorted(self._handlers)

    def dispatch(self, request: ActionRequest) -> ActionResponse:
        handler = self._handlers.get(request.action)
        if handler is None:
            return ActionResponse(False, f"Unsupported action: {request.action}")
        try:
            result = handler(request.payload)
        except Exception as exc:
            return ActionResponse(False, str(exc))
        if isinstance(result, ActionResponse):
            return result
        if isinstance(result, Mapping):
            return ActionResponse(bool(result.get("ok", True)), str(result.get("message", "")), dict(result.get("data", {})))
        return ActionResponse(True)


def send_action(host: str, port: int, request: ActionRequest, *, timeout: float = 3.0) -> ActionResponse:
    with socket.create_connection((host, port), timeout=timeout) as client:
        client.sendall((json.dumps({"action": request.action, "payload": request.payload}) + "\n").encode("utf-8"))
        response = client.makefile("r", encoding="utf-8").readline()
    payload = json.loads(response)
    return ActionResponse(bool(payload.get("ok")), str(payload.get("message", "")), dict(payload.get("data", {})))


def serve_actions(registry: ActionRegistry, *, host: str = "127.0.0.1", port: int = 0) -> tuple[socketserver.ThreadingTCPServer, threading.Thread]:
    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            raw = self.rfile.readline().decode("utf-8")
            try:
                payload = json.loads(raw)
                response = registry.dispatch(ActionRequest(str(payload.get("action", "")), dict(payload.get("payload", {}))))
            except Exception as exc:
                response = ActionResponse(False, str(exc))
            self.wfile.write((response.to_json() + "\n").encode("utf-8"))

    server = socketserver.ThreadingTCPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
