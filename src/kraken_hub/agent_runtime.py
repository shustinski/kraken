"""Lazy lifecycle manager for the durable loopback Kraken Agent."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from kraken_core.plugins import PluginInventoryItem


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class LocalAgentRuntime:
    def __init__(
        self,
        data_dir: Path | str,
        plugins: tuple[PluginInventoryItem, ...] = (),
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.plugins = plugins
        self.process: subprocess.Popen[bytes] | None = None
        self.base_url = ""
        self.token = ""
        self.protocol_by_capability: dict[str, str] = {}

    def _registry(self) -> tuple[Path, frozenset[str]]:
        from .app import build_launch_command

        specs: list[dict[str, Any]] = []
        capabilities: set[str] = set()
        protocols: dict[str, str] = {}
        for inventory in self.plugins:
            plugin = inventory.metadata
            if not plugin.enabled or not (inventory.installed or plugin.source_dir):
                continue
            if not plugin.capabilities:
                continue
            plugin_root = Path(plugin.source_dir).resolve() if plugin.source_dir else None
            command = build_launch_command(
                plugin,
                root=None if plugin_root is None else plugin_root.parent.parent,
            )
            for capability in plugin.capabilities:
                if capability.operation in capabilities:
                    raise ValueError(
                        f"Несколько плагинов объявляют capability {capability.operation}"
                    )
                capabilities.add(capability.operation)
                protocols[capability.operation] = plugin.protocol_version
                specs.append(
                    {
                        "operation": capability.operation,
                        "command": command,
                        "working_directory": (
                            None if plugin_root is None else str(plugin_root)
                        ),
                        "interactive": (
                            "interactive" in capability.modes
                            and "headless" not in capability.modes
                        ),
                    }
                )
        target = self.data_dir / "plugin-registry.json"
        _atomic_json(target, {"plugins": specs})
        self.protocol_by_capability = protocols
        return target, frozenset(capabilities)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        *,
        timeout: float = 3.0,
    ) -> dict[str, Any]:
        body = (
            None
            if payload is None
            else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        )
        request = Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise TypeError("Kraken Agent returned an invalid response")
        return result

    def ensure_started(self) -> frozenset[str]:
        if self.base_url and self.token:
            try:
                health = self._request("GET", "/api/v1/health")
            except (OSError, ValueError, TypeError):
                self.base_url = ""
                self.token = ""
            else:
                if health.get("status") == "ok":
                    return self._registry()[1]
        registry, capabilities = self._registry()
        connection = self.data_dir / f"connection-{secrets.token_hex(8)}.json"
        token = secrets.token_urlsafe(32)
        command = [
            sys.executable,
            "-m",
            "kraken_agent",
            "--data-dir",
            str(self.data_dir),
            "--token",
            token,
            "--plugins-config",
            str(registry),
            "--connection-file",
            str(connection),
        ]
        creation_flags = (
            int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if os.name == "nt"
            else 0
        )
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        deadline = time.monotonic() + 12.0
        payload: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                break
            if connection.is_file():
                try:
                    loaded = json.loads(connection.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    time.sleep(0.05)
                    continue
                if isinstance(loaded, dict):
                    payload = loaded
                    break
            time.sleep(0.05)
        connection.unlink(missing_ok=True)
        if payload is None:
            raise RuntimeError("Kraken Agent did not publish a connection file")
        self.base_url = str(payload.get("url", ""))
        self.token = str(payload.get("token", ""))
        if self.token != token:
            raise RuntimeError("Kraken Agent connection token does not match")
        health = self._request("GET", "/api/v1/health")
        if health.get("status") != "ok":
            raise RuntimeError("Kraken Agent health check failed")
        return capabilities

    def shutdown(self) -> None:
        if not self.base_url or not self.token:
            return
        try:
            self._request("POST", "/api/v1/shutdown", {})
        finally:
            process = self.process
            if process is not None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        "Kraken Agent did not stop gracefully within 10 seconds"
                    ) from exc
            self.base_url = ""
            self.token = ""


__all__ = ["LocalAgentRuntime"]
