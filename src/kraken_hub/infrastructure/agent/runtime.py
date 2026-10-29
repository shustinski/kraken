"""Lazy lifecycle manager for the durable loopback Kraken Agent."""

from __future__ import annotations

import json
import importlib.util
import os
import secrets
import subprocess
import sys
import tempfile
import time
from threading import RLock
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.parse import urlsplit

from kraken_core.plugins import PluginInventoryItem, is_plugin_launchable
from kraken_agent.runner import PluginRegistry


_KARAKAL_WORKER_OPERATIONS = frozenset({"layer.confidence.analyze.v1", "karakal.analyze.v1"})


def _agent_command(plugin_id: str, operation: str, command: list[str]) -> list[str]:
    if plugin_id == "karakal" and operation in _KARAKAL_WORKER_OPERATIONS:
        return [sys.executable, "-m", "karakal.worker"]
    return command


def _ensure_analysis_operation(
    specs: list[dict[str, Any]],
    capabilities: set[str],
    protocols: dict[str, str],
    plugins: tuple[PluginInventoryItem, ...],
) -> None:
    if "karakal.analyze.v1" in capabilities:
        return
    karakal = next(
        (item.metadata for item in plugins if item.metadata.id == "karakal" and item.metadata.enabled),
        None,
    )
    if karakal is None:
        return
    capabilities.add("karakal.analyze.v1")
    protocols["karakal.analyze.v1"] = karakal.protocol_version
    specs.append(
        {
            "operation": "karakal.analyze.v1",
            "command": [sys.executable, "-m", "karakal.worker"],
            "working_directory": karakal.source_dir or None,
            "interactive": False,
        }
    )


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
        *, builtin_stitching: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.plugins = plugins
        self.builtin_stitching = builtin_stitching
        self.process: subprocess.Popen[bytes] | None = None
        self.base_url = ""
        self.token = ""
        self.protocol_by_capability: dict[str, str] = {}
        self._log_handle = None
        self._lifecycle_lock = RLock()

    def _discard_process(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self.process = None
        self.base_url = ""
        self.token = ""
        log_handle = self._log_handle
        self._log_handle = None
        if log_handle is not None:
            log_handle.close()

    def _registry(self) -> tuple[Path, frozenset[str]]:
        from kraken_hub.infrastructure.agent.launch import build_launch_command

        specs: list[dict[str, Any]] = []
        capabilities: set[str] = set()
        protocols: dict[str, str] = {}
        for inventory in self.plugins:
            plugin = inventory.metadata
            if not plugin.enabled or not is_plugin_launchable(inventory):
                continue
            if not plugin.capabilities:
                continue
            plugin_root = Path(plugin.source_dir).resolve() if plugin.source_dir else None
            executable_path = plugin.executable_for().path
            if not executable_path and plugin_root is not None and (plugin_root / "__main__.py").is_file():
                command = [sys.executable, str(plugin_root / "__main__.py")]
            else:
                command = build_launch_command(
                    plugin,
                    root=None if plugin_root is None else plugin_root.parent.parent,
                )
            for capability in plugin.capabilities:
                if capability.operation in capabilities:
                    raise ValueError(f"Несколько плагинов объявляют capability {capability.operation}")
                capabilities.add(capability.operation)
                protocols[capability.operation] = plugin.protocol_version
                specs.append(
                    {
                        "operation": capability.operation,
                        "command": _agent_command(plugin.id, capability.operation, command),
                        "working_directory": (
                            str(Path(executable_path).expanduser().resolve(strict=False).parent)
                            if executable_path
                            else None
                            if plugin_root is None
                            else str(plugin_root)
                        ),
                        "interactive": (
                            capability.default_mode == "interactive"
                            if capability.default_mode is not None
                            else "interactive" in capability.modes and "headless" not in capability.modes
                        ),
                        "modes": list(capability.modes),
                    }
                )
        _ensure_analysis_operation(specs, capabilities, protocols, self.plugins)
        if self.builtin_stitching and all(importlib.util.find_spec(name) is not None for name in ("numpy", "PIL")):
            if "layer.stitch.v1" in capabilities:
                raise ValueError("A plugin conflicts with built-in layer stitching")
            capabilities.add("layer.stitch.v1")
            protocols["layer.stitch.v1"] = "2.0"
            specs.append({"operation": "layer.stitch.v1", "command": [sys.executable,
                *(["--kraken-stitch-worker"] if getattr(sys, "frozen", False) else
                  ["-m", "kraken_manager.infrastructure.stitch_worker"])], "interactive": False, "modes": ["headless"]})
        target = self.data_dir / "plugin-registry.json"
        _atomic_json(target, {"plugins": specs})
        self.protocol_by_capability = protocols
        return target, frozenset(capabilities)

    def interactive_registry(self, operation: str) -> PluginRegistry:
        from dataclasses import replace

        path, _capabilities = self._registry()
        spec = PluginRegistry.from_json(path).get(operation)
        supports_interactive = any(
            capability.operation == operation and "interactive" in capability.modes
            for item in self.plugins
            for capability in item.metadata.capabilities
        )
        if spec is None or not supports_interactive:
            raise ValueError(f"Interactive plugin is not installed for {operation}")
        return PluginRegistry({operation: replace(spec, interactive=True)})

    def execution_registry(self, operation: str) -> PluginRegistry:
        path, _capabilities = self._registry()
        spec = PluginRegistry.from_json(path).get(operation)
        if spec is None:
            raise ValueError(f"Plugin is not installed for {operation}")
        return PluginRegistry({operation: spec})

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        *,
        timeout: float = 3.0,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
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
        with self._lifecycle_lock:
            return self._ensure_started()

    def _ensure_started(self) -> frozenset[str]:
        if self.base_url and self.token:
            try:
                health = self._request("GET", "/api/v1/health")
            except OSError, ValueError, TypeError:
                self._discard_process()
            else:
                if health.get("status") == "ok":
                    return self._registry()[1]
        # The Agent owns its jobs and can outlive any individual Hub session.
        saved_connection = self.data_dir / "runtime-connection.json"
        try:
            saved = json.loads(saved_connection.read_text(encoding="utf-8"))
            address = urlsplit(saved["url"])
            if address.scheme != "http" or address.hostname not in {"127.0.0.1", "::1"}:
                raise ValueError("Invalid local Agent address")
            self.base_url, self.token = saved["url"], saved["token"]
            if self._request("GET", "/api/v1/health").get("status") == "ok":
                return self._registry()[1]
        except (OSError, ValueError, TypeError, KeyError):
            pass
        self.base_url = self.token = ""
        registry, capabilities = self._registry()
        connection = self.data_dir / f"connection-{secrets.token_hex(8)}.json"
        command = [
            sys.executable,
            *(["--kraken-agent"] if getattr(sys, "frozen", False) else ["-m", "kraken_agent"]),
            "--data-dir",
            str(self.data_dir),
            "--plugins-config",
            str(registry),
            "--connection-file",
            str(connection),
            "--independent",
        ]
        environment = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[1])
        python_path = environment.get("PYTHONPATH", "")
        python_path_entries = tuple(entry for entry in python_path.split(os.pathsep) if entry)
        if source_root not in python_path_entries:
            environment["PYTHONPATH"] = os.pathsep.join((source_root, *python_path_entries))
        from kraken_core.process_lifetime import detached_creationflags
        creation_flags = detached_creationflags(int(getattr(subprocess, "CREATE_NO_WINDOW", 0)))
        log_path = self.data_dir / "agent.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log_path.open("ab")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
            start_new_session=os.name != "nt",
            env=environment,
        )
        deadline = time.monotonic() + 12.0
        payload: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                break
            if connection.is_file():
                try:
                    loaded = json.loads(connection.read_text(encoding="utf-8"))
                except OSError, json.JSONDecodeError:
                    time.sleep(0.05)
                    continue
                if isinstance(loaded, dict):
                    payload = loaded
                    break
            time.sleep(0.05)
        connection.unlink(missing_ok=True)
        if payload is None:
            self._discard_process()
            raise RuntimeError("Kraken Agent did not publish a connection file")
        self.base_url = str(payload.get("url", ""))
        self.token = str(payload.get("token", ""))
        if not self.base_url or not self.token:
            self._discard_process()
            raise RuntimeError("Kraken Agent published an incomplete connection file")
        try:
            health = self._request("GET", "/api/v1/health")
        except (OSError, ValueError, TypeError) as exc:
            self._discard_process()
            raise RuntimeError("Kraken Agent health check failed") from exc
        if health.get("status") != "ok":
            self._discard_process()
            raise RuntimeError("Kraken Agent health check failed")
        _atomic_json(saved_connection, payload)
        saved_connection.chmod(0o600)
        return capabilities

    def detach(self) -> None:
        """Release this client's handles while leaving durable jobs running."""
        with self._lifecycle_lock:
            self.process = None
            self.base_url = self.token = ""
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None

    def shutdown(self) -> None:
        with self._lifecycle_lock:
            self._shutdown()

    def _shutdown(self) -> None:
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
                    raise RuntimeError("Kraken Agent did not stop gracefully within 10 seconds") from exc
            self.base_url = ""
            self.token = ""
            self.process = None
            (self.data_dir / "runtime-connection.json").unlink(missing_ok=True)
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None


__all__ = ["LocalAgentRuntime"]
