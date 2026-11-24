"""External process ownership, observations, and reproducible action traces."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .server import ManagedServer


class InspectorSession:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.artifacts = self.workspace / ".tmp-tests" / "inspector"
        self.process: subprocess.Popen | None = None
        self.socket_name = ""
        self.token = ""
        self.run_dir: Path | None = None
        self._log_stream = None
        self.actions: list[dict] = []
        self.defects: list[dict] = []
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._samples: deque[dict] = deque(maxlen=2000)
        self._max_rss_growth_bytes: int | None = None
        self._login_password = ""
        self._server_url = ""
        self._campaigns: dict[str, dict] = {}
        self._campaign_lock = threading.Lock()
        self._campaign_thread: threading.Thread | None = None
        self._campaign_cancel = threading.Event()
        self._expected_fault = False
        self._managed_server: ManagedServer | None = None
        self._agent_process: subprocess.Popen | None = None
        self._agent_log = None
        self._agent_url = ""
        self._agent_token = ""

    def _agent_health(self) -> dict:
        request = Request(
            f"{self._agent_url}/api/v1/health",
            headers={"Authorization": f"Bearer {self._agent_token}"},
        )
        with urlopen(request, timeout=2) as response:
            return json.load(response)

    def start_agent(self, *, timeout: float = 15) -> dict:
        if self._agent_process is not None and self._agent_process.poll() is None:
            raise RuntimeError("An Agent is already owned by this Inspector session")
        self.artifacts.mkdir(parents=True, exist_ok=True)
        if self.run_dir is None:
            self.run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=self.artifacts))
        connection_file = self.run_dir / "agent-connection.json"
        connection_file.unlink(missing_ok=True)
        self._agent_log = (self.run_dir / "agent.log").open("ab")
        env = {key: value for key, value in os.environ.items() if not key.startswith("KRAKEN_")}
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        try:
            self._agent_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "kraken_agent",
                    "--data-dir",
                    str(self.run_dir / "agent"),
                    "--connection-file",
                    str(connection_file),
                ],
                cwd=self.workspace,
                env=env,
                stdout=self._agent_log,
                stderr=subprocess.STDOUT,
                **kwargs,
            )
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self._agent_process.poll() is not None:
                    raise RuntimeError(
                        f"Agent exited with code {self._agent_process.returncode}; see {self.run_dir / 'agent.log'}"
                    )
                if connection_file.exists():
                    connection = json.loads(connection_file.read_text(encoding="utf-8"))
                    self._agent_url = connection["url"]
                    self._agent_token = connection["token"]
                    try:
                        health = self._agent_health()
                    except URLError, TimeoutError, OSError:
                        pass
                    else:
                        if health == {"status": "ok", "api_version": "v1"}:
                            return {"pid": self._agent_process.pid, "health": health, "url": self._agent_url}
                time.sleep(0.1)
            raise TimeoutError("Agent did not become healthy")
        except Exception:
            self.stop_agent()
            raise

    def stop_agent(self) -> dict:
        process = self._agent_process
        stopped = process is not None and process.poll() is None
        if stopped:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._agent_log is not None:
            self._agent_log.close()
            self._agent_log = None
        self._agent_process = None
        self._agent_url = ""
        self._agent_token = ""
        return {"stopped": stopped, "exit_code": process.returncode if process is not None else None}

    def start_server(self, *, postgres_bin: Path, timeout: float = 60) -> dict:
        if self._managed_server is not None:
            raise RuntimeError("A server is already owned by this Inspector session")
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("Start the disposable server before Hub")
        self.artifacts.mkdir(parents=True, exist_ok=True)
        if self.run_dir is None or self._agent_process is None:
            self.run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=self.artifacts))
        managed = ManagedServer(self.workspace, self.run_dir, postgres_bin)
        result = managed.start(timeout=timeout)
        self._managed_server = managed
        return result

    def stop_server(self) -> dict:
        if self.process is not None and self.process.poll() is None:
            self.stop()
        if self._managed_server is None:
            return {"stopped": False}
        result = self._managed_server.stop()
        self._managed_server = None
        return result

    def revision(self) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.workspace,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    def launch_hub(self, *, server_url: str = "", timeout: float = 45, _reuse_data: bool = False) -> dict:
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("Hub is already running in this Inspector session")
        if not (self.workspace / "src" / "kraken_hub").is_dir():
            raise ValueError("workspace is not a Kraken checkout")
        self.artifacts.mkdir(parents=True, exist_ok=True)
        if not _reuse_data and self._managed_server is None and self._agent_process is None:
            self.run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=self.artifacts))
            self._login_password = secrets.token_urlsafe(24)
        elif not self._login_password:
            self._login_password = secrets.token_urlsafe(24)
        if self.run_dir is None:
            raise RuntimeError("No Inspector data to restart")
        if server_url and (self._managed_server is None or server_url != self._managed_server.url):
            raise ValueError("Hub may connect only to the disposable server owned by this Inspector session")
        if self._managed_server is not None:
            server_url = self._managed_server.url
        self._server_url = server_url
        self._expected_fault = False
        self.socket_name = f"kraken-inspector-{secrets.token_hex(16)}"
        self.token = secrets.token_urlsafe(32)
        env = {key: value for key, value in os.environ.items() if not key.startswith("KRAKEN_")}
        env.update(
            {
                "KRAKEN_DATA_DIR": str(self.run_dir / "data"),
                "KRAKEN_DEV_AUTO_LOGIN": "1",
                "KRAKEN_DEV_USERNAME": "inspector",
                "KRAKEN_DEV_DISPLAY_NAME": "Kraken Inspector",
                "KRAKEN_DEV_PASSWORD": self._login_password,
                "KRAKEN_INSPECTOR_TOKEN": self.token,
            }
        )
        if server_url:
            env["KRAKEN_SERVER_URL"] = server_url
            assert self._managed_server is not None
            env["KRAKEN_INSPECTOR_SERVER_USERNAME"] = self._managed_server.username
            env["KRAKEN_INSPECTOR_SERVER_PASSWORD"] = self._managed_server.password
        else:
            env.pop("KRAKEN_SERVER_URL", None)
        command = [
            sys.executable,
            "-m",
            "kraken_hub",
            "--settings-dir",
            str(self.run_dir / "settings"),
            "--inspector-socket",
            self.socket_name,
        ]
        self._log_stream = (self.run_dir / "hub.log").open("a", encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            command,
            cwd=self.workspace,
            env=env,
            stdout=self._log_stream,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Hub exited during startup ({self.process.returncode}): {self.get_logs(30)}")
            try:
                result = self.inspect_ui(timeout_ms=500)
                if result.get("windows"):
                    return {
                        "pid": self.process.pid,
                        "run_dir": str(self.run_dir),
                        "revision": self.revision(),
                        "server_mode": bool(server_url),
                        "windows": [{"id": w["id"], "title": w["title"]} for w in result["windows"]],
                    }
            except ConnectionError, TimeoutError:
                pass
            time.sleep(0.1)
        raise TimeoutError(f"Hub did not become inspectable in {timeout}s; log: {self.get_logs(30)}")

    def restart_hub(self, *, timeout: float = 45) -> dict:
        if self.run_dir is None:
            raise RuntimeError("Hub has not been launched")
        self.stop()
        return self.launch_hub(server_url=self._server_url, timeout=timeout, _reuse_data=True)

    def _request(self, payload: dict, *, timeout_ms: int = 5000) -> dict:
        if not self.socket_name or not self.token:
            raise RuntimeError("Hub has not been launched")
        from PyQt6.QtNetwork import QLocalSocket

        socket = QLocalSocket()
        socket.connectToServer(self.socket_name)
        if not socket.waitForConnected(timeout_ms):
            raise ConnectionError(f"Hub Inspector is unavailable: {socket.errorString()}")
        socket.write(json.dumps({**payload, "token": self.token}).encode() + b"\n")
        if not socket.waitForBytesWritten(timeout_ms):
            raise TimeoutError("Inspector request was not sent")
        deadline = time.monotonic() + timeout_ms / 1000
        while not socket.canReadLine():
            remaining = int((deadline - time.monotonic()) * 1000)
            if remaining <= 0 or not socket.waitForReadyRead(remaining):
                raise TimeoutError("Hub did not answer Inspector request")
        return json.loads(bytes(socket.readLine()))

    def inspect_ui(self, *, timeout_ms: int = 5000) -> dict:
        return self._request({"op": "inspect"}, timeout_ms=timeout_ms)

    def wait_for_ui(self, *, window_title: str = "", text: str = "", timeout: float = 10) -> dict:
        if not window_title and not text:
            raise ValueError("window_title or text is required")
        if not 0 < timeout <= 60:
            raise ValueError("timeout must be between 0 and 60 seconds")
        started = time.monotonic()
        deadline = started + timeout
        while True:
            state = self.inspect_ui(timeout_ms=min(2000, max(100, int((deadline - time.monotonic()) * 1000))))
            for window in state.get("windows", []):
                if window_title and window["title"] == window_title:
                    return {"found": True, "window": window["id"], "elapsed_seconds": time.monotonic() - started}
                if text:
                    matching = next(
                        (node for node in window["nodes"] if node["text"] == text and node["visible"]), None
                    )
                    if matching is not None:
                        return {
                            "found": True,
                            "window": window["id"],
                            "element": matching["id"],
                            "elapsed_seconds": time.monotonic() - started,
                        }
            if time.monotonic() >= deadline:
                return {"found": False, "elapsed_seconds": time.monotonic() - started}
            time.sleep(0.05)

    def perform_action(self, element_id: str, action: str, value: str = "", *, x: float = 0.5, y: float = 0.5) -> dict:
        before = self.inspect_ui()
        target = next(
            (node for window in before.get("windows", []) for node in window["nodes"] if node["id"] == element_id), None
        )
        if target is not None and action == "type" and target.get("sensitive"):
            return {"error": "sensitive_field"}
        result = self._request({"op": "action", "id": element_id, "action": action, "value": value, "x": x, "y": y})
        self.record_action(element_id, action, value, result, before=before, target=target, x=x, y=y)
        if result.get("ok"):
            result["invariants"] = self.evaluate_invariants()
        return result

    def screenshot(self, window: int = 0) -> dict:
        result = self._request({"op": "screenshot", "window": window})
        if "png_base64" in result:
            import base64

            assert self.run_dir is not None
            path = self.run_dir / f"screenshot-{int(time.time() * 1000)}.png"
            path.write_bytes(base64.b64decode(result.pop("png_base64")))
            result["path"] = str(path)
        return result

    def record_action(
        self,
        element_id: str,
        action: str,
        value: str,
        result: dict,
        *,
        before: dict | None = None,
        target: dict | None = None,
        x: float = 0.5,
        y: float = 0.5,
    ) -> None:
        self.actions.append(
            {
                "at": datetime.now(UTC).isoformat(),
                "id": element_id,
                "action": action,
                "value": value,
                "result": result,
                "selector": {key: target.get(key, "") for key in ("class", "object_name", "text")} if target else None,
                "x": x,
                "y": y,
                "before_windows": [w["title"] for w in before.get("windows", [])] if before else [],
            }
        )

    def collect_manual_events(self) -> dict:
        result = self._request({"op": "events"})
        events = result.get("events", [])
        for event in events:
            self.record_action(
                event["id"],
                event["action"],
                event.get("value", ""),
                {"observed": True, "source": "manual"},
                target=event,
                x=event.get("x") if event.get("x") is not None else 0.5,
                y=event.get("y") if event.get("y") is not None else 0.5,
            )
        return {"count": len(events), "events": events}

    def save_scenario(self) -> Path:
        if self.process is not None and self.process.poll() is None:
            self.collect_manual_events()
        self.artifacts.mkdir(parents=True, exist_ok=True)
        path = self.artifacts / f"scenario-{secrets.token_hex(8)}.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "revision": self.revision(),
                    "created_at": datetime.now(UTC).isoformat(),
                    "run_dir": str(self.run_dir) if self.run_dir else None,
                    "actions": self.actions,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def replay_scenario(self, path: Path) -> dict:
        resolved = path.resolve()
        if self.artifacts.resolve() not in resolved.parents:
            raise ValueError("scenario must be under Inspector artifacts")
        data = json.loads(resolved.read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("actions"), list):
            raise ValueError("unsupported scenario")
        outcomes = []
        for item in data["actions"]:
            current = self.inspect_ui()
            nodes = [node for window in current.get("windows", []) for node in window["nodes"]]
            selector = item.get("selector")
            if selector:
                matching = [
                    node for node in nodes if all(node.get(key, "") == value for key, value in selector.items())
                ]
                if len(matching) != 1:
                    outcomes.append({"error": "selector_not_unique", "selector": selector, "matches": len(matching)})
                    break
                element_id = matching[0]["id"]
            else:
                element_id = item["id"]
            outcome = self.perform_action(
                element_id,
                item["action"],
                item.get("value", ""),
                x=float(item.get("x", 0.5)),
                y=float(item.get("y", 0.5)),
            )
            outcomes.append(outcome)
            if "error" in outcome:
                break
        return {
            "attempted": len(outcomes),
            "total": len(data["actions"]),
            "outcomes": outcomes,
            "actions_dispatched": len(outcomes) == len(data["actions"]) and all("ok" in x for x in outcomes),
            "behavior_verified": False,
        }

    def explore_states(
        self,
        *,
        steps: int = 10,
        seed: int = 0,
        cancel: threading.Event | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> dict:
        import random

        if not 1 <= steps <= 100:
            raise ValueError("steps must be between 1 and 100")
        rng = random.Random(seed)
        visited: set[tuple[str, str]] = set()
        transitions = []
        for _ in range(steps):
            if cancel is not None and cancel.is_set():
                break
            state = self.inspect_ui()
            candidates = [
                node
                for window in state.get("windows", [])
                for node in window["nodes"]
                if node["class"] in {"QPushButton", "QToolButton", "QAction"}
                and node["enabled"]
                and node["visible"]
                and node["text"]
            ]
            available = [node for node in candidates if (node["id"], node["text"]) not in visited]
            if not available:
                break
            node = rng.choice(available)
            visited.add((node["id"], node["text"]))
            outcome = self.perform_action(node["id"], "click")
            checks = outcome.get("invariants", {"checks": [], "violations": []})
            transitions.append(
                {
                    "id": node["id"],
                    "text": node["text"],
                    "result": {key: value for key, value in outcome.items() if key != "invariants"},
                    "invariants": checks,
                }
            )
            if progress is not None:
                progress(len(transitions))
            if "error" in outcome or checks["violations"]:
                break
        return {
            "seed": seed,
            "transitions": transitions,
            "visited": len(visited),
            "cancelled": cancel.is_set() if cancel is not None else False,
            "invariants": self.evaluate_invariants(),
        }

    def start_exploration(self, *, steps: int = 10, seed: int = 0) -> dict:
        if not 1 <= steps <= 100:
            raise ValueError("steps must be between 1 and 100")
        if self._campaign_thread is not None and self._campaign_thread.is_alive():
            raise RuntimeError("an exploration campaign is already running")
        self.inspect_ui()
        identifier = secrets.token_hex(8)
        self._campaign_cancel.clear()
        with self._campaign_lock:
            self._campaigns[identifier] = {
                "id": identifier,
                "status": "running",
                "completed_steps": 0,
                "requested_steps": steps,
                "seed": seed,
            }

        def progress(count: int) -> None:
            with self._campaign_lock:
                self._campaigns[identifier]["completed_steps"] = count

        def run() -> None:
            try:
                result = self.explore_states(steps=steps, seed=seed, cancel=self._campaign_cancel, progress=progress)
                status = "cancelled" if result["cancelled"] else "complete"
                self.artifacts.mkdir(parents=True, exist_ok=True)
                path = self.artifacts / f"campaign-{identifier}.json"
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                with self._campaign_lock:
                    self._campaigns[identifier].update({"status": status, "result": result, "path": str(path)})
            except Exception as exc:
                with self._campaign_lock:
                    self._campaigns[identifier].update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})

        self._campaign_thread = threading.Thread(target=run, name="kraken-inspector-explorer", daemon=True)
        self._campaign_thread.start()
        return {"id": identifier, "status": "running"}

    def get_exploration(self, identifier: str) -> dict:
        with self._campaign_lock:
            campaign = self._campaigns.get(identifier)
            if campaign is None:
                raise ValueError("unknown campaign")
            return dict(campaign)

    def cancel_exploration(self, identifier: str) -> dict:
        campaign = self.get_exploration(identifier)
        if campaign["status"] == "running":
            self._campaign_cancel.set()
        return self.get_exploration(identifier)

    def evaluate_invariants(self) -> dict:
        if self.process is None and self._managed_server is None and self._agent_process is None:
            return {"checks": [], "violations": [], "status": "not_evaluated"}
        checks = []
        if self.process is not None:
            alive = self.process.poll() is None
            checks.append({"name": "hub_alive", "status": "pass" if alive else "fail"})
            if alive:
                try:
                    self.inspect_ui(timeout_ms=2000)
                    checks.append({"name": "ui_responsive", "status": "pass"})
                except (ConnectionError, TimeoutError) as exc:
                    checks.append({"name": "ui_responsive", "status": "fail", "detail": str(exc)})
        if self._agent_process is not None:
            try:
                health = self._agent_health()
                checks.append(
                    {
                        "name": "agent_healthy",
                        "status": "pass" if health == {"status": "ok", "api_version": "v1"} else "fail",
                        "health": health,
                    }
                )
            except (OSError, TimeoutError) as exc:
                checks.append({"name": "agent_healthy", "status": "fail", "detail": str(exc)})
        if self._managed_server is not None:
            try:
                healthy = self._managed_server.health()
                checks.append(
                    {
                        "name": "server_healthy",
                        "status": "pass"
                        if healthy.get("status") == "ok" and healthy.get("metadata") == "postgresql"
                        else "fail",
                        "health": healthy,
                    }
                )
            except (OSError, TimeoutError) as exc:
                checks.append({"name": "server_healthy", "status": "fail", "detail": str(exc)})
        violations = [check for check in checks if check["status"] == "fail"]
        for violation in violations:
            if self._expected_fault:
                continue
            if not any(
                item.get("kind") == "invariant"
                and item.get("name") == violation["name"]
                and item.get("run_dir") == str(self.run_dir)
                for item in self.defects
            ):
                evidence_path = self._write_defect_evidence(violation)
                self.defects.append(
                    {
                        "kind": "invariant",
                        "name": violation["name"],
                        "run_dir": str(self.run_dir),
                        "evidence_path": str(evidence_path),
                    }
                )
        return {"checks": checks, "violations": violations}

    def _write_defect_evidence(self, violation: dict) -> Path:
        if self.run_dir is None:
            raise RuntimeError("No run directory for defect evidence")
        path = self.run_dir / f"defect-{secrets.token_hex(8)}.json"
        path.write_text(
            json.dumps(
                {
                    "revision": self.revision(),
                    "violation": violation,
                    "actions": self.actions,
                    "log_tail": self.get_logs(100),
                    "at": datetime.now(UTC).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def inject_fault(self, kind: str) -> dict:
        if kind != "hub_termination":
            raise ValueError("supported fault: hub_termination")
        if self.process is None or self.process.poll() is not None or self.run_dir is None:
            raise RuntimeError("No owned Hub process is running")
        import psutil

        owned = psutil.Process(self.process.pid)
        children = owned.children(recursive=True)
        self._expected_fault = True
        for child in children:
            child.kill()
        owned.kill()
        psutil.wait_procs([*children, owned], timeout=5)
        self.process.wait(timeout=5)
        evidence_path = self._write_defect_evidence({"name": "hub_termination", "status": "injected"})
        return {
            "injected": True,
            "kind": kind,
            "actual_os_termination": True,
            "restart_available": True,
            "exit_code": self.process.returncode,
            "evidence_path": str(evidence_path),
        }

    def get_logs(self, lines: int = 100) -> str:
        if self.run_dir is None:
            return ""
        path = self.run_dir / "hub.log"
        if not path.exists():
            return ""
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, min(lines, 500)) :])

    def environment(self) -> dict:
        process = self.process
        result = {
            "revision": self.revision(),
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "hub_pid": process.pid if process else None,
            "hub_alive": process is not None and process.poll() is None,
        }
        if self._managed_server is not None:
            result["server_pid"] = self._managed_server.process.pid if self._managed_server.process else None
            result["server_url"] = self._managed_server.url
        if self._agent_process is not None:
            result["agent_pid"] = self._agent_process.pid
            result["agent_alive"] = self._agent_process.poll() is None
            result["agent_url"] = self._agent_url
        if process is not None and process.poll() is None:
            import psutil

            monitored = psutil.Process(process.pid)
            with monitored.oneshot():
                result.update(
                    {
                        "rss_bytes": monitored.memory_info().rss,
                        "thread_count": monitored.num_threads(),
                        "child_pids": [child.pid for child in monitored.children(recursive=True)],
                    }
                )
        return result

    def _sample(self) -> dict:
        import psutil

        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Hub is not running")
        monitored = psutil.Process(self.process.pid)
        with monitored.oneshot():
            info = monitored.memory_info()
            io = monitored.io_counters()
            sample = {
                "at": datetime.now(UTC).isoformat(),
                "rss_bytes": info.rss,
                "cpu_percent": monitored.cpu_percent(interval=None),
                "threads": monitored.num_threads(),
                "children": len(monitored.children(recursive=True)),
                "read_bytes": io.read_bytes,
                "write_bytes": io.write_bytes,
                "system_available_bytes": psutil.virtual_memory().available,
            }
            if hasattr(monitored, "num_handles"):
                sample["handles"] = monitored.num_handles()
            return sample

    def start_monitoring(self, *, interval_ms: int = 500, max_rss_growth_bytes: int | None = None) -> dict:
        if not 100 <= interval_ms <= 60_000:
            raise ValueError("interval_ms must be between 100 and 60000")
        if max_rss_growth_bytes is not None and max_rss_growth_bytes < 0:
            raise ValueError("max_rss_growth_bytes must be nonnegative")
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            raise RuntimeError("monitoring is already active")
        self._samples.clear()
        self._samples.append(self._sample())
        self._max_rss_growth_bytes = max_rss_growth_bytes
        self._monitor_stop.clear()

        def sample_loop() -> None:
            while not self._monitor_stop.wait(interval_ms / 1000):
                try:
                    self._samples.append(self._sample())
                except RuntimeError, OSError:
                    break

        self._monitor_thread = threading.Thread(target=sample_loop, name="kraken-inspector-monitor", daemon=True)
        self._monitor_thread.start()
        return {"started": True, "interval_ms": interval_ms}

    def get_monitoring_results(self, *, limit: int = 100) -> dict:
        samples = list(self._samples)
        anomalies = []
        if samples and self._max_rss_growth_bytes is not None:
            growth = samples[-1]["rss_bytes"] - samples[0]["rss_bytes"]
            if growth > self._max_rss_growth_bytes:
                anomalies.append(
                    {
                        "kind": "rss_growth",
                        "growth_bytes": growth,
                        "limit_bytes": self._max_rss_growth_bytes,
                        "classification": "policy_violation",
                    }
                )
        return {
            "running": self._monitor_thread is not None and self._monitor_thread.is_alive(),
            "sample_count": len(samples),
            "samples": samples[-max(1, min(limit, 200)) :],
            "anomalies": anomalies,
        }

    def stop_monitoring(self) -> dict:
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2)
        return self.get_monitoring_results()

    def stop(self) -> dict:
        self._campaign_cancel.set()
        if self._campaign_thread is not None:
            self._campaign_thread.join(timeout=6)
        self.stop_monitoring()
        process = self.process
        if process is None or process.poll() is not None:
            if self._log_stream is not None:
                self._log_stream.close()
                self._log_stream = None
            return {"stopped": False}
        import psutil

        owned = psutil.Process(process.pid)
        children = owned.children(recursive=True)
        for child in children:
            child.terminate()
        owned.terminate()
        _, alive = psutil.wait_procs([*children, owned], timeout=5)
        for item in alive:
            item.kill()
        process.wait(timeout=5)
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None
        return {"stopped": True, "exit_code": process.returncode}
