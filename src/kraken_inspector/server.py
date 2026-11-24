"""Disposable PostgreSQL and real Kraken Server owned by one Inspector run."""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class ManagedServer:
    def __init__(self, workspace: Path, run_dir: Path, postgres_bin: Path) -> None:
        self.workspace = workspace
        self.run_dir = run_dir
        self.postgres_bin = postgres_bin
        self.cluster = run_dir / "postgres"
        self.process: subprocess.Popen | None = None
        self._server_log = None
        self._postgres_started = False
        self.url = ""
        self.username = "inspector"
        self.password = secrets.token_urlsafe(24)

    def _pg_ctl(self, *arguments: str) -> None:
        with (self.run_dir / "pgctl.log").open("ab") as log:
            result = subprocess.run(
                [str(self.postgres_bin / "pg_ctl.exe"), "-D", str(self.cluster), *arguments],
                cwd=self.workspace,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=40,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"pg_ctl failed ({result.returncode}); see {self.run_dir / 'pgctl.log'}")

    def health(self) -> dict:
        with urlopen(f"{self.url}/api/v1/health", timeout=2) as response:
            return json.load(response)

    def start(self, *, timeout: float = 60) -> dict:
        if not (self.postgres_bin / "initdb.exe").is_file() or not (self.postgres_bin / "pg_ctl.exe").is_file():
            raise ValueError("PostgreSQL initdb and pg_ctl binaries are required")
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("Server is already running")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "initdb.log").open("wb") as log:
            created = subprocess.run(
                [
                    str(self.postgres_bin / "initdb.exe"),
                    "-D",
                    str(self.cluster),
                    "-U",
                    "kraken_audit",
                    "--auth=trust",
                    "--encoding=UTF8",
                    "--no-locale",
                ],
                cwd=self.workspace,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=60,
                check=False,
            )
        if created.returncode != 0:
            raise RuntimeError(f"Temporary initdb failed; see {self.run_dir / 'initdb.log'}")
        pg_port = _free_port()
        self._pg_ctl("-l", str(self.run_dir / "postgres.log"), "-o", f"-h 127.0.0.1 -p {pg_port}", "-w", "start")
        self._postgres_started = True
        try:
            database_url = f"postgresql+psycopg://kraken_audit@127.0.0.1:{pg_port}/postgres?connect_timeout=5"
            from kraken_server.configuration import run_migrations, write_config

            with (self.run_dir / "migration.log").open("w", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    run_migrations(database_url)
            server_port = _free_port()
            config = write_config(
                self.run_dir / "server.toml",
                database_url=database_url,
                blob_root=self.run_dir / "blobs",
                source_root=self.run_dir / "source",
                derived_root=self.run_dir / "derived",
                host="127.0.0.1",
                port=server_port,
            )
            self.url = f"http://127.0.0.1:{server_port}"
            self._server_log = (self.run_dir / "server.log").open("wb")
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            isolated_environment = {key: value for key, value in os.environ.items() if not key.startswith("KRAKEN_")}
            self.process = subprocess.Popen(
                [sys.executable, "-m", "kraken_server", "--config", str(config.path)],
                cwd=self.workspace,
                env=isolated_environment,
                stdout=self._server_log,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Kraken Server exited during startup ({self.process.returncode})")
                try:
                    health = self.health()
                    if health.get("status") == "ok":
                        break
                except URLError, TimeoutError, OSError:
                    pass
                time.sleep(0.1)
            else:
                raise TimeoutError("Kraken Server did not become healthy")
            request = Request(
                f"{self.url}/api/v1/auth/accounts",
                method="POST",
                data=json.dumps(
                    {"username": self.username, "display_name": "Kraken Inspector", "password": self.password}
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=10) as response:
                if response.status != 201:
                    raise RuntimeError(f"Server account registration returned {response.status}")
            return {
                "url": self.url,
                "pid": self.process.pid,
                "health": health,
                "run_dir": str(self.run_dir),
                "database": "disposable_postgresql",
            }
        except Exception:
            self.stop()
            raise

    def stop(self) -> dict:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self._server_log is not None:
            self._server_log.close()
            self._server_log = None
        if self._postgres_started:
            self._pg_ctl("-w", "stop")
            self._postgres_started = False
        return {"stopped": True}
