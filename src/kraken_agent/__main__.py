"""Command line entry point for Kraken Agent."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
from pathlib import Path

from .runner import PluginRegistry, SubprocessPluginRunner
from .service import AgentControlServer


def _default_data_dir() -> Path:
    override = os.environ.get("KRAKEN_AGENT_DATA")
    if override:
        return Path(override)
    return Path.home() / ".kraken" / "agent"


def main() -> int:
    parser = argparse.ArgumentParser(description="Kraken durable local plugin agent")
    parser.add_argument("--data-dir", type=Path, default=_default_data_dir())
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "::1"))
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--token", help="One-time control token; generated when omitted")
    parser.add_argument("--plugins-config", type=Path, help="JSON operation-to-command registry")
    parser.add_argument("--connection-file", type=Path)
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    control = AgentControlServer.create(args.data_dir / "jobs.sqlite3", token=args.token)
    control.host = args.host
    control.port = args.port
    recovered = control.store.recover_interrupted()
    httpd = control.build_http_server()
    runner = None
    runner_thread = None
    if args.plugins_config is not None:
        runner = SubprocessPluginRunner(
            control.store,
            args.data_dir / "staging",
            PluginRegistry.from_json(args.plugins_config),
        )
        runner_thread = threading.Thread(
            target=runner.run_forever,
            name="kraken-agent-worker",
            daemon=True,
        )
        runner_thread.start()
    if args.connection_file is not None:
        args.connection_file.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{args.connection_file.name}.",
            suffix=".tmp",
            dir=args.connection_file.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "api_version": "v1",
                        "pid": os.getpid(),
                        "url": f"http://{args.host}:{control.port}",
                        "token": control.token,
                    },
                    stream,
                    ensure_ascii=False,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, args.connection_file)
            try:
                os.chmod(args.connection_file, 0o600)
            except OSError:
                pass
        finally:
            Path(temporary_name).unlink(missing_ok=True)
    print(
        f"Kraken Agent listening on http://{args.host}:{control.port}; "
        f"recovered={recovered}"
    )
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if runner is not None:
            runner.stop()
        if runner_thread is not None:
            runner_thread.join(timeout=10)
        httpd.server_close()
        if args.connection_file is not None:
            args.connection_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
