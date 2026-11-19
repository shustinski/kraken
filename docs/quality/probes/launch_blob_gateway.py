"""Launch the Kraken Blob Gateway exactly the way the server's own lifecycle would.

The user's ``config/local/server.toml`` has ``[blob_gateway] enabled = false``,
so the server does not start the gateway itself: ``ServerConfig.apply_to_environment``
clears every ``KRAKEN_BLOB_GATEWAY_*`` variable when the section is disabled, and
``kraken_server.composition.postgresql_composition`` only constructs a
``BlobGatewayManager`` when ``KRAKEN_BLOB_GATEWAY_PUBLIC_URL`` is present.

Instead of editing the user's configuration, the bring-up probe mirrors
``BlobGatewayManager.start()`` with the values from the ``[blob_gateway]``
section of server.toml:

* same executable (``[blob_gateway].executable``);
* same child environment — ``BlobGatewayManager._child_environment()`` is called
  directly, so ``KRAKEN_BLOB_ROOT`` / ``KRAKEN_BLOB_GATEWAY_BIND`` /
  ``KRAKEN_BLOB_GATEWAY_SECRET`` (plus the same base ``PATH`` / ``SYSTEMROOT`` /
  ... variable set) are byte-identical to what the server would pass. That is
  what keeps the gateway's ticket-HMAC secret in sync with the server's
  ``BlobTicketSigner``;
* same ``subprocess.Popen`` flags (no arguments, ``stdin=DEVNULL``,
  ``CREATE_NO_WINDOW`` on Windows). The only deliberate deviation: stdout and
  stderr are appended to the log file supplied by the probe instead of being
  inherited, so the gateway output is captured for the evidence record.

Usage::

    python launch_blob_gateway.py --config <server.toml> --log <file> --pid-file <file>

Prints one JSON document on stdout:

* ``{"status": "ok", "pid": N, "executable": ..., "bind": ..., "public_url": ..., "blob_root": ...}``
  when the gateway was started, or
* ``{"status": "BLOCKED", "error": "<exact reason>", ...paths...}`` otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tomllib
from pathlib import Path

from kraken_server.blob_gateway import BlobGatewayManager
from kraken_server.configuration import unprotect_secret


def load_gateway_manager(config_path: Path) -> BlobGatewayManager:
    """Build the exact BlobGatewayManager the server would construct if enabled."""
    with config_path.open("rb") as stream:
        document = tomllib.load(stream)
    gateway = document.get("blob_gateway", {})
    storage = document.get("storage", {})

    def resolve(value: object) -> Path:
        # Same semantics as ServerConfig.load's internal ``resolve``.
        text = str(value or "").strip()
        if not text:
            raise ValueError("Required Kraken Server path is missing")
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = config_path.parent / candidate
        return candidate.resolve()

    return BlobGatewayManager(
        public_url=str(gateway.get("public_url", "")),
        bind=str(gateway.get("bind", "")),
        blob_root=resolve(storage.get("blob_root")),
        executable=resolve(gateway.get("executable")),
        secret=unprotect_secret(resolve(gateway.get("secret"))),
        ticket_lifetime_seconds=int(gateway.get("ticket_lifetime_seconds", 900)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Kraken Blob Gateway for the bring-up probe")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--pid-file", required=True, type=Path)
    args = parser.parse_args()

    try:
        manager = load_gateway_manager(args.config)
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1

    paths = {
        "executable": str(manager.executable),
        "bind": manager.bind,
        "public_url": manager.public_url,
        "blob_root": str(manager.blob_root),
    }
    if not manager.executable.is_file():
        # Same error text BlobGatewayManager.start() raises for a missing binary.
        print(json.dumps(
            {"status": "BLOCKED", "error": f"Kraken Blob Gateway executable was not found: {manager.executable}",
             **paths}, ensure_ascii=False))
        return 1

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.pid_file.parent.mkdir(parents=True, exist_ok=True)
    environment = manager._child_environment()  # exact server-lifecycle child environment
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with args.log.open("ab") as log_file:
        process = subprocess.Popen(  # noqa: S603 - executable is administrator-owned configuration
            [str(manager.executable)],
            env=environment,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    args.pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "pid": process.pid, **paths}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
