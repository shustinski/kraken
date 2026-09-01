"""Run Kraken Server with Uvicorn."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .configuration import ServerConfig, default_config_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Kraken shared project server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--config", type=Path, help="Path to packaged server.toml")
    parser.add_argument("--service", action="store_true", help="Run under Windows Service Control Manager")
    parser.add_argument(
        "--development",
        action="store_true",
        help="Use ephemeral services and development Bearer identities (never for production)",
    )
    args = parser.parse_args()
    config_path = args.config or default_config_path()
    configuration = None
    if config_path.is_file():
        configuration = ServerConfig.load(config_path)
        configuration.apply_to_environment()
        os.environ["KRAKEN_SERVER_CONFIG"] = str(configuration.path)
    if args.service:
        from .windows_service import run_windows_service

        run_windows_service(config_path)
        return 0
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install Kraken with the 'server' extra") from exc
    if args.development:
        os.environ["KRAKEN_SERVER_DEVELOPMENT"] = "1"
    elif configuration is None and not os.environ.get("KRAKEN_SERVER_COMPOSITION"):
        raise SystemExit(
            "Production composition is not configured. Set KRAKEN_SERVER_COMPOSITION or use --development explicitly."
        )
    uvicorn.run(
        "kraken_server.runtime:create_app_from_environment",
        host=configuration.host if configuration is not None else args.host,
        port=configuration.port if configuration is not None else args.port,
        reload=args.reload,
        factory=True,
        ssl_certfile=(
            None if configuration is None or configuration.tls_cert_file is None
            else str(configuration.tls_cert_file)
        ),
        ssl_keyfile=(
            None if configuration is None or configuration.tls_key_file is None
            else str(configuration.tls_key_file)
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
