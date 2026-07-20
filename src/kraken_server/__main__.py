"""Run Kraken Server with Uvicorn."""

from __future__ import annotations

import argparse
import os


def main() -> int:
    parser = argparse.ArgumentParser(description="Kraken shared project server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--development",
        action="store_true",
        help="Use ephemeral services and development Bearer identities (never for production)",
    )
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install Kraken with the 'server' extra") from exc
    if args.development:
        os.environ["KRAKEN_SERVER_DEVELOPMENT"] = "1"
    elif not os.environ.get("KRAKEN_SERVER_COMPOSITION"):
        raise SystemExit(
            "Production composition is not configured. Set KRAKEN_SERVER_COMPOSITION or use --development explicitly."
        )
    uvicorn.run(
        "kraken_server.runtime:create_app_from_environment",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
