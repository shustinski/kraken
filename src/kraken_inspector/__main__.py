"""Run the Inspector MCP server on standard input and output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .mcp import InspectorMCP, serve
from .session import InspectorSession


def main() -> None:
    parser = argparse.ArgumentParser(prog="kraken-inspector")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    serve(InspectorMCP(InspectorSession(args.workspace)), sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    main()
