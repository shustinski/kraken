"""Allow `python -m cartograph`."""

from __future__ import annotations

from .application.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
