"""Allow `python -m karakal`."""

from __future__ import annotations

from .debug.standalone_run import main


if __name__ == "__main__":
    raise SystemExit(main())
