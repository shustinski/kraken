"""Alias for the standalone Qt entrypoint."""
from __future__ import annotations

import sys
from pathlib import Path


if __package__ in {None, ""}:
    package_parent = Path(__file__).resolve().parents[2]
    package_parent_text = str(package_parent)
    if package_parent_text not in sys.path:
        sys.path.insert(0, package_parent_text)
    from karakal.debug.standalone_run import main
else:
    from .standalone_run import main


if __name__ == "__main__":
    raise SystemExit(main())
