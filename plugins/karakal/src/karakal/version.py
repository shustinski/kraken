"""Single version source for Karakal builds and auto-update.

Version is <release>.<algorithm number>-beta<N>, e.g. 0.2.91-beta0:
- the middle number is the grid-defect algorithm version (core/algorithm_version.py),
  so a build tells which analysis it runs;
- beta N counts tester builds of one algorithm version; it goes back to 0 when the
  algorithm number changes.
"""

from __future__ import annotations

from .core.algorithm_version import GRID_ALGORITHM_NUMBER

RELEASE = "0.2"
BETA = 0

APP_VERSION = f"{RELEASE}.{GRID_ALGORITHM_NUMBER}-beta{BETA}"
__version__ = APP_VERSION


def numeric_version(version: str | None = None) -> str:
    """Return Inno-compatible numeric VersionInfoVersion (X.Y.Z.0)."""

    text = str(version or APP_VERSION).strip()
    core = text.split("-", 1)[0].split("+", 1)[0]
    parts = [part for part in core.split(".") if part.isdigit()]
    while len(parts) < 4:
        parts.append("0")
    return ".".join(parts[:4])
