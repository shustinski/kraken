"""Single version source for Karakal builds and auto-update.

Version is <release>.<algorithm number><interface number>-beta<N>, e.g. 0.2.9101-beta0:
- the leading digits of the third part are the grid-defect algorithm version
  (core/algorithm_version.py), so a build tells which analysis it runs;
- the last two digits are the interface version: +1 for every interface change
  (01 = gradient panel with drop rules, frame sets and set export);
- beta N counts tester builds with the same algorithm and interface.
"""

from __future__ import annotations

from .core.algorithm_version import GRID_ALGORITHM_NUMBER

RELEASE = "0.2"
INTERFACE_NUMBER = 1
BETA = 0

APP_VERSION = f"{RELEASE}.{GRID_ALGORITHM_NUMBER}{INTERFACE_NUMBER:02d}-beta{BETA}"
__version__ = APP_VERSION


def numeric_version(version: str | None = None) -> str:
    """Return Inno-compatible numeric VersionInfoVersion (X.Y.Z.0)."""

    text = str(version or APP_VERSION).strip()
    core = text.split("-", 1)[0].split("+", 1)[0]
    parts = [part for part in core.split(".") if part.isdigit()]
    while len(parts) < 4:
        parts.append("0")
    return ".".join(parts[:4])
