"""Single version source for Karakal builds and auto-update.

Version is <release>.<algorithm number><interface number>-beta<N>, e.g. 0.2.9101-beta0:
- the leading digits of the third part are the grid-defect algorithm version
  (core/algorithm_version.py), so a build tells which analysis it runs;
- the last two digits are the interface version: +1 for every interface change
  (01 = gradient panel with drop rules, frame sets and set export;
  02 = drop by the colour scale only, no defect sets, set note and badge above the matrix;
  03 = export only from the drop panel, numbers next to their names, black frames for the rest of the run;
  04 = update menu: channel, check and what's new only, no Help menu;
  05 = export folder name, no export report, no fill filters, collapsible cell defects block;
  06 = minimum debris size setting in the main and frame windows, debris size on hover;
  07 = matrix follows a larger debris size and switched-off types at once, stale-matrix warning;
  08 = masks and confidence maps in one folder, Settings menu with file suffixes;
  09 = split and unknown defect types, no edge-clip type);
- beta N counts tester builds with the same algorithm and interface; scripts/publish.ps1
  picks the next free N from the update folder and writes it back here.

A stable release is the same number without the suffix, e.g. 0.2.9103: newer than any
0.2.9103-betaN, older than the next 0.2.9104-beta0.
"""

from __future__ import annotations

from .core.algorithm_version import GRID_ALGORITHM_NUMBER

RELEASE = "0.3"
INTERFACE_NUMBER = 9
BETA = 0


def base_version() -> str:
    """Release number without the beta suffix: the stable version of this code."""

    return f"{RELEASE}.{GRID_ALGORITHM_NUMBER}{INTERFACE_NUMBER:02d}"


APP_VERSION = f"{base_version()}-beta{BETA}"
__version__ = APP_VERSION


def numeric_version(version: str | None = None) -> str:
    """Return Inno-compatible numeric VersionInfoVersion (X.Y.Z.0)."""

    text = str(version or APP_VERSION).strip()
    core = text.split("-", 1)[0].split("+", 1)[0]
    parts = [part for part in core.split(".") if part.isdigit()]
    while len(parts) < 4:
        parts.append("0")
    return ".".join(parts[:4])
