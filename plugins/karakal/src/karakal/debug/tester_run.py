"""Run Karakal from source the way testers see it.

The tester exe bakes the ``tester`` build profile in at build time: only the
grid-defect analysis, Russian UI, no saved calibration, the version in the title.
From source the same UI comes from ``KARAKAL_BUILD_PROFILE=tester``; this entry
point sets it before anything of Karakal is imported. Settings go to their own
settings.ini (the exe keeps one beside itself), so trying the tester UI does not
touch the developer settings.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepare_tester_environment() -> Path:
    os.environ["KARAKAL_BUILD_PROFILE"] = "tester"
    settings_dir = str(os.environ.get("KARAKAL_SETTINGS_DIR", "") or "").strip()
    if not settings_dir:
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "Karakal" / "tester_from_source"
        settings_dir = str(base)
        os.environ["KARAKAL_SETTINGS_DIR"] = settings_dir
    path = Path(settings_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


_prepare_tester_environment()

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
