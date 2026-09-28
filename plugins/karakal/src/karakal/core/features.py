"""Build profile for Karakal.

``dev`` is the default: every feature stays available, including unfinished ones.
``tester`` hides unfinished UI. From source it is selected with
``KARAKAL_BUILD_PROFILE=tester``. A frozen tester build reads the profile file
baked in at build time and ignores the environment variable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

TESTER_VERSION = "0.1.0-test1"
_PROFILES = {"dev", "tester"}


def _bundled_profile_text() -> str:
    if not getattr(sys, "frozen", False):
        return ""
    base = Path(getattr(sys, "_MEIPASS", ""))
    for name in ("build_profile.txt", "karakal_build_profile.txt"):
        path = base / "resources" / name
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            return text
    return ""


def build_profile() -> str:
    if getattr(sys, "frozen", False):
        lines = _bundled_profile_text().splitlines()
        name = (lines[0] if lines else "dev").strip().lower()
        return "tester" if name == "tester" else "dev"
    raw = os.environ.get("KARAKAL_BUILD_PROFILE", "dev").strip().lower()
    return raw if raw in _PROFILES else "dev"


def tester_build() -> bool:
    return build_profile() == "tester"


def display_version() -> str:
    if not tester_build():
        from ..version import __version__

        return __version__
    if getattr(sys, "frozen", False):
        lines = _bundled_profile_text().splitlines()
        if len(lines) > 1 and lines[1].strip():
            return lines[1].strip()
    return TESTER_VERSION


def show_class_conflict() -> bool:
    return not tester_build()


def show_pair_matrices() -> bool:
    return not tester_build()


def show_other_analysis_profiles() -> bool:
    return not tester_build()


def show_app_mode_switch() -> bool:
    return not tester_build()


def show_reference_frame() -> bool:
    return not tester_build()


def show_conductor_zone() -> bool:
    return not tester_build()


def show_threshold_window() -> bool:
    return not tester_build()


def show_grid_examples() -> bool:
    return not tester_build()


def show_other_mode_export() -> bool:
    return not tester_build()
