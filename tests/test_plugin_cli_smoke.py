"""Палитра W3 — CLI-доступность плагинов бета-объёма. S13 (CLI)."""

from __future__ import annotations

import subprocess
import sys


def test_contour_help_exits_zero():
    completed = subprocess.run(
        [sys.executable, "-m", "contour", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip()


def test_neuralimage_version_exits_zero():
    completed = subprocess.run(
        [sys.executable, "-m", "neuralimage", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip()
