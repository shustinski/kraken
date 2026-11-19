"""Regression: root uv sync does not install NeuralImage's CUDA PyTorch extra."""

from __future__ import annotations

import functools
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

kraken_defect = "D17"


@functools.lru_cache(maxsize=1)
def _offline_resolution_available() -> bool:
    """Whether ``uv`` can resolve the locked dependency set offline (D21).

    Same idiom as ``tests/offline/test_offline_sync.py::_pypi_is_reachable``:
    one cached subprocess probe per run, so a closed-network machine skips the
    resolution-dependent tests deterministically instead of failing the gate.
    The probe executes exactly what ``_dry_run("--all-groups", "--all-extras")``
    runs — the full ``uv sync --directory <ROOT> --locked --dry-run --offline
    --all-groups --all-extras`` chain against a throwaway
    ``UV_PROJECT_ENVIRONMENT`` — because availability of the contour build
    deps alone is not sufficient: with the offline kit cache the build deps
    resolve, yet the locked resolution still fails on the unsatisfiable
    Python>=3.15 torch fork (see D21). Exit 0 ⇒ True; any nonzero ⇒ False.
    """
    with tempfile.TemporaryDirectory() as scratch:
        try:
            completed = subprocess.run(
                [
                    "uv",
                    "sync",
                    "--directory",
                    str(ROOT),
                    "--locked",
                    "--dry-run",
                    "--offline",
                    "--all-groups",
                    "--all-extras",
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=dict(os.environ, UV_PROJECT_ENVIRONMENT=scratch),
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError):
            return False
    return completed.returncode == 0


def _require_offline_resolution() -> None:
    if not _offline_resolution_available():
        pytest.skip(
            "offline uv resolution unavailable on this machine (build deps and/or "
            "cu132 index pages missing, or unresolved Python>=3.15 torch fork) — see D21"
        )


def _dry_run(*extra: str) -> str:
    """``uv sync --dry-run`` against a throwaway empty project environment (D17).

    The dry run reports the difference between the locked resolution and the
    current project environment. Diffing against the machine's real ``.venv``
    makes the assertions state-dependent: with torch already installed it
    appears in the "Would uninstall" list (and with it missing from a
    workspace sync it appears in none). Pointing ``UV_PROJECT_ENVIRONMENT`` at
    a fresh empty directory makes the result depend only on ``uv.lock``.
    """
    with tempfile.TemporaryDirectory() as scratch:
        completed = subprocess.run(
            [
                "uv",
                "sync",
                "--directory",
                str(ROOT),
                "--locked",
                "--dry-run",
                "--offline",
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=dict(os.environ, UV_PROJECT_ENVIRONMENT=scratch),
        )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout + completed.stderr


def test_offline_resolution_probe_matches_dry_run() -> None:
    """D21 pin: the gate probe agrees with one direct ``_dry_run`` invocation.

    The probe's boolean must equal the recorded exit status of a real
    ``_dry_run("--all-groups", "--all-extras")`` — True only when that call
    exits 0, False when it exits nonzero — so the gate can never skip past a
    resolution that actually works, or run into one that cannot.
    """
    available = _offline_resolution_available()
    try:
        _dry_run("--all-groups", "--all-extras")
        dry_run_ok = True
    except AssertionError:
        dry_run_ok = False
    assert available == dry_run_ok


def test_root_all_extras_and_groups_do_not_select_torch() -> None:
    _require_offline_resolution()
    output = _dry_run("--all-groups", "--all-extras")
    assert "torch==" not in output
    assert "+cpu" not in output


def test_workspace_all_packages_selects_cuda_132_torch_not_cpu() -> None:
    _require_offline_resolution()
    output = _dry_run("--all-packages", "--all-groups", "--all-extras")
    assert "torch==2.12.0+cu132" in output
    assert "torchvision==0.27.0+cu132" in output
    assert "+cpu" not in output
