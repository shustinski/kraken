"""Regression: root uv sync does not install NeuralImage's CUDA PyTorch extra."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

kraken_defect = "D17"


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


def test_root_all_extras_and_groups_do_not_select_torch() -> None:
    output = _dry_run("--all-groups", "--all-extras")
    assert "torch==" not in output
    assert "+cpu" not in output


def test_workspace_all_packages_selects_cuda_132_torch_not_cpu() -> None:
    output = _dry_run("--all-packages", "--all-groups", "--all-extras")
    assert "torch==2.12.0+cu132" in output
    assert "torchvision==0.27.0+cu132" in output
    assert "+cpu" not in output
