from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_source_launcher_runs_without_installation_or_pythonpath() -> None:
    plugin_root = Path(__file__).parents[1]
    launcher = plugin_root / 'run_neuralimage.py'
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)

    completed = subprocess.run(
        [sys.executable, str(launcher), '--version'],
        cwd=plugin_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert 'NeuralImage' in completed.stdout
