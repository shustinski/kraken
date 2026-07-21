from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_package_main_file_can_be_executed_directly() -> None:
    entrypoint = Path(__file__).parents[1] / 'src' / 'neuralimage' / '__main__.py'

    completed = subprocess.run(
        [sys.executable, str(entrypoint), '--version'],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert 'NeuralImage' in completed.stdout
