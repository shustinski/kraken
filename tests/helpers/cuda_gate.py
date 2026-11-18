"""Shared CUDA precondition probe (D15.1).

On machines whose torch CUDA init path crashes the whole process (Windows
STATUS_ACCESS_VIOLATION), the question "is CUDA usable?" must not be asked
in-process: the call kills the interpreter instead of returning. Probe it in a
disposable subprocess first. ``cuda_precondition_holds()`` returning True means
"torch answered under ``CUDA_VISIBLE_DEVICES=-1`` and no CUDA device is
usable" (the bounded-CPU test may run); a crash, timeout or non-zero exit
means it must skip.
"""
from __future__ import annotations

import os
import subprocess
import sys

CUDA_PROBE_TIMEOUT_SECONDS = 300


def cuda_precondition_holds() -> bool:
    probe = "import sys, torch; sys.exit(0 if not torch.cuda.is_available() else 1)"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            timeout=CUDA_PROBE_TIMEOUT_SECONDS,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0
