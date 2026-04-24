import os
import sys
from pathlib import Path


def _set_workdir_to_executable_dir() -> None:
    if not getattr(sys, 'frozen', False):
        return
    try:
        exe_dir = Path(sys.executable).resolve().parent
        os.chdir(exe_dir)
    except Exception:
        # Best-effort hook; startup should not fail because of CWD setup.
        pass


_set_workdir_to_executable_dir()
