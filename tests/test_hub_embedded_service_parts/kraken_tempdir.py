"""Temporary-directory cleanup that survives reserved DOS names."""
from __future__ import annotations

import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from kraken_manager.infrastructure.workspace_files import _remove_tree


@contextmanager
def kraken_tempdir() -> Iterator[str]:
    """Like ``tempfile.TemporaryDirectory()``, but reserved-name-safe on exit.

    ``TemporaryDirectory`` cleans up with ``shutil.rmtree``, which cannot remove
    a Kraken project tree on Windows: the source root contains a real ``aux``
    category directory, and ``rmtree`` fails with WinError 145 (reserved DOS
    device name). Cleanup goes through the same reserved-name-safe
    ``_remove_tree`` the product code uses (defect D10).

    The yielded path is fully resolved: when ``TEMP`` is an 8.3 short path
    (Windows), the raw ``mkdtemp`` path would no longer compare equal to the
    ``Path.resolve()``-normalised paths the service reports. Yielding the
    canonical form keeps harness-built paths comparable with them.
    """
    directory = Path(tempfile.mkdtemp()).resolve()
    try:
        yield str(directory)
    finally:
        # Fresh trees can be briefly held by the OS/antivirus, which surfaces
        # as transient WinError 145 on directory removal; retry before failing.
        for attempt in range(3):
            try:
                _remove_tree(directory)
                return
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(0.1 * (attempt + 1))
