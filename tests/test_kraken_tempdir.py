"""Regression tests: kraken_tempdir must propagate body exceptions.

The cleanup retry loop in the helper's ``finally`` used to ``return`` after a
successful removal. In a ``@contextmanager`` generator a ``return`` inside
``finally`` suppresses the in-flight body exception, so the context manager
reported success and test failures were silently hidden (surfacing later as
confusing errors such as ``UnboundLocalError`` after the ``with`` block).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_hub_embedded_service_parts.kraken_tempdir import kraken_tempdir


def test_body_exception_propagates_and_dir_is_removed() -> None:
    path: str | None = None
    with pytest.raises(ValueError, match="body failure"):
        with kraken_tempdir() as directory:
            path = directory
            assert Path(directory).is_dir()
            raise ValueError("body failure")
    assert path is not None
    assert not Path(path).exists(), "the temp dir must still be cleaned up"


def test_normal_use_still_cleans_up() -> None:
    with kraken_tempdir() as directory:
        assert Path(directory).is_dir()
    assert not Path(directory).exists()
