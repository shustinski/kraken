"""Shared pytest fixtures for the ViaLaNet Polygon Widget test suite."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session", autouse=True)
def _qt_application() -> Iterator[object]:
    """Provide a single ``QApplication`` instance for the entire test session.

    Many widget tests instantiate Qt objects directly. Creating a single
    application up-front avoids ``QWidget: Must construct a QApplication``
    errors and keeps tests fast.
    """
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app
    app.processEvents()
