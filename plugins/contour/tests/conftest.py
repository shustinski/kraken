"""Shared pytest fixtures for the ViaLaNet Polygon Widget test suite."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session", autouse=True)
def _isolated_settings_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    previous = os.environ.get("VIALANET_SETTINGS_DIR")
    os.environ["VIALANET_SETTINGS_DIR"] = str(tmp_path_factory.mktemp("contour-settings"))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("VIALANET_SETTINGS_DIR", None)
        else:
            os.environ["VIALANET_SETTINGS_DIR"] = previous


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
