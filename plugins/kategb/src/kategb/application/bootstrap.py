from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass

from PyQt6.QtWidgets import QApplication
from kraken_core.qt import configure_application_identity
from kraken_core.styles import load_shared_stylesheet

from kategb import __version__
from kategb.presentation.qt.window import KateGBWindow


@dataclass(slots=True)
class KateGBApplicationComponents:
    app: QApplication
    window: KateGBWindow


def build_application(argv: Sequence[str] | None = None, *, apply_qss: bool = True) -> KateGBApplicationComponents:
    qt_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    app = QApplication.instance() or QApplication(qt_argv)
    assert isinstance(app, QApplication)
    app.setOrganizationName("ViaLaNet")
    app.setApplicationName("KateGB")
    app.setApplicationVersion(__version__)
    configure_application_identity(app, app_id="ViaLaNet.KateGB", icon_name="kategb")
    if apply_qss:
        app.setStyleSheet(load_shared_stylesheet("dark_modern.qss"))
    return KateGBApplicationComponents(app=app, window=KateGBWindow())
