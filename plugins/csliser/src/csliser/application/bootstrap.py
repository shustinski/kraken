from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass

from PyQt6.QtWidgets import QApplication
from kraken_core.qt import configure_application_identity
from kraken_core.styles import load_shared_stylesheet

from csliser import __version__
from csliser.presentation.qt.window import CSliserWindow


@dataclass(slots=True)
class CSliserApplicationComponents:
    app: QApplication
    window: CSliserWindow


def build_application(argv: Sequence[str] | None = None, *, apply_qss: bool = True) -> CSliserApplicationComponents:
    qt_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    app = QApplication.instance() or QApplication(qt_argv)
    assert isinstance(app, QApplication)
    app.setOrganizationName("Kraken")
    app.setApplicationName("CSliser")
    app.setApplicationVersion(__version__)
    configure_application_identity(app, app_id="Krarken.CSliser", icon_name="csliser")
    if apply_qss:
        app.setStyleSheet(load_shared_stylesheet("dark_modern.qss"))
    window = CSliserWindow()
    return CSliserApplicationComponents(app=app, window=window)
