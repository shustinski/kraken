from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass

from PyQt6.QtWidgets import QApplication
from kraken_core.qt import configure_application_identity
from kraken_core.styles import load_shared_stylesheet

from cartograph import __version__
from cartograph.presentation.qt.window import CartographWindow


@dataclass(slots=True)
class CartographApplicationComponents:
    app: QApplication
    window: CartographWindow


def build_application(argv: Sequence[str] | None = None, *, apply_qss: bool = True) -> CartographApplicationComponents:
    qt_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    app = QApplication.instance() or QApplication(qt_argv)
    assert isinstance(app, QApplication)
    app.setOrganizationName("Kraken")
    app.setApplicationName("Cartograph")
    app.setApplicationVersion(__version__)
    configure_application_identity(app, app_id="Kraken.Cartograph", icon_name="cartograph")
    if apply_qss:
        app.setStyleSheet(load_shared_stylesheet("dark_modern.qss"))
    return CartographApplicationComponents(app=app, window=CartographWindow())
