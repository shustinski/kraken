"""Provide the standalone Qt application entrypoint for the lite mismatch-only widget."""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from Validation_gradient_widget_lite.app.main_window import ValidationGradientLiteMainWindow
else:
    from ..app.main_window import ValidationGradientLiteMainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = ValidationGradientLiteMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
