"""Standalone Qt entrypoint for the lite widget."""
from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from neuralimage.Validation_gradient_widget_lite.app.main_window import ValidationGradientLiteMainWindow
else:
    from ..app.main_window import ValidationGradientLiteMainWindow


def main() -> int:
    from kraken_core.process_lifetime import bind_child_process_lifetime

    bind_child_process_lifetime()
    mp.freeze_support()
    app = QApplication(sys.argv)
    window = ValidationGradientLiteMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
