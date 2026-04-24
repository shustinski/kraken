"""Allow `python -m Validation_gradient_widget_lite`."""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from neuralimage.Validation_gradient_widget_lite.debug.main import main
else:
    from .debug.main import main


if __name__ == "__main__":
    raise SystemExit(main())
