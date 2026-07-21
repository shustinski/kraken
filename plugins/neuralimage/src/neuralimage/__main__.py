from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path


def _resolve_main() -> Callable[[], None]:
    if not __package__:
        # Support IDEs that execute this file directly instead of launching
        # the package with ``python -m neuralimage``.
        source_root = str(Path(__file__).resolve().parents[1])
        if source_root not in sys.path:
            sys.path.insert(0, source_root)

    from neuralimage.main import main

    return main

if __name__ == "__main__":
    _resolve_main()()
