"""Run NeuralImage directly from a source checkout.

Open this file in VS Code and press F5. Installing the package or configuring
PYTHONPATH is not required.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


def _load_application() -> Callable[[Sequence[str] | None], None]:
    source_root = Path(__file__).resolve().parent / 'src'
    source_root_text = str(source_root)
    if source_root_text not in sys.path:
        sys.path.insert(0, source_root_text)

    from neuralimage.main import main

    return main


def main(argv: Sequence[str] | None = None) -> None:
    _load_application()(argv)


if __name__ == '__main__':
    mp.freeze_support()
    main()
