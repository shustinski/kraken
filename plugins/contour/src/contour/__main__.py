from __future__ import annotations

import multiprocessing as mp

from .batch_processor import configure_batch_runtime
from .application.cli import main as contour_main

if __name__ == "__main__":
    mp.freeze_support()
    configure_batch_runtime()
    contour_main()
