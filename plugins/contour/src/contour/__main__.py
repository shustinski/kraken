from __future__ import annotations

import multiprocessing as mp

from .application.cli import main as contour_main

if __name__ == "__main__":
    mp.freeze_support()
    contour_main()
