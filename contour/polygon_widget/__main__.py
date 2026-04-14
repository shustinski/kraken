from __future__ import annotations

import multiprocessing as mp

from .application.cli import main


if __name__ == "__main__":
    mp.freeze_support()
    main()
