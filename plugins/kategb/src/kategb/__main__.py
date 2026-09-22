from __future__ import annotations

import multiprocessing as mp

from kategb.application.cli import main
from kraken_core.process_lifetime import bind_child_process_lifetime


if __name__ == "__main__":
    mp.freeze_support()
    bind_child_process_lifetime()
    main()
