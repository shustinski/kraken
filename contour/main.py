from __future__ import annotations

import multiprocessing as mp

from polygon_widget.application.cli import main


if __name__ == "__main__":
    mp.freeze_support()
    main()
