from __future__ import annotations

import multiprocessing as mp
from typing import Sequence

from .bootstrap import build_application


def main(argv: Sequence[str] | None = None) -> None:
    app, window = build_application(argv)
    window.show()
    app.exec()


if __name__ == "__main__":
    mp.freeze_support()
    main()
