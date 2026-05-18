from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path


def _add_source_paths() -> None:
    plugin_root = Path(__file__).resolve().parent
    workspace_root = plugin_root.parent.parent
    for path in (workspace_root / "src", plugin_root / "src"):
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)


def main() -> None:
    _add_source_paths()
    from krona.__main__ import main as krona_main

    mp.freeze_support()
    krona_main()


if __name__ == "__main__":
    main()
