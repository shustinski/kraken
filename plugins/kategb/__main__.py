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
    from kraken_core.process_lifetime import bind_child_process_lifetime

    bind_child_process_lifetime()
    from kategb.application.cli import main as kategb_main

    mp.freeze_support()
    kategb_main()


if __name__ == "__main__":
    main()
