from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    plugin_root = Path(__file__).resolve().parent
    workspace_root = plugin_root.parent.parent
    for source_root in (workspace_root / "src", plugin_root / "src"):
        source_text = str(source_root)
        if source_root.exists() and source_text not in sys.path:
            sys.path.insert(0, source_text)
    from neuralimage.main import main as neuralimage_main

    return int(neuralimage_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
