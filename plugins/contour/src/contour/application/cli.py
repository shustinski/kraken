from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    args = list(argv) if argv is not None else sys.argv[1:]
    if any(item in {"-h", "--help"} for item in args):
        parser = argparse.ArgumentParser(prog="contour", description="Standalone launcher for Contour.")
        parser.add_argument("paths", nargs="*", help="Optional image files or a single directory to load on startup.")
        parser.add_argument("--input-dir", help="Input image directory.")
        parser.add_argument("--output-dir", help="Output directory for exported results.")
        parser.add_argument("--cif-dir", help="Directory with CIF overlays.")
        parser.add_argument("--pipeline-json", help="Path to pipeline JSON config.")
        parser.add_argument("--language", choices=("ru", "en"), default="ru", help="UI language override.")
        parser.add_argument("--width", type=int, default=1680, help="Initial window width.")
        parser.add_argument("--height", type=int, default=980, help="Initial window height.")
        parser.add_argument("--no-qss", action="store_true", help="Do not apply the main application QSS theme.")
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
        parser.add_argument("--log-file", default=None, help="Path to the log file.")
        parser.print_help()
        return

    from .bootstrap import build_application

    app, window = build_application(args)
    window.show()
    app.exec()


if __name__ == "__main__":
    mp.freeze_support()
    main()
