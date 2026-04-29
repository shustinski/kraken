from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    args = list(argv) if argv is not None else sys.argv[1:]
    if any(item in {"-h", "--help"} for item in args):
        parser = argparse.ArgumentParser(prog="kategb", description="Standalone launcher for KateGB.")
        parser.add_argument("--no-qss", action="store_true", help="Do not apply the shared Kraken stylesheet.")
        parser.print_help()
        return

    from kategb.application.bootstrap import build_application

    components = build_application(args, apply_qss="--no-qss" not in args)
    components.window.show()
    components.app.exec()
