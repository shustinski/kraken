"""cli: main."""
from __future__ import annotations
import os
import sys

from kraken_admin.cli_parts.execute import _execute

from kraken_admin.cli_parts.parser import _parser

def main() -> int:
    from kraken_core.process_lifetime import bind_child_process_lifetime

    bind_child_process_lifetime()
    args = _parser().parse_args()
    if args.command is None:
        from kraken_admin.admin_app import main as admin_window

        return admin_window()
    try:
        return _execute(args)
    except KeyboardInterrupt:
        print("\nОперация отменена.", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        if os.environ.get("KRAKEN_DEBUG") == "1":
            raise
        return 2
