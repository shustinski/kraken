import argparse
import importlib
import multiprocessing as mp
import os
import platform
from importlib.util import find_spec
from typing import Sequence


# def _preload_windows_torch_dll() -> None:
#     if platform.system() != 'Windows':
#         return
#     try:
#         import ctypes

#         torch_spec = find_spec('torch')
#         if torch_spec is None or torch_spec.origin is None:
#             return
#         dll_path = os.path.join(os.path.dirname(torch_spec.origin), 'lib', 'c10.dll')
#         if os.path.exists(dll_path):
#             ctypes.CDLL(os.path.normpath(dll_path))
#     except Exception:
#         pass


# _preload_windows_torch_dll()

from controller import AppController
from lib.version import get_app_title


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ui-only',
        action='store_true',
        help='Run only the UI layer without presenter/business logic.',
    )
    parser.add_argument(
        '--web',
        action='store_true',
        help='Run Django web UI instead of Qt UI.',
    )
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host for Django web UI. Default: 127.0.0.1',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port for Django web UI. Default: 8000',
    )
    parser.add_argument(
        '--version',
        action='version',
        version=get_app_title(),
    )
    return parser


def _run_web_ui(host: str, port: int) -> None:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'webui_project.settings')
    try:
        execute_from_command_line = importlib.import_module('django.core.management').execute_from_command_line
    except ImportError as exc:
        raise RuntimeError('Django is not installed. Install requirements-dev.txt first.') from exc
    execute_from_command_line(['manage.py', 'runserver', f'{host}:{port}'])


def _run_desktop_ui(*, ui_only: bool) -> None:
    controller = AppController(ui_only=ui_only)
    controller.exec()


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.web:
        _run_web_ui(args.host, args.port)
        return
    _run_desktop_ui(ui_only=args.ui_only)


if __name__ == '__main__':
    mp.freeze_support()
    main()
