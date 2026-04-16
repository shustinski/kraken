import argparse
import importlib
import multiprocessing as mp
import os
import platform
import sys
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
from lib.version import get_app_title


_STD_STREAM_FALLBACKS: list[object] = []


def _ensure_standard_streams() -> None:
    for stream_name, original_name in (('stdout', '__stdout__'), ('stderr', '__stderr__')):
        stream = getattr(sys, stream_name, None)
        if stream is not None:
            continue
        original_stream = getattr(sys, original_name, None)
        if original_stream is not None:
            setattr(sys, stream_name, original_stream)
            continue
        fallback_stream = open(os.devnull, 'w', encoding='utf-8', buffering=1)
        _STD_STREAM_FALLBACKS.append(fallback_stream)
        setattr(sys, stream_name, fallback_stream)


_ensure_standard_streams()


def _configure_multiprocessing_start_method() -> str | None:
    override = str(os.getenv('NEURALIMAGE_MP_START_METHOD', '') or '').strip().lower()
    if override:
        requested_method = override
    elif sys.platform.startswith('linux'):
        # Linux defaults to "fork", which is fragile once Qt and CUDA-enabled
        # torch objects exist in the parent process. Prefer spawn for desktop
        # runtime stability and parity with Windows behavior.
        requested_method = 'spawn'
    else:
        requested_method = ''
    if not requested_method:
        return mp.get_start_method(allow_none=True)

    current_method = mp.get_start_method(allow_none=True)
    if current_method is None:
        mp.set_start_method(requested_method, force=False)
        return requested_method
    return current_method


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
        management = importlib.import_module('django.core.management')
    except ImportError as exc:
        raise RuntimeError('Django is not installed. Install requirements-dev.txt first.') from exc
    management.execute_from_command_line(['manage.py', 'migrate', '--noinput'])
    management.execute_from_command_line(['manage.py', 'runserver', f'{host}:{port}', '--noreload'])


def _run_desktop_ui(*, ui_only: bool) -> None:
    from controller import AppController

    controller = AppController(ui_only=ui_only)
    controller.exec()


def main(argv: Sequence[str] | None = None) -> None:
    _configure_multiprocessing_start_method()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.web:
        _run_web_ui(args.host, args.port)
        return
    _run_desktop_ui(ui_only=args.ui_only)


if __name__ == '__main__':
    mp.freeze_support()
    main()
