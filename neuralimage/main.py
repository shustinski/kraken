import argparse
import multiprocessing as mp
import os
import importlib

from controller import AppController


def _run_web_ui(host: str, port: int) -> None:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'webui_project.settings')
    try:
        execute_from_command_line = importlib.import_module('django.core.management').execute_from_command_line
    except ImportError as exc:
        raise RuntimeError('Django is not installed. Install requirements-dev.txt first.') from exc

    execute_from_command_line(['manage.py', 'runserver', f'{host}:{port}'])

def main(argv=None):
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
    args = parser.parse_args(argv)

    if args.web:
        _run_web_ui(args.host, args.port)
        return

    controller = AppController(ui_only=args.ui_only)
    controller.exec()

if __name__ == '__main__':
    mp.freeze_support()
    main()
