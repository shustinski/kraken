import argparse
import multiprocessing as mp
import os
import sys
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
from neuralimage.lib.version import get_app_title


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
        '--version',
        action='version',
        version=get_app_title(),
    )
    parser.add_argument(
        '--kraken-job-manifest',
        default=None,
        help='Kraken Agent job manifest (managed headless mode).',
    )
    parser.add_argument(
        '--kraken-result-manifest',
        default=None,
        help='Kraken Agent result manifest (managed headless mode).',
    )
    parser.add_argument(
        '--kraken-staging-root',
        default=None,
        help='Kraken Agent staging workspace (managed headless mode).',
    )
    parser.add_argument(
        '--kraken-workspace-context',
        default=None,
        help='Kraken two-root local workspace context (interactive training).',
    )
    return parser


def _run_desktop_ui(*, ui_only: bool, workspace_session=None) -> None:
    from neuralimage.controller import AppController

    controller = AppController(ui_only=ui_only)
    if workspace_session is not None:
        workspace_session.attach(controller)
    controller.exec()


def main(argv: Sequence[str] | None = None) -> None:
    _configure_multiprocessing_start_method()
    parser = _build_parser()
    args = parser.parse_args(argv)
    from neuralimage.kraken_bridge import (
        NeuralImageWorkspaceSession,
        load_session_from_values,
    )

    kraken_session = load_session_from_values(
        job_manifest=args.kraken_job_manifest,
        result_manifest=args.kraken_result_manifest,
        staging_root=args.kraken_staging_root,
    )
    if kraken_session is not None:
        if args.kraken_workspace_context:
            parser.error('Agent and direct workspace modes cannot be combined.')
        if args.ui_only:
            parser.error('Kraken Agent mode is headless and cannot be combined with --ui-only.')
        kraken_session.run_headless()
        return
    workspace_session = (
        NeuralImageWorkspaceSession.load(args.kraken_workspace_context)
        if args.kraken_workspace_context
        else None
    )
    if args.ui_only and workspace_session is not None:
        parser.error('Kraken workspace training cannot be combined with --ui-only.')
    _run_desktop_ui(ui_only=args.ui_only, workspace_session=workspace_session)


if __name__ == '__main__':
    mp.freeze_support()
    main()
