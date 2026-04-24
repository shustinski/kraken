from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_dist_path(root: Path | None = None) -> Path:
    base = project_root() if root is None else Path(root)
    return base / 'dist' / 'linux'


def default_work_path(root: Path | None = None) -> Path:
    base = project_root() if root is None else Path(root)
    return base / 'build' / 'linux'


def build_linux_pyinstaller_args(
    *,
    spec_path: Path,
    dist_path: Path,
    work_path: Path,
) -> list[str]:
    return [
        '--noconfirm',
        '--clean',
        '--distpath',
        str(dist_path),
        '--workpath',
        str(work_path),
        str(spec_path),
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Build Linux desktop bundle for NeuralImage via PyInstaller.',
    )
    parser.add_argument(
        '--with-webui',
        action='store_true',
        help='Include optional Django web UI dependencies and assets in the build.',
    )
    parser.add_argument(
        '--dist-path',
        type=Path,
        default=default_dist_path(),
        help='PyInstaller dist output directory. Default: dist/linux',
    )
    parser.add_argument(
        '--work-path',
        type=Path,
        default=default_work_path(),
        help='PyInstaller work output directory. Default: build/linux',
    )
    parser.add_argument(
        '--app-name',
        default='NeuralImage',
        help='Bundle name inside the Linux dist directory.',
    )
    parser.add_argument(
        '--spec',
        type=Path,
        default=project_root() / 'NeuralImage.spec',
        help='Path to the shared PyInstaller spec file.',
    )
    return parser


def _validate_linux_host() -> None:
    allow_cross_build = str(os.getenv('NEURALIMAGE_ALLOW_CROSS_BUILD', '') or '').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }
    if sys.platform.startswith('linux') or allow_cross_build:
        return
    raise SystemExit(
        'Linux build target must be executed on a Linux host. '
        'Set NEURALIMAGE_ALLOW_CROSS_BUILD=1 only if you intentionally know the host toolchain matches the target.',
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_linux_host()

    spec_path = Path(args.spec).resolve()
    dist_path = Path(args.dist_path).resolve()
    work_path = Path(args.work_path).resolve()
    dist_path.mkdir(parents=True, exist_ok=True)
    work_path.mkdir(parents=True, exist_ok=True)

    os.environ['NEURALIMAGE_BUILD_TARGET'] = 'linux'
    os.environ['NEURALIMAGE_APP_NAME'] = str(args.app_name or 'NeuralImage').strip() or 'NeuralImage'
    os.environ['NEURALIMAGE_INCLUDE_WEBUI'] = '1' if bool(args.with_webui) else '0'

    from PyInstaller.__main__ import run as pyinstaller_run

    pyinstaller_run(
        build_linux_pyinstaller_args(
            spec_path=spec_path,
            dist_path=dist_path,
            work_path=work_path,
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
