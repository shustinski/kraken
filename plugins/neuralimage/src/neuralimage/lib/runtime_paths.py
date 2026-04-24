from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    # Source layout: plugins/neuralimage/src/neuralimage/lib/runtime_paths.py
    return Path(__file__).resolve().parents[3]


def runtime_root() -> Path:
    if not bool(getattr(sys, 'frozen', False)):
        return project_root()

    executable_dir = Path(sys.executable).resolve().parent
    internal_dir = executable_dir / '_internal'
    if internal_dir.exists():
        return internal_dir

    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        return Path(meipass)

    return executable_dir


def internal_root() -> Path:
    if bool(getattr(sys, 'frozen', False)):
        return runtime_root()
    return project_root() / 'resources' / 'internal'


def resources_root() -> Path:
    return runtime_root() / 'resources'


def resolve_internal_path(*parts: str) -> Path:
    return internal_root().joinpath(*parts)


def resolve_resource_path(*parts: str) -> Path:
    return resources_root().joinpath(*parts)
