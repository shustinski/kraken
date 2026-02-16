import importlib

import pytest


def safe_import_or_skip(module_name: str):
    """
    Skip test module if optional dependency cannot be imported for any reason
    (missing package, broken DLLs, incompatible runtime).
    """
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Cannot import optional dependency '{module_name}': {exc}", allow_module_level=True)
