import importlib

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-benchmarks",
        action="store_true",
        default=False,
        help="Run benchmark-marked tests.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-benchmarks"):
        return

    skip_benchmark = pytest.mark.skip(reason="need --run-benchmarks option to run")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)


def safe_import_or_skip(module_name: str):
    """
    Skip test module if optional dependency cannot be imported for any reason
    (missing package, broken DLLs, incompatible runtime).
    """
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Cannot import optional dependency '{module_name}': {exc}", allow_module_level=True)
