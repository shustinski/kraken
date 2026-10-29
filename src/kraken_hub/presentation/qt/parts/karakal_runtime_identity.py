"""manager_app: karakal runtime identity."""
from __future__ import annotations
import os
import platform
from importlib.metadata import PackageNotFoundError, version
from kraken_core.analysis_run_protocol import (
    AnalysisRuntimeIdentity,
)

def _installed_version(distribution: str, fallback: str = "not-installed") -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return fallback

def _karakal_runtime_identity() -> AnalysisRuntimeIdentity:
    return AnalysisRuntimeIdentity(
        engine_version=_installed_version("karakal", "1.0.0"),
        engine_build=os.environ.get("KARAKAL_BUILD_HASH", "development"),
        python_version=platform.python_version(),
        numpy_version=_installed_version("numpy"),
        opencv_version=_installed_version("opencv-python", _installed_version("opencv-python-headless")),
        operating_system=f"{platform.system()} {platform.release()}",
    )
