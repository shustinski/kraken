"""Standalone Qt entrypoint for Karakal."""
from __future__ import annotations

import ctypes
import faulthandler
import logging
import multiprocessing as mp
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path


_FAULT_LOG_HANDLE = None
_LOGGER = logging.getLogger(__name__)


def ensure_package_parent_on_sys_path(module_file: str | Path, package_name: str = "karakal") -> Path | None:
    """Insert the package parent directory into ``sys.path`` when needed."""

    module_path = Path(module_file).resolve()
    package_name = str(package_name or "").strip()
    if not package_name:
        return None
    for ancestor in (module_path.parent,) + tuple(module_path.parents):
        nested_package = ancestor / package_name / "__init__.py"
        if nested_package.is_file():
            package_parent = ancestor
        elif ancestor.name == package_name and (ancestor / "__init__.py").is_file():
            package_parent = ancestor.parent
        else:
            continue
        package_parent_text = str(package_parent)
        if package_parent_text not in sys.path:
            sys.path.insert(0, package_parent_text)
        return package_parent
    return None


if __package__ in {None, ""}:
    ensure_package_parent_on_sys_path(__file__)
    from karakal.ui.app_icon import apply_karakal_icon
else:
    from ..ui.app_icon import apply_karakal_icon


def _set_windows_app_user_model_id(app_id: str = "kraken.karakal") -> None:
    """Set a stable Windows AppUserModelID so the taskbar uses the right icon."""

    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(str(app_id))
    except (AttributeError, OSError, TypeError, ValueError) as error:
        _LOGGER.debug("Could not set Windows AppUserModelID: %s", error)


def _install_crash_logging() -> Path | None:
    """Persist Python and native crash diagnostics for windowed builds."""

    global _FAULT_LOG_HANDLE
    try:
        local_app_data = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        log_path = local_app_data / "Karakal" / "karakal-crash.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _FAULT_LOG_HANDLE = log_path.open("a", encoding="utf-8", buffering=1)
        faulthandler.enable(file=_FAULT_LOG_HANDLE, all_threads=True)
    except (OSError, RuntimeError, ValueError) as error:
        _LOGGER.warning("Could not enable Karakal crash logging: %s", error)
        return None

    previous_hook = sys.excepthook

    def log_unhandled_exception(exc_type, exc_value, exc_traceback) -> None:
        try:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] Unhandled exception\n")
                handle.writelines(traceback.format_exception(exc_type, exc_value, exc_traceback))
        except OSError as error:
            _LOGGER.error("Could not append to Karakal crash log %s: %s", log_path, error)
        previous_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = log_unhandled_exception
    return log_path


def _load_main_window_class():
    if __package__ in {None, ""}:
        ensure_package_parent_on_sys_path(__file__)
        from karakal.app.main_window import KarakalMainWindow
    else:
        from ..app.main_window import KarakalMainWindow
    return KarakalMainWindow


def main() -> int:
    if any(arg == "--benchmark-comparison" for arg in sys.argv[1:]):
        if __package__ in {None, ""}:
            ensure_package_parent_on_sys_path(__file__)
            from karakal.comparison.benchmark import main as benchmark_main
        else:
            from ..comparison.benchmark import main as benchmark_main
        benchmark_args = [arg for arg in sys.argv[1:] if arg != "--benchmark-comparison"]
        return int(benchmark_main(benchmark_args))

    from PyQt6.QtWidgets import QApplication

    mp.freeze_support()
    _install_crash_logging()
    _set_windows_app_user_model_id()
    app = QApplication(sys.argv)
    app.setApplicationName("Karakal")
    app.setApplicationDisplayName("Karakal")
    apply_karakal_icon()
    window_class = _load_main_window_class()
    window = window_class()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
