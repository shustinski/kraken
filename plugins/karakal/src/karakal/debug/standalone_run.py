"""Standalone Qt entrypoint for Karakal."""
from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path


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


def _load_main_window_class():
    if __package__ in {None, ""}:
        ensure_package_parent_on_sys_path(__file__)
        from karakal.app.main_window import KarakalMainWindow
    else:
        from ..app.main_window import KarakalMainWindow
    return KarakalMainWindow


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    window_class = _load_main_window_class()
    mp.freeze_support()
    app = QApplication(sys.argv)
    window = window_class()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
