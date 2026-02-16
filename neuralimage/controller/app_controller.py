import sys
import importlib
import os
from pathlib import Path


_DLL_DIR_HANDLES: list[object] = []

class AppController:

    def __init__(self, ui_only: bool = False):
        # Important on Windows: importing torch after QApplication may fail with WinError 1114
        # due to conflicting native DLL initialization order.
        self._torch_available = True
        if not ui_only:
            try:
                self._preload_torch()
            except RuntimeError as exc:
                self._torch_available = False
                ui_only = True
                print(
                    "[WARN] Torch initialization failed; starting in UI-only mode.\n"
                    f"{exc}\n"
                    "Tip: rebuild package with UPX disabled and verify VC++ runtime is installed."
                )

        from PyQt6.QtWidgets import QApplication

        self.app = QApplication(sys.argv)
        self.app.setOrganizationName('INME')
        self.app.setApplicationName('NeuralImage')
        self.app.setStyleSheet(load_qss_from_resource())
        # self.app.setStyle(QStyleFactory.create('cde'))
        self.main_window_presenter = None
        self.main_window = None

        if ui_only:
            self.main_window = self._build_ui_only_window()
            self.main_window.show()
        else:
            # Keep import lazy so UI-only mode can run without heavy business dependencies.
            from presenter import MainPresenter

            self.main_window_presenter = MainPresenter()

    def exec(self):
        return self.app.exec()

    def _build_ui_only_window(self):
        from view import MainView, SettingsPanel

        settings_panel = SettingsPanel()
        window = MainView(side_panel=settings_panel)

        # Wire internal view signals so base UI behavior works in standalone mode.
        settings_panel.connect_internal_signals()
        window.connect_internal_signals()
        return window

    @staticmethod
    def _preload_torch():
        if sys.platform.startswith('win'):
            _prepare_windows_dll_paths()
        try:
            importlib.import_module("torch")
        except OSError as exc:
            raise RuntimeError(
                "PyTorch failed to initialize native DLLs before Qt startup. "
                "Run with '--ui-only' or reinstall torch/runtime dependencies.\n"
                f"Original error: {exc}"
            ) from exc

def load_qss_from_resource() -> str:
    """Читает style.qss из Qt‑ресурса."""
    from PyQt6.QtCore import QFile, QIODevice, QTextStream

    qfile = QFile("_internal/resources/dark_modern.qss")
    # qfile = QFile("_internal/resources//style.qss")
    # qfile = QFile("_internal/resources//new_style.qss")
    # Открываем в режиме «только чтение» + «текстовый» (чтобы Qt
    # понимал, что это текст, а не бинарные данные)
    if not qfile.open(
            QIODevice.OpenModeFlag.ReadOnly | QIODevice.OpenModeFlag.Text
    ):
        raise IOError(f"Cannot open QSS file")

    stream = QTextStream(qfile)

    qss = stream.readAll()
    qfile.close()
    return qss


def _prepare_windows_dll_paths() -> None:
    # In frozen builds, torch native dependencies are under `_internal/torch/lib`.
    # Explicitly adding these directories makes DLL resolution deterministic.
    candidates: list[Path] = []
    exe_dir = Path(sys.executable).resolve().parent
    candidates.append(exe_dir / '_internal')
    candidates.append(exe_dir / '_internal' / 'torch' / 'lib')

    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        meipass_path = Path(meipass)
        candidates.append(meipass_path)
        candidates.append(meipass_path / 'torch' / 'lib')

    # Also include torch/lib from a normal python environment (non-frozen run).
    try:
        spec = importlib.util.find_spec('torch')
        if spec is not None and spec.origin is not None:
            torch_pkg = Path(spec.origin).resolve().parent
            candidates.append(torch_pkg / 'lib')
    except Exception:
        pass

    existing_dirs = []
    seen = set()
    for path in candidates:
        key = str(path).lower()
        if key in seen or not path.is_dir():
            continue
        seen.add(key)
        existing_dirs.append(path)

    if hasattr(os, 'add_dll_directory'):
        for path in existing_dirs:
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(str(path)))
            except OSError:
                continue

    current_path = os.environ.get('PATH', '')
    prepend = ';'.join(str(p) for p in existing_dirs)
    if prepend:
        os.environ['PATH'] = f'{prepend};{current_path}' if current_path else prepend
