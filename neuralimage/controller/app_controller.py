import sys
import importlib
import os
from pathlib import Path

from lib.shared_styles import load_shared_stylesheet


_DLL_DIR_HANDLES: list[object] = []


def format_backend_unavailable_message(error: Exception | str) -> str:
    technical_reason = str(error).strip() or 'Unknown backend initialization error.'
    return (
        "РќРµ СѓРґР°Р»РѕСЃСЊ РёРЅРёС†РёР°Р»РёР·РёСЂРѕРІР°С‚СЊ РІС‹С‡РёСЃР»РёС‚РµР»СЊРЅС‹Р№ backend PyTorch.\n\n"
        "РџСЂРёР»РѕР¶РµРЅРёРµ Р·Р°РїСѓС‰РµРЅРѕ С‚РѕР»СЊРєРѕ РІ СЂРµР¶РёРјРµ РёРЅС‚РµСЂС„РµР№СЃР°, РїРѕСЌС‚РѕРјСѓ РєРЅРѕРїРєР° 'Р—Р°РїСѓСЃРє' РЅРµРґРѕСЃС‚СѓРїРЅР°.\n\n"
        "РџСЂРѕРІРµСЂСЊС‚Рµ РЅР°Р»РёС‡РёРµ Microsoft Visual C++ Redistributable, СЃРѕРІРјРµСЃС‚РёРјРѕСЃС‚СЊ torch/РґСЂР°Р№РІРµСЂРѕРІ "
        "Рё С†РµР»РѕСЃС‚РЅРѕСЃС‚СЊ С„Р°Р№Р»РѕРІ РїСЂРёР»РѕР¶РµРЅРёСЏ.\n\n"
        f"РўРµС…РЅРёС‡РµСЃРєР°СЏ РїСЂРёС‡РёРЅР°: {technical_reason}"
    )


def apply_backend_unavailable_ui_state(window, message: str) -> None:
    title_getter = getattr(window, 'windowTitle', None)
    title_setter = getattr(window, 'setWindowTitle', None)
    if callable(title_getter) and callable(title_setter):
        current_title = str(title_getter() or '').strip()
        if current_title and '[UI only]' not in current_title:
            title_setter(f'{current_title} [UI only]')

    start_button = getattr(window, 'btn_start', None)
    log_signal = getattr(window, 'log_message', None)
    can_log = log_signal is not None and hasattr(log_signal, 'emit')
    if can_log:
        log_signal.emit(f'Р—Р°РїСѓСЃРє РЅРµРґРѕСЃС‚СѓРїРµРЅ: {message}')

    warning_signal = getattr(window, 'show_warning', None)
    can_warn = warning_signal is not None and hasattr(warning_signal, 'emit')
    if can_warn:
        warning_signal.emit(message)

    if start_button is not None:
        if hasattr(window, '_set_start_enabled'):
            window._set_start_enabled(False)
        if hasattr(start_button, 'setEnabled'):
            start_button.setEnabled(True)
        if hasattr(start_button, 'setToolTip'):
            start_button.setToolTip(message)
        clicked_signal = getattr(start_button, 'clicked', None)
        if (
            hasattr(clicked_signal, 'connect')
            and not bool(getattr(window, '_backend_unavailable_click_handler_installed', False))
        ):
            def _show_backend_unavailable_reason() -> None:
                if can_warn:
                    warning_signal.emit(message)
                if can_log:
                    log_signal.emit(f'Р—Р°РїСѓСЃРє РЅРµРґРѕСЃС‚СѓРїРµРЅ: {message}')

            clicked_signal.connect(_show_backend_unavailable_reason)
            setattr(window, '_backend_unavailable_click_handler_installed', True)


class AppController:

    def __init__(self, ui_only: bool = False):
        # Important on Windows: importing torch after QApplication may fail with WinError 1114
        # due to conflicting native DLL initialization order.
        self._torch_available = True
        self._backend_unavailable_message: str | None = None
        if not ui_only:
            try:
                self._preload_torch()
            except RuntimeError as exc:
                self._torch_available = False
                self._backend_unavailable_message = format_backend_unavailable_message(exc)
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
            if self._backend_unavailable_message:
                self._announce_backend_unavailable(self.main_window, self._backend_unavailable_message)
        else:
            # Keep import lazy so UI-only mode can run without heavy business dependencies.
            from bootstrap.composition_root import create_main_presenter

            self.main_window_presenter = create_main_presenter()

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
    def _announce_backend_unavailable(window, message: str) -> None:
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(0, lambda: apply_backend_unavailable_ui_state(window, message))

    @staticmethod
    def _preload_torch():
        if sys.platform.startswith('win'):
            _prepare_windows_dll_paths()
        try:
            importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyTorch is not installed. "
                "Run with '--ui-only' or install project dependencies first.\n"
                f"Original error: {exc}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                "PyTorch failed to initialize native DLLs before Qt startup. "
                "Run with '--ui-only' or reinstall torch/runtime dependencies.\n"
                f"Original error: {exc}"
            ) from exc


def load_qss_from_resource() -> str:
    return load_shared_stylesheet("dark_modern.qss")


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
