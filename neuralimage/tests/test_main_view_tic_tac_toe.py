import pytest
from pathlib import Path
import uuid

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QWidget

from view.main_window import MainView


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_temp_dir() -> Path:
    base = Path(".codex_tmp_scratch")
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"ttt_view_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_main_view_opens_tic_tac_toe_dialog(qapp, monkeypatch):
    monkeypatch.setenv("NEURALIMAGE_SETTINGS_DIR", str(_make_temp_dir()))
    view = MainView(QWidget())
    view.connect_internal_signals()

    action = getattr(view, "open_tic_tac_toe_action", None)
    assert action is not None

    action.trigger()
    qapp.processEvents()

    dialog = getattr(view, "_tic_tac_toe_dialog", None)
    assert dialog is not None
    assert dialog.isVisible()


def test_main_view_exposes_validation_gradient_plugin_action(qapp, monkeypatch):
    monkeypatch.setenv("NEURALIMAGE_SETTINGS_DIR", str(_make_temp_dir()))
    view = MainView(QWidget())
    opened: list[str] = []
    view.open_validation_gradient_requested.connect(lambda: opened.append("opened"))
    view.connect_internal_signals()

    action = getattr(view, "_open_validation_gradient_action", None)
    assert action is not None

    action.trigger()
    qapp.processEvents()

    assert opened == ["opened"]
