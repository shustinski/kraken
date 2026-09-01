from __future__ import annotations

from PyQt6.QtCore import QSettings

import karakal.app.main_window as main_window


def test_widget_exposes_only_supported_application_modes(tmp_path, monkeypatch, qtbot) -> None:
    settings = QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(main_window, "QSettings", lambda *_args: settings)
    widget = main_window.KarakalWidget()
    qtbot.addWidget(widget)

    modes = {str(widget.app_mode_combo.itemData(index)) for index in range(widget.app_mode_combo.count())}
    menu_modes = {str(action.data()) for action in widget._mode_menu.actions()}

    assert modes == {"validation", "grid_inspection"}
    assert menu_modes == modes
    assert widget.main_mode_stack.count() == 2
    assert not any("manager" in action.text().lower() for action in widget.findChildren(main_window.QAction))
