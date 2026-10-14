"""test_hub_manager_app: test performer table supports."""
from __future__ import annotations
from types import SimpleNamespace
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog,
    QDialog,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
)
from kraken_hub.manager_app import (
    _performer_panel,
)

def test_performer_dialog_uses_color_picker(qapp, monkeypatch) -> None:
    created: list[dict[str, str]] = []

    class Service:
        @staticmethod
        def list_performers():
            return ()

        @staticmethod
        def create_manual_performer(**kwargs):
            created.append(kwargs)

    monkeypatch.setattr(
        QColorDialog,
        "getColor",
        lambda *_args: QColor("#123ABC"),
    )

    def complete_dialog(dialog: QDialog):
        name = dialog.findChild(QLineEdit)
        color = dialog.findChild(QPushButton, "performerColorPicker")
        assert name is not None
        assert color is not None
        assert color.text() == "#60A5FA"
        name.setText("Reviewer")
        color.click()
        assert color.text() == "#123ABC"
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", complete_dialog)
    panel = _performer_panel(Service())
    add_button = panel.findChild(QPushButton)
    assert add_button is not None
    add_button.click()

    assert created == [{"name": "Reviewer", "color": "#123ABC"}]

def test_performer_table_supports_color_editing_and_context_actions(qapp, monkeypatch) -> None:
    performers = [
        SimpleNamespace(
            id="performer-1",
            name="Reviewer",
            color="#60A5FA",
            principal_id=None,
        )
    ]
    updates: list[dict[str, str]] = []
    archived: list[str] = []

    class Service:
        @staticmethod
        def list_performers():
            return tuple(performers)

        @staticmethod
        def update_performer(**kwargs):
            updates.append(kwargs)
            current = performers[0]
            performers[0] = SimpleNamespace(
                id=current.id,
                name=kwargs["name"],
                color=kwargs["color"],
                principal_id=current.principal_id,
            )

        @staticmethod
        def archive_performer(performer_id):
            archived.append(performer_id)
            performers.clear()

    monkeypatch.setattr(QColorDialog, "getColor", lambda *_args: QColor("#123ABC"))
    panel = _performer_panel(Service())
    table = panel.findChild(QTableWidget)
    assert table is not None
    panel.show()
    qapp.processEvents()
    assert table.item(0, 1).text() == ""
    assert table.item(0, 1).data(Qt.ItemDataRole.UserRole) == "#60A5FA"
    color_center = table.visualItemRect(table.item(0, 1)).center()
    assert table.viewport().grab().toImage().pixelColor(color_center).name().upper() == "#60A5FA"

    table.cellClicked.emit(0, 1)
    assert updates[-1] == {
        "performer_id": "performer-1",
        "name": "Reviewer",
        "color": "#123ABC",
    }
    assert table.item(0, 1).data(Qt.ItemDataRole.UserRole) == "#123ABC"

    def complete_edit(dialog: QDialog):
        name = dialog.findChild(QLineEdit)
        assert name is not None
        name.setText("Edited reviewer")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", complete_edit)
    requested_action = {"text": "Изменить"}
    monkeypatch.setattr(
        QMenu,
        "exec",
        lambda menu, *_args: next(action for action in menu.actions() if action.text() == requested_action["text"]),
    )
    qapp.processEvents()
    position = table.visualItemRect(table.item(0, 0)).center()
    table.customContextMenuRequested.emit(position)
    assert updates[-1]["name"] == "Edited reviewer"

    requested_action["text"] = "Удалить"
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)
    table.customContextMenuRequested.emit(position)
    assert archived == ["performer-1"]
    assert table.rowCount() == 0
