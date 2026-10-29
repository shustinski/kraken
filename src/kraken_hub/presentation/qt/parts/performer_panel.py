"""manager_app: performer panel."""
from __future__ import annotations
from kraken_manager.domain.identity import Performer
from kraken_hub.composition import EmbeddedProjectService

def _performer_panel(service: EmbeddedProjectService):
    from PyQt6.QtCore import QModelIndex, Qt
    from PyQt6.QtGui import QColor, QPainter
    from PyQt6.QtWidgets import (
        QColorDialog,
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QHBoxLayout,
        QLineEdit,
        QMenu,
        QMessageBox,
        QPushButton,
        QStyledItemDelegate,
        QStyleOptionViewItem,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    class ColorSwatchDelegate(QStyledItemDelegate):
        def paint(
            self,
            painter: QPainter | None,
            option: QStyleOptionViewItem,
            index: QModelIndex,
        ) -> None:
            super().paint(painter, option, index)
            if painter is None:
                return
            color = QColor(str(index.data(Qt.ItemDataRole.UserRole) or ""))
            if not color.isValid():
                return
            painter.save()
            swatch = option.rect.adjusted(8, 5, -8, -5)
            painter.setPen(QColor("#64748B"))
            painter.setBrush(color)
            painter.drawRoundedRect(swatch, 4, 4)
            painter.restore()

    host = QWidget()
    layout = QVBoxLayout(host)
    actions = QHBoxLayout()
    add = QPushButton("Добавить исполнителя")
    refresh_button = QPushButton("Обновить")
    actions.addWidget(add)
    actions.addWidget(refresh_button)
    actions.addStretch(1)
    layout.addLayout(actions)
    table = QTableWidget(0, 3)
    table.setHorizontalHeaderLabels(("Имя", "Цвет", "Учётная запись"))
    table.horizontalHeader().setStretchLastSection(True)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    table.setItemDelegateForColumn(1, ColorSwatchDelegate(table))
    layout.addWidget(table)
    performers: list[Performer] = []

    def refresh() -> None:
        nonlocal performers
        performers = list(service.list_performers())
        table.setRowCount(len(performers))
        for row, performer in enumerate(performers):
            name_item = QTableWidgetItem(performer.name)
            color_item = QTableWidgetItem()
            color_item.setData(Qt.ItemDataRole.UserRole, performer.color)
            color_item.setToolTip("Нажмите, чтобы изменить цвет")
            account_item = QTableWidgetItem("GitLab" if performer.principal_id is not None else "Ручной")
            table.setItem(row, 0, name_item)
            table.setItem(row, 1, color_item)
            table.setItem(row, 2, account_item)

    def performer_dialog(
        *,
        title: str,
        initial_name: str = "",
        initial_color: str = "#60A5FA",
    ) -> tuple[str, str] | None:
        dialog = QDialog(host)
        dialog.setWindowTitle(title)
        form = QFormLayout(dialog)
        name = QLineEdit(initial_name)
        selected_color = QColor(initial_color)
        color = QPushButton()
        color.setObjectName("performerColorPicker")
        color.setToolTip("Выбрать цвет исполнителя")

        def update_color_button() -> None:
            color_value = selected_color.name(QColor.NameFormat.HexRgb).upper()
            text_color = "#111827" if selected_color.lightnessF() > 0.65 else "#FFFFFF"
            color.setText(color_value)
            color.setStyleSheet(f"background-color: {color_value}; color: {text_color};")

        def choose_color() -> None:
            nonlocal selected_color
            chosen_color = QColorDialog.getColor(selected_color, dialog, "Цвет исполнителя")
            if not chosen_color.isValid():
                return
            selected_color = chosen_color
            update_color_button()

        color.clicked.connect(choose_color)
        update_color_button()
        form.addRow("Имя", name)
        form.addRow("Цвет", color)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return name.text(), selected_color.name(QColor.NameFormat.HexRgb).upper()

    def create() -> None:
        values = performer_dialog(title="Новый исполнитель")
        if values is None:
            return
        try:
            service.create_manual_performer(name=values[0], color=values[1])
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось создать исполнителя", str(exc))
            return
        refresh()

    def edit(row: int) -> None:
        if not 0 <= row < len(performers):
            return
        performer = performers[row]
        values = performer_dialog(
            title="Изменить исполнителя",
            initial_name=performer.name,
            initial_color=performer.color,
        )
        if values is None:
            return
        try:
            service.update_performer(
                performer_id=performer.id,
                name=values[0],
                color=values[1],
            )
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось изменить исполнителя", str(exc))
            return
        refresh()

    def change_color(row: int, column: int) -> None:
        if column != 1 or not 0 <= row < len(performers):
            return
        performer = performers[row]
        selected_color = QColorDialog.getColor(
            QColor(performer.color),
            host,
            "Цвет исполнителя",
        )
        if not selected_color.isValid():
            return
        try:
            service.update_performer(
                performer_id=performer.id,
                name=performer.name,
                color=selected_color.name(QColor.NameFormat.HexRgb).upper(),
            )
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось изменить цвет исполнителя", str(exc))
            return
        refresh()

    def delete(row: int) -> None:
        if not 0 <= row < len(performers):
            return
        performer = performers[row]
        answer = QMessageBox.question(
            host,
            "Удалить исполнителя",
            f"Удалить исполнителя «{performer.name}»?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            service.archive_performer(performer.id)
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось удалить исполнителя", str(exc))
            return
        refresh()

    def context_menu(position) -> None:
        row = table.rowAt(position.y())
        if row < 0:
            return
        table.selectRow(row)
        menu = QMenu(table)
        edit_action = menu.addAction("Изменить")
        delete_action = menu.addAction("Удалить")
        selected_action = menu.exec(table.viewport().mapToGlobal(position))
        if selected_action is edit_action:
            edit(row)
        elif selected_action is delete_action:
            delete(row)

    add.clicked.connect(create)
    refresh_button.clicked.connect(refresh)
    table.cellClicked.connect(change_color)
    table.customContextMenuRequested.connect(context_menu)
    refresh()
    return host
