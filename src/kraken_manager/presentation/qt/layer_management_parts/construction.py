"""layer_management: construction."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QDialogButtonBox,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from kraken_manager.presentation.qt.layer_management_parts.format_history_payload import _format_history_payload

from kraken_manager.presentation.qt.layer_management_parts.format_scalar import (
    _display_value,
    _event_type_label,
    _local_timestamp,
)


class ConstructionMixin:
    def __init__(
        self,
        snapshot: ObjectPropertiesSnapshot,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.snapshot = snapshot
        self.setObjectName("objectPropertiesDialog")
        self.setWindowTitle(f"Свойства: {snapshot.title}")
        self.setModal(True)
        self.resize(880, 620)

        tabs = QTabWidget(self)
        tabs.setObjectName("objectPropertiesTabs")
        self.properties_table = self._table(("Свойство", "Значение"))
        self.properties_table.setObjectName("objectPropertiesTable")
        self.properties_table.setRowCount(len(snapshot.properties))
        for row, (name, value) in enumerate(snapshot.properties):
            self.properties_table.setItem(row, 0, QTableWidgetItem(str(name)))
            value_item = QTableWidgetItem(_display_value(value))
            value_item.setToolTip(value_item.text())
            self.properties_table.setItem(row, 1, value_item)
        self.properties_table.resizeRowsToContents()
        tabs.addTab(self.properties_table, "Свойства")

        self.history_table = self._table(
            ("Дата и время", "Пользователь", "Событие", "Подробности")
        )
        self.history_table.setObjectName("objectHistoryTable")
        self.history_table.setRowCount(len(snapshot.history))
        for row, entry in enumerate(snapshot.history):
            timestamp = QTableWidgetItem(_local_timestamp(entry.recorded_at))
            timestamp.setToolTip(entry.recorded_at)
            self.history_table.setItem(row, 0, timestamp)
            self.history_table.setItem(row, 1, QTableWidgetItem(entry.actor or "—"))
            event_item = QTableWidgetItem(_event_type_label(entry.event_type))
            event_item.setToolTip(entry.event_type)
            self.history_table.setItem(row, 2, event_item)
            summary = _format_history_payload(entry.event_type, entry.payload)
            details = QTableWidgetItem(summary)
            details.setToolTip(_display_value(entry.payload))
            self.history_table.setItem(row, 3, details)
        self.history_table.resizeRowsToContents()
        if snapshot.temporal_loader is not None:
            self.history_table.setToolTip(
                "Дважды щёлкните событие, чтобы открыть состояние объекта на этот момент"
            )
            self.history_table.cellDoubleClicked.connect(
                self._open_temporal_state
            )
        tabs.addTab(self.history_table, f"История изменений ({len(snapshot.history)})")
        self.notes_table = self._mapping_table(
            snapshot.notes,
            (
                ("revision", "Ревизия"),
                ("body", "Текст"),
                ("author", "Автор"),
                ("recorded_at", "Дата и время"),
            ),
        )
        self.notes_table.setObjectName("objectNotesTable")
        tabs.addTab(
            self._action_tab(
                self.notes_table,
                snapshot.notes,
                (),
                general_actions=snapshot.actions,
            ),
            f"Заметки ({len(snapshot.notes)})",
        )
        self.files_table = self._mapping_table(
            snapshot.files,
            (
                ("name", "Название"),
                ("scope", "Тип"),
                ("active_version", "Активная версия"),
                ("archived", "Архив"),
            ),
        )
        self.files_table.setObjectName("objectFilesTable")
        tabs.addTab(
            self._action_tab(
                self.files_table,
                snapshot.files,
                snapshot.file_actions,
            ),
            f"Файлы ({len(snapshot.files)})",
        )
        self.versions_table = self._mapping_table(
            snapshot.versions,
            (
                ("filename", "Имя"),
                ("size", "Размер"),
                ("sha256", "SHA-256"),
                ("author", "Автор"),
                ("created_at", "Дата"),
                ("parent", "Родитель"),
                ("tool", "Инструмент"),
                ("provenance", "Provenance"),
                ("active", "Активная"),
            ),
        )
        self.versions_table.setObjectName("objectVersionsTable")
        tabs.addTab(
            self._action_tab(
                self.versions_table,
                snapshot.versions,
                snapshot.version_actions,
            ),
            f"Версии ({len(snapshot.versions)})",
        )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("Закрыть")
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs, 1)
        layout.addWidget(buttons)
