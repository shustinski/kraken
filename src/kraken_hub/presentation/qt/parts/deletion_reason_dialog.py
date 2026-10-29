"""manager_app: deletion reason dialog."""
from __future__ import annotations
from kraken_core.plugins import PluginInventoryItem

def _deletion_reason_dialog(parent, project_name: str) -> str | None:
    from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QVBoxLayout

    dialog = QDialog(parent)
    dialog.setWindowTitle("Запросить удаление")
    layout = QVBoxLayout(dialog)
    summary = QLabel(
        f"Отправить заявку на полное удаление проекта «{project_name}»?\n"
        "После подтверждения в Kraken Admin будут удалены исходные изображения и результаты.\n"
        "До подтверждения проект останется доступен."
    )
    summary.setWordWrap(True)
    layout.addWidget(summary)
    reason = QLineEdit(dialog)
    reason.setObjectName("deletionRequestReason")
    reason.setPlaceholderText("Необязательно")
    form = QFormLayout()
    form.addRow("Причина удаления", reason)
    layout.addLayout(form)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Отправить заявку")
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return reason.text()

def _plugin_panel(items: list[PluginInventoryItem], *, launch_plugin=None):
    from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

    host = QWidget()
    layout = QVBoxLayout(host)
    for inventory in items:
        plugin = inventory.metadata
        row = QFrame()
        row.setObjectName("pluginInventoryRow")
        row_layout = QHBoxLayout(row)
        capabilities = ", ".join(capability.operation for capability in plugin.capabilities) or "standalone only"
        label = QLabel(f"{plugin.display_name} {plugin.version}\n{plugin.description}\n{capabilities}")
        label.setWordWrap(True)
        row_layout.addWidget(label, 1)
        button = QPushButton("Открыть")
        button.setEnabled(launch_plugin is not None and plugin.enabled and (inventory.installed or bool(plugin.source_dir)))
        if launch_plugin is not None:
            button.clicked.connect(lambda _checked=False, selected=plugin: launch_plugin(selected))
        row_layout.addWidget(button)
        layout.addWidget(row)
    layout.addStretch(1)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(host)
    return scroll
