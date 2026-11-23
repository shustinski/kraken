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

def _stop_plugin_processes(processes) -> None:
    """Stop plugin processes opened from the panel when its window closes."""
    for process in processes:
        if getattr(process, "poll", lambda: 0)() is None:
            process.terminate()
    for process in processes:
        wait = getattr(process, "wait", None)
        if not callable(wait):
            continue
        try:
            wait(timeout=5)
        except Exception:
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()


def _plugin_panel(items: list[PluginInventoryItem], *, launch_plugin=None):
    from PyQt6.QtCore import QEvent, QObject
    from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

    host = QWidget()
    layout = QVBoxLayout(host)
    processes: list = []
    scroll = QScrollArea()

    class _StopOnClose(QObject):
        def eventFilter(self, watched, event):  # noqa: N802 - Qt override
            if event.type() == QEvent.Type.Close:
                _stop_plugin_processes(processes)
            return False

    owner = _StopOnClose(scroll)
    attached: dict[str, object] = {"window": None}

    def start(selected) -> None:
        if launch_plugin is None:
            return
        process = launch_plugin(selected)
        if process is None:
            return
        processes.append(process)
        window = scroll.window()
        if window is not None and attached["window"] is not window:
            window.installEventFilter(owner)
            attached["window"] = window

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
            button.clicked.connect(lambda _checked=False, selected=plugin: start(selected))
        row_layout.addWidget(button)
        layout.addWidget(row)
    layout.addStretch(1)
    scroll.setWidgetResizable(True)
    scroll.setWidget(host)
    scroll.plugin_processes = processes
    return scroll
