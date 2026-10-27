"""Shared nonmodal viewer for verified local and remote Karakal PNG maps."""
import json

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QDialog, QLabel, QPlainTextEdit, QScrollArea, QToolButton, QVBoxLayout


def show_analysis_map(parent, run_id, frame_id, data, *, expression, language="Русский", sources=(),
                      runtime=None, object_name="analysisMapDialog"):
    def text(ru, en):
        return ru if language.lower().startswith(("ru", "рус")) else en

    pixmap = QPixmap()
    if not pixmap.loadFromData(data, "PNG"):
        raise ValueError(text("Некорректная карта PNG", "Invalid PNG map"))
    dialog = QDialog(parent)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.setObjectName(object_name)
    dialog.setWindowTitle(text("Карта расхождений", "Discrepancy map") + ": " + run_id + " / " + frame_id)
    layout = QVBoxLayout(dialog)
    legend = QLabel(text("Красный: только левая часть выражения · Синий: только правая · Белый: совпадение · Чёрный: фон",
                         "Red: left operand only · Blue: right operand only · White: overlap · Black: background"))
    legend.setWordWrap(True)
    layout.addWidget(legend)
    caption = QLabel(json.dumps(expression, ensure_ascii=False))
    caption.setWordWrap(True)
    caption.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    layout.addWidget(caption)
    runtime_label = QLabel(text("Среда worker", "Worker runtime"))
    runtime_label.setToolTip(json.dumps(runtime, ensure_ascii=False, indent=2) if runtime else
                             text("Среда worker не подтверждена", "Worker runtime unverified"))
    layout.addWidget(runtime_label)
    if runtime:
        receipt = QPlainTextEdit()
        receipt.setObjectName("analysisMapRuntime")
        receipt.setReadOnly(True)
        receipt.setMaximumHeight(100)
        receipt.setPlainText(json.dumps(runtime, ensure_ascii=False, indent=2))
        layout.addWidget(receipt)
    if sources:
        details = QPlainTextEdit()
        details.setObjectName("analysisMapInputs")
        details.setReadOnly(True)
        details.setPlainText("\n".join(f"{item['binding_key']}: {item['artifact_version_id']}\nSHA-256: {item['sha256']}" for item in sources))
        details.setMaximumHeight(130)
        details.setVisible(False)
        toggle = QToolButton()
        toggle.setText(text("Входные версии", "Input versions"))
        toggle.setCheckable(True)
        toggle.toggled.connect(details.setVisible)
        layout.addWidget(toggle)
        layout.addWidget(details)
    image = QLabel()
    image.setObjectName("analysisMapImage")
    image.setPixmap(pixmap)
    scroll = QScrollArea()
    scroll.setWidget(image)
    layout.addWidget(scroll)
    dialog.resize(min(1000, max(500, pixmap.width() + 50)), min(800, max(300, pixmap.height() + 100)))
    dialog.show()
    return dialog


def show_runtime_report(parent, report, *, language="Русский"):
    from PyQt6.QtWidgets import QDialogButtonBox

    russian = language.lower().startswith(("ru", "рус"))
    dialog = QDialog(parent)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.setObjectName("analysisRuntimeDialog")
    dialog.setWindowTitle("Среда worker" if russian else "Worker runtime")
    layout = QVBoxLayout(dialog)
    label = QLabel(("Записанная среда вычисления; отсутствие записи не означает совпадение сред."
                    if russian else "Recorded computation environment; a missing receipt does not establish compatibility."))
    label.setWordWrap(True)
    layout.addWidget(label)
    content = QPlainTextEdit()
    content.setObjectName("analysisRuntimeReport")
    content.setReadOnly(True)
    content.setPlainText(json.dumps(report, ensure_ascii=False, indent=2))
    layout.addWidget(content)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.close)
    layout.addWidget(buttons)
    dialog.resize(850, 550)
    dialog.show()
    return dialog
