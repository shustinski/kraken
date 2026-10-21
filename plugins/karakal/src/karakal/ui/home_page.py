"""Home page shown when Karakal starts: the last projects (open one at once), open a project from
any folder, or start a new run."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

Translate = Callable[..., str]
_MUTED = "color: #8a94a3;"


def _when(text: str) -> str:
    try:
        return datetime.fromisoformat(text).strftime("%d.%m.%Y %H:%M")
    except (TypeError, ValueError):
        return ""


class HomePage(QWidget):
    openRequested = pyqtSignal(str)
    browseRequested = pyqtSignal()
    newRunRequested = pyqtSignal()
    batchRequested = pyqtSignal()
    forgetRequested = pyqtSignal(str)
    projectsFolderRequested = pyqtSignal()

    def __init__(self, translate: Translate, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("karakalHomePage")
        self._t = translate
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 36, 48, 36)
        root.setSpacing(12)
        self.title = QLabel(self)
        self.title.setStyleSheet("font-size: 24px; font-weight: 700;")
        root.addWidget(self.title)
        self.subtitle = QLabel(self)
        self.subtitle.setStyleSheet(_MUTED)
        self.subtitle.setWordWrap(True)
        root.addWidget(self.subtitle)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.new_button = QPushButton(self)
        self.new_button.setStyleSheet("font-weight: 700;")
        self.batch_button = QPushButton(self)
        self.batch_button.setStyleSheet("font-weight: 700;")
        self.browse_button = QPushButton(self)
        self.folder_button = QPushButton(self)
        for button in (self.new_button, self.batch_button, self.browse_button, self.folder_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        root.addLayout(buttons)
        self.recent_title = QLabel(self)
        self.recent_title.setStyleSheet("font-size: 15px; font-weight: 600; margin-top: 12px;")
        root.addWidget(self.recent_title)
        self.projects = QListWidget(self)
        self.projects.setAlternatingRowColors(True)
        self.projects.setSpacing(2)
        root.addWidget(self.projects, 1)
        self.empty_hint = QLabel(self)
        self.empty_hint.setWordWrap(True)
        self.empty_hint.setStyleSheet(_MUTED)
        root.addWidget(self.empty_hint)
        actions = QHBoxLayout()
        self.open_button = QPushButton(self)
        self.forget_button = QPushButton(self)
        actions.addWidget(self.open_button)
        actions.addWidget(self.forget_button)
        actions.addStretch(1)
        root.addLayout(actions)

        self.new_button.clicked.connect(self.newRunRequested)
        self.batch_button.clicked.connect(self.batchRequested)
        self.browse_button.clicked.connect(self.browseRequested)
        self.folder_button.clicked.connect(self.projectsFolderRequested)
        self.open_button.clicked.connect(self._open_current)
        self.forget_button.clicked.connect(self._forget_current)
        self.projects.itemActivated.connect(lambda item: self._open_item(item))
        self.projects.itemDoubleClicked.connect(lambda item: self._open_item(item))
        self.projects.currentItemChanged.connect(lambda *_args: self._sync_buttons())
        self.retranslate(translate)
        self.set_entries([])

    def retranslate(self, translate: Translate) -> None:
        self._t = translate
        self.title.setText(self._t("home.title"))
        self.subtitle.setText(self._t("home.subtitle"))
        self.new_button.setText(self._t("home.new_run"))
        self.batch_button.setText(self._t("home.batch"))
        self.batch_button.setToolTip(self._t("home.batch_hint"))
        self.browse_button.setText(self._t("home.open_other"))
        self.folder_button.setText(self._t("home.projects_folder"))
        self.recent_title.setText(self._t("home.recent"))
        self.empty_hint.setText(self._t("home.empty"))
        self.open_button.setText(self._t("home.open"))
        self.forget_button.setText(self._t("home.forget"))

    def set_entries(self, entries: list[dict]) -> None:
        self.projects.clear()
        for entry in entries:
            path = Path(entry["path"])
            if entry.get("missing"):
                details = self._t("home.missing")
            else:
                parts = []
                if entry.get("frames"):
                    parts.append(self._t("home.frames", count=entry["frames"]))
                layers = [name for name in entry.get("layers") or () if name]
                if layers:
                    parts.append(", ".join(layers))
                when = _when(str(entry.get("saved_at") or ""))
                if when:
                    parts.append(self._t("home.saved", when=when))
                details = " · ".join(parts)
            item = QListWidgetItem(f"{entry.get('name') or path.name}\n{details}\n{path}")
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            if entry.get("missing"):
                item.setForeground(Qt.GlobalColor.gray)
            self.projects.addItem(item)
        if self.projects.count():
            self.projects.setCurrentRow(0)
        self.empty_hint.setVisible(self.projects.count() == 0)
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        has = self.projects.currentItem() is not None
        self.open_button.setEnabled(has)
        self.forget_button.setEnabled(has)

    def _open_item(self, item: QListWidgetItem | None) -> None:
        if item is not None:
            self.openRequested.emit(str(item.data(Qt.ItemDataRole.UserRole) or ""))

    def _open_current(self) -> None:
        self._open_item(self.projects.currentItem())

    def _forget_current(self) -> None:
        item = self.projects.currentItem()
        if item is not None:
            self.forgetRequested.emit(str(item.data(Qt.ItemDataRole.UserRole) or ""))
