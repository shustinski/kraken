"""Card widget that shows one Contour path (folder) with its status.

The card replaces a bare ``QLineEdit``: it shows the folder name, its parent
directory and a short status line, and keeps manual editing one double-click
away. ``text()`` / ``setText()`` mirror ``QLineEdit`` so existing callers keep
working; ``setText`` never emits :attr:`PathCard.pathCommitted`, just like
``QLineEdit.textEdited`` is not emitted for programmatic changes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from PyQt6.QtCore import QDir, QEvent, QObject, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QFileSystemModel,
    QGuiApplication,
    QIcon,
    QMouseEvent,
    QResizeEvent,
)
from PyQt6.QtWidgets import (
    QCompleter,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

ROLE_INPUT = "input"
ROLE_CIF = "cif"
ROLE_OUTPUT = "output"
ROLE_DATASET = "dataset"
PATH_CARD_ROLES = (ROLE_INPUT, ROLE_CIF, ROLE_OUTPUT, ROLE_DATASET)

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_ERROR = "error"
STATUS_MUTED = "muted"
_STATUS_LEVELS = (STATUS_OK, STATUS_WARN, STATUS_ERROR, STATUS_MUTED)
_STATUS_MARKS = {STATUS_OK: "✓ ", STATUS_WARN: "⚠ ", STATUS_ERROR: "✗ ", STATUS_MUTED: ""}

PATH_CARD_STYLE = """
QFrame#pathCard {
    border: 1px solid palette(mid);
    border-radius: 6px;
    background: palette(base);
}
QFrame#pathCard:hover {
    border-color: palette(highlight);
}
QFrame#pathCard[empty="true"] {
    border-style: dashed;
    background: transparent;
}
QFrame#pathCard[locked="true"] {
    background: palette(window);
}
QLabel#pathCardRole {
    color: palette(placeholder-text);
    font-size: 10px;
    font-weight: 600;
}
QLabel#pathCardName {
    font-weight: 600;
}
QLabel#pathCardName[empty="true"] {
    font-weight: 400;
    color: palette(placeholder-text);
}
QLabel#pathCardParent {
    color: palette(placeholder-text);
    font-size: 11px;
}
QLabel#pathCardStatus {
    font-size: 11px;
}
QLabel#pathCardStatus[level="ok"] { color: #16A34A; }
QLabel#pathCardStatus[level="warn"] { color: #D97706; }
QLabel#pathCardStatus[level="error"] { color: #DC2626; }
QLabel#pathCardStatus[level="muted"] { color: palette(placeholder-text); }
QLabel#pathCardBadge {
    border: 1px solid palette(mid);
    border-radius: 7px;
    padding: 0 6px;
    font-size: 10px;
    color: palette(placeholder-text);
}
QToolButton#pathCardMenuButton {
    border: none;
    border-radius: 4px;
    padding: 0 6px;
    font-size: 16px;
    font-weight: 700;
}
QToolButton#pathCardMenuButton:hover {
    background: palette(midlight);
}
QToolButton#pathCardMenuButton::menu-indicator {
    image: none;
}
"""

_MAX_RECENT_MENU_ITEMS = 8

# Translation keys per role: (role caption, empty-state text, choose-files action).
_ROLE_TEXT_KEYS: dict[str, tuple[str, str, str]] = {
    ROLE_INPUT: ("input_directory_label", "path_card_empty_input", "pick_input_files_button"),
    ROLE_CIF: ("cif_overlay_directory_label", "path_card_empty_cif", "merge_cif_files_button"),
    ROLE_OUTPUT: ("output_directory_label", "path_card_empty_output", ""),
    ROLE_DATASET: ("dataset_directory_label", "path_card_empty_dataset", ""),
}
_ROLE_ICONS: dict[str, QStyle.StandardPixmap] = {
    ROLE_INPUT: QStyle.StandardPixmap.SP_DirIcon,
    ROLE_CIF: QStyle.StandardPixmap.SP_FileDialogDetailedView,
    ROLE_OUTPUT: QStyle.StandardPixmap.SP_DialogSaveButton,
    ROLE_DATASET: QStyle.StandardPixmap.SP_DriveHDIcon,
}


def split_path_for_display(path: str) -> tuple[str, str]:
    """Return ``(name, parent)`` for a path; the root of a drive is its own name."""

    text = str(path or "").strip()
    if not text:
        return "", ""
    candidate = Path(text)
    name = candidate.name or str(candidate)
    parent = str(candidate.parent) if candidate.name and str(candidate.parent) != "." else ""
    return name, parent


class ElidedLabel(QLabel):
    """Single-line label that elides its full text to the available width."""

    def __init__(self, mode: Qt.TextElideMode = Qt.TextElideMode.ElideMiddle, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = ""
        self._elide_mode = mode
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setTextFormat(Qt.TextFormat.PlainText)

    def full_text(self) -> str:
        return self._full_text

    def set_full_text(self, text: str) -> None:
        self._full_text = str(text or "")
        self._apply_elision()

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._apply_elision()

    def _apply_elision(self) -> None:
        width = max(0, self.contentsRect().width())
        elided = self.fontMetrics().elidedText(self._full_text, self._elide_mode, width) if width else self._full_text
        super().setText(elided)


class _EditorKeyFilter(QObject):
    def __init__(self, card: PathCard) -> None:
        super().__init__(card)
        self._card = card

    def eventFilter(self, watched: QObject | None, event: QEvent | None) -> bool:
        if event is None:
            return False
        if event.type() == QEvent.Type.KeyPress and getattr(event, "key", lambda: None)() == Qt.Key.Key_Escape:
            self._card.cancel_edit()
            return True
        if event.type() == QEvent.Type.FocusOut and self._card.is_editing():
            self._card.commit_edit()
        return False


class PathCard(QFrame):
    """Readable, status-aware replacement for a directory ``QLineEdit``."""

    pathCommitted = pyqtSignal(str)
    browseDirectoryRequested = pyqtSignal()
    browseFilesRequested = pyqtSignal()
    filesDropped = pyqtSignal(list)

    def __init__(
        self,
        role: str,
        tr: Callable[..., str],
        *,
        allow_files: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if role not in PATH_CARD_ROLES:
            raise ValueError(f"Unknown path card role: {role}")
        self._role = role
        self._tr = tr
        self._allow_files = allow_files
        self._path = ""
        self._summary_override = ""
        self._status_text = ""
        self._status_level = STATUS_MUTED
        self._locked = False
        self._lock_tooltip = ""
        self._badge_tooltip = ""
        self._editing = False
        self._recent_provider: Callable[[], Sequence[str]] | None = None
        self._base_tooltip = ""

        self.setObjectName("pathCard")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(PATH_CARD_STYLE)
        self._build()
        self.retranslate()

    # ----------------------------------------------------------------- layout
    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 4, 2, 4)
        root.setSpacing(8)

        self.icon_label = QLabel()
        self.icon_label.setFixedWidth(20)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        icon = self.style().standardIcon(_ROLE_ICONS[self._role]) if self.style() is not None else QIcon()
        self.icon_label.setPixmap(icon.pixmap(18, 18))
        root.addWidget(self.icon_label)

        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(1)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        self.role_label = QLabel()
        self.role_label.setObjectName("pathCardRole")
        header.addWidget(self.role_label)
        header.addStretch(1)
        self.badge_label = QLabel()
        self.badge_label.setObjectName("pathCardBadge")
        self.badge_label.hide()
        header.addWidget(self.badge_label)
        body.addLayout(header)

        self._stack = QStackedWidget()
        self._stack.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        display = QWidget()
        display_layout = QVBoxLayout(display)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(0)
        self.name_label = ElidedLabel(Qt.TextElideMode.ElideRight)
        self.name_label.setObjectName("pathCardName")
        self.parent_label = ElidedLabel(Qt.TextElideMode.ElideMiddle)
        self.parent_label.setObjectName("pathCardParent")
        display_layout.addWidget(self.name_label)
        display_layout.addWidget(self.parent_label)
        self._stack.addWidget(display)

        self.editor = QLineEdit()
        self.editor.setObjectName("pathCardEditor")
        completer_model = QFileSystemModel(self.editor)
        completer_model.setFilter(QDir.Filter.AllDirs | QDir.Filter.NoDotAndDotDot | QDir.Filter.Drives)
        completer_model.setRootPath("")
        completer = QCompleter(completer_model, self.editor)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.editor.setCompleter(completer)
        self.editor.returnPressed.connect(self.commit_edit)
        self.editor.installEventFilter(_EditorKeyFilter(self))
        self._stack.addWidget(self.editor)
        body.addWidget(self._stack)

        self.status_label = ElidedLabel(Qt.TextElideMode.ElideRight)
        self.status_label.setObjectName("pathCardStatus")
        self.status_label.hide()
        body.addWidget(self.status_label)
        root.addLayout(body, 1)

        self.menu_button = QToolButton()
        self.menu_button.setObjectName("pathCardMenuButton")
        self.menu_button.setText("⋮")
        self.menu_button.setCursor(Qt.CursorShape.ArrowCursor)
        self.menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu = QMenu(self.menu_button)
        self.menu.aboutToShow.connect(self._rebuild_menu)
        self.menu_button.setMenu(self.menu)
        root.addWidget(self.menu_button, 0, Qt.AlignmentFlag.AlignTop)

    # --------------------------------------------------------- QLineEdit API
    def text(self) -> str:
        return self._path

    def setText(self, text: str) -> None:
        self._path = str(text or "")
        if self._editing:
            self.editor.setText(self._path)
        self._refresh_display()

    # -------------------------------------------------------------- public API
    @property
    def role(self) -> str:
        return self._role

    def set_status(self, text: str, level: str = STATUS_MUTED) -> None:
        self._status_text = str(text or "")
        self._status_level = level if level in _STATUS_LEVELS else STATUS_MUTED
        self._refresh_status()

    def status_text(self) -> str:
        return self._status_text

    def status_level(self) -> str:
        return self._status_level

    def set_summary_override(self, summary: str) -> None:
        """Show ``summary`` instead of the folder name (e.g. "12 selected files")."""

        self._summary_override = str(summary or "")
        self._refresh_display()

    def set_locked(self, locked: bool, tooltip: str = "") -> None:
        self._locked = bool(locked)
        self._lock_tooltip = str(tooltip or "")
        if self._locked and self._editing:
            self.cancel_edit()
        self.setAcceptDrops(not self._locked)
        self.setCursor(Qt.CursorShape.ArrowCursor if self._locked else Qt.CursorShape.PointingHandCursor)
        self._set_dynamic_property(self, "locked", self._locked)
        self._refresh_badge()
        self._refresh_display()

    def is_locked(self) -> bool:
        return self._locked

    def set_badge(self, text: str, tooltip: str = "") -> None:
        """Small pill in the header, independent of locking (e.g. "Kraken")."""

        self.badge_label.setText(str(text or ""))
        self._badge_tooltip = str(tooltip or "")
        self._refresh_badge()

    def set_recent_provider(self, provider: Callable[[], Sequence[str]] | None) -> None:
        self._recent_provider = provider

    def setToolTip(self, tooltip: str | None) -> None:
        self._base_tooltip = str(tooltip or "")
        self._refresh_tooltip()

    def retranslate(self) -> None:
        role_key, _empty_key, _files_key = _ROLE_TEXT_KEYS[self._role]
        self.role_label.setText(self._tr(role_key).upper())
        self.menu_button.setToolTip(self._tr("path_card_menu_tooltip"))
        self.menu_button.setAccessibleName(self._tr("path_card_menu_tooltip"))
        self.editor.setPlaceholderText(self._tr("path_card_editor_placeholder"))
        self._refresh_display()

    # ------------------------------------------------------------ editing API
    def is_editing(self) -> bool:
        return self._editing

    def begin_edit(self) -> None:
        if self._locked or self._editing:
            return
        self._editing = True
        self.editor.setText(self._path)
        self._stack.setCurrentWidget(self.editor)
        self.editor.setFocus(Qt.FocusReason.OtherFocusReason)
        self.editor.selectAll()

    def commit_edit(self) -> None:
        if not self._editing:
            return
        self._editing = False
        value = self.editor.text().strip()
        self._stack.setCurrentIndex(0)
        if value == self._path:
            return
        self._commit(value)

    def cancel_edit(self) -> None:
        if not self._editing:
            return
        self._editing = False
        self.editor.setText(self._path)
        self._stack.setCurrentIndex(0)

    # ----------------------------------------------------------------- events
    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        super().mouseReleaseEvent(event)
        if event is None or event.button() != Qt.MouseButton.LeftButton or self._locked or self._editing:
            return
        if not self._path:
            self.browseDirectoryRequested.emit()

    def mouseDoubleClickEvent(self, event: QMouseEvent | None) -> None:
        super().mouseDoubleClickEvent(event)
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self.begin_edit()

    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        if event is None:
            return
        if not self._locked and self._dropped_local_paths(event):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent | None) -> None:
        if event is None or self._locked:
            return
        paths = self._dropped_local_paths(event)
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        self.handle_dropped_paths(paths)

    def handle_dropped_paths(self, paths: Sequence[str]) -> None:
        """A single folder sets this card's path; files go to :attr:`filesDropped`."""

        normalized = [str(Path(path)) for path in paths if str(path).strip()]
        if not normalized or self._locked:
            return
        if len(normalized) == 1 and Path(normalized[0]).is_dir():
            self._commit(normalized[0])
            return
        if self._allow_files:
            self.filesDropped.emit(normalized)

    @staticmethod
    def _dropped_local_paths(event: QDragEnterEvent | QDropEvent) -> list[str]:
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            return []
        return [url.toLocalFile() for url in mime.urls() if url.isLocalFile() and url.toLocalFile()]

    # ------------------------------------------------------------------- menu
    def _rebuild_menu(self) -> None:
        menu = self.menu
        menu.clear()
        _role_key, _empty_key, files_key = _ROLE_TEXT_KEYS[self._role]
        if not self._locked:
            menu.addAction(self._tr("path_card_choose_folder"), self.browseDirectoryRequested.emit)
            if self._allow_files and files_key:
                menu.addAction(self._tr(files_key), self.browseFilesRequested.emit)
            recent_menu = menu.addMenu(self._tr("path_card_recent"))
            assert recent_menu is not None
            self._fill_recent_menu(recent_menu)
            menu.addAction(self._tr("path_card_edit_manually"), self.begin_edit)
            menu.addSeparator()
        open_action = menu.addAction(self._tr("path_card_open_in_explorer"), self._open_in_explorer)
        copy_action = menu.addAction(self._tr("path_card_copy_path"), self._copy_path)
        has_path = bool(self._path)
        if open_action is not None:
            open_action.setEnabled(has_path and Path(self._path).exists())
        if copy_action is not None:
            copy_action.setEnabled(has_path)
        if not self._locked:
            menu.addSeparator()
            clear_action = menu.addAction(self._tr("path_card_clear"), lambda: self._commit(""))
            if clear_action is not None:
                clear_action.setEnabled(has_path)

    def _fill_recent_menu(self, recent_menu: QMenu) -> None:
        recent = list(self._recent_provider() if self._recent_provider is not None else ())
        current = str(Path(self._path)) if self._path else ""
        entries = [path for path in recent if path and str(Path(path)) != current][:_MAX_RECENT_MENU_ITEMS]
        if not entries:
            placeholder = recent_menu.addAction(self._tr("path_card_recent_empty"))
            if placeholder is not None:
                placeholder.setEnabled(False)
            return
        for path in entries:
            action = recent_menu.addAction(path, lambda value=path: self._commit(value))
            if action is not None and not Path(path).exists():
                action.setEnabled(False)
                action.setToolTip(self._tr("path_status_missing"))

    def _open_in_explorer(self) -> None:
        if self._path and Path(self._path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._path))

    def _copy_path(self) -> None:
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None and self._path:
            clipboard.setText(self._path)

    # ---------------------------------------------------------------- helpers
    def _commit(self, value: str) -> None:
        if self._locked:
            return
        self.setText(value)
        self.pathCommitted.emit(self._path)

    def _refresh_display(self) -> None:
        empty = not self._path and not self._summary_override
        _role_key, empty_key, _files_key = _ROLE_TEXT_KEYS[self._role]
        if self._summary_override:
            self.name_label.set_full_text(self._summary_override)
            self.parent_label.set_full_text(self._path)
        elif self._path:
            name, parent = split_path_for_display(self._path)
            self.name_label.set_full_text(name)
            self.parent_label.set_full_text(parent)
        else:
            self.name_label.set_full_text("＋ " + self._tr(empty_key) if not self._locked else self._tr(empty_key))
            self.parent_label.set_full_text("")
        # The full path is in the tooltip; a second path line is only worth its
        # height when the name line shows a summary instead of a folder name.
        self.parent_label.setVisible(bool(self._summary_override and self.parent_label.full_text()))
        self._set_dynamic_property(self, "empty", empty)
        self._set_dynamic_property(self.name_label, "empty", empty)
        self._refresh_tooltip()

    def _refresh_status(self) -> None:
        if self._status_text:
            self.status_label.set_full_text(_STATUS_MARKS[self._status_level] + self._status_text)
        else:
            self.status_label.set_full_text("")
        self.status_label.setVisible(bool(self._status_text))
        self._set_dynamic_property(self.status_label, "level", self._status_level)
        self._refresh_tooltip()

    def _refresh_badge(self) -> None:
        self.badge_label.setVisible(bool(self.badge_label.text()))
        self.badge_label.setToolTip(self._lock_tooltip or self._badge_tooltip)

    def _refresh_tooltip(self) -> None:
        lines = [line for line in (self._path, self._status_text, self._lock_tooltip, self._base_tooltip) if line]
        super().setToolTip("\n".join(lines))

    @staticmethod
    def _set_dynamic_property(widget: QWidget, name: str, value: object) -> None:
        if widget.property(name) == value:
            return
        widget.setProperty(name, value)
        style = widget.style()
        if style is not None:
            style.unpolish(widget)
            style.polish(widget)


__all__ = [
    "PATH_CARD_ROLES",
    "PATH_CARD_STYLE",
    "ROLE_CIF",
    "ROLE_DATASET",
    "ROLE_INPUT",
    "ROLE_OUTPUT",
    "STATUS_ERROR",
    "STATUS_MUTED",
    "STATUS_OK",
    "STATUS_WARN",
    "ElidedLabel",
    "PathCard",
    "split_path_for_display",
]
