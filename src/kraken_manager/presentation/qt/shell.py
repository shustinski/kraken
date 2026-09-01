"""Project-manager desktop shell, intentionally free of composition logic."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

from PyQt6.QtCore import QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .pages import (
    AdministrationPage,
    MyWorkPage,
    PerformersPage,
    PluginsPage,
    ProjectCatalogPage,
    ProjectWorkspacePage,
    StatisticsPage,
)


class ProjectManagerShell(QMainWindow):
    """Navigation and page registry for the future desktop composition root."""

    pageChanged = pyqtSignal(str)
    navigationRequested = pyqtSignal(str)
    layersRequested = pyqtSignal()
    cellVisualModeChanged = pyqtSignal(str, str)
    reviewReturnRequested = pyqtSignal()
    framePropertiesRequested = pyqtSignal()

    DEFAULT_NAVIGATION = (
        ("projects", "Проекты"),
        ("my_work", "Моя работа"),
        ("statistics", "Статистика"),
        ("performers", "Исполнители"),
        ("plugins", "Плагины"),
        ("administration", "Администрирование"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("projectManagerShell")
        self.setWindowTitle("Kraken")
        self.resize(1280, 800)
        self._pages: dict[str, QWidget] = {}
        self._navigation_items: dict[str, QListWidgetItem] = {}
        self.close_guard: Callable[[], bool] | None = None

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.sidebar = self._build_sidebar()
        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("managerPageStack")
        root.addWidget(self.sidebar)
        root.addWidget(self.page_stack, 1)
        self.setCentralWidget(central)
        self._build_workspace_menus()

        self._register_default_pages()
        self.navigation_list.currentRowChanged.connect(self._on_navigation_row_changed)
        self.show_page("projects")

    def _build_workspace_menus(self) -> None:
        self._ui_settings = QSettings("Kraken", "KrakenHub")
        management = self.menuBar().addMenu("Управление")
        self.layers_action = QAction("Слои…", self)
        self.layers_action.setEnabled(False)
        self.layers_action.triggered.connect(self.layersRequested)
        management.addAction(self.layers_action)

        self.actions_menu = self.menuBar().addMenu("Действия")
        self.actions_menu.setEnabled(False)
        self.review_return_action = QAction("Загрузить проверенные файлы…", self)
        self.review_return_action.triggered.connect(self.reviewReturnRequested)
        self.actions_menu.addAction(self.review_return_action)
        self.frame_properties_action = QAction("Статистика выбранного кадра…", self)
        self.frame_properties_action.setEnabled(False)
        self.frame_properties_action.triggered.connect(self.framePropertiesRequested)
        self.actions_menu.addAction(self.frame_properties_action)

        self.view_menu = self.menuBar().addMenu("Вид")
        self.view_menu.setEnabled(False)
        self._visual_actions: dict[tuple[str, str], QAction] = {}
        labels = {
            "time": "Время",
            "performer": "Исполнитель",
            "quality": "Качество",
            "status": "Статус",
            "thumbnail": "Миниатюра",
        }
        defaults = {"border": "status", "fill": "thumbnail"}
        for channel, title, modes in (
            ("border", "Рамка ячейки", ("time", "performer", "quality", "status")),
            ("fill", "Заполнение ячейки", ("time", "performer", "quality", "status", "thumbnail")),
        ):
            submenu = self.view_menu.addMenu(title)
            group = QActionGroup(self)
            group.setExclusive(True)
            selected = str(self._ui_settings.value(f"matrix/{channel}-mode", defaults[channel]))
            if selected not in modes:
                selected = defaults[channel]
            for mode in modes:
                action = QAction(labels[mode], self, checkable=True)
                action.setData(mode)
                action.setChecked(mode == selected)
                action.triggered.connect(
                    lambda _checked=False, c=channel, m=mode: self._set_visual_mode(c, m)
                )
                group.addAction(action)
                submenu.addAction(action)
                self._visual_actions[(channel, mode)] = action

    def _set_visual_mode(self, channel: str, mode: str) -> None:
        self._ui_settings.setValue(f"matrix/{channel}-mode", mode)
        self.cellVisualModeChanged.emit(channel, mode)

    def visual_modes(self) -> tuple[str, str]:
        border = next(
            (mode for (channel, mode), action in self._visual_actions.items() if channel == "border" and action.isChecked()),
            "status",
        )
        fill = next(
            (mode for (channel, mode), action in self._visual_actions.items() if channel == "fill" and action.isChecked()),
            "thumbnail",
        )
        return border, fill

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("managerSidebar")
        sidebar.setMinimumWidth(190)
        sidebar.setMaximumWidth(280)
        sidebar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setSpacing(12)
        brand = QLabel("KRAKEN")
        brand.setObjectName("managerBrand")
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(brand)
        self.navigation_list = QListWidget()
        self.navigation_list.setObjectName("managerNavigation")
        self.navigation_list.setSpacing(3)
        layout.addWidget(self.navigation_list, 1)
        self.session_label = QLabel("Нет активной сессии")
        self.session_label.setObjectName("sessionSummary")
        self.session_label.setWordWrap(True)
        layout.addWidget(self.session_label)
        self.sync_status_label = QLabel("")
        self.sync_status_label.setObjectName("syncStatus")
        self.sync_status_label.setWordWrap(True)
        self.sync_status_label.hide()
        layout.addWidget(self.sync_status_label)
        return sidebar

    def _register_default_pages(self) -> None:
        pages: Mapping[str, QWidget] = {
            "projects": ProjectCatalogPage(),
            "my_work": MyWorkPage(),
            "statistics": StatisticsPage(),
            "performers": PerformersPage(),
            "plugins": PluginsPage(),
            "administration": AdministrationPage(),
        }
        labels = dict(self.DEFAULT_NAVIGATION)
        for key, page in pages.items():
            self.register_page(key, labels[key], page)

    def register_page(
        self,
        key: str,
        label: str,
        page: QWidget,
        *,
        navigation: bool = True,
    ) -> None:
        normalized = str(key).strip()
        if not normalized:
            raise ValueError("page key must not be empty")
        if normalized in self._pages:
            raise ValueError(f"page {normalized!r} is already registered")
        self._pages[normalized] = page
        self.page_stack.addWidget(page)
        if navigation:
            item = QListWidgetItem(str(label))
            item.setData(Qt.ItemDataRole.UserRole, normalized)
            self.navigation_list.addItem(item)
            self._navigation_items[normalized] = item

    def replace_page(self, key: str, page: QWidget) -> QWidget:
        """Swap a placeholder without changing its navigation position."""
        if key not in self._pages:
            raise KeyError(key)
        previous = self._pages[key]
        stack_index = self.page_stack.indexOf(previous)
        was_current = self.page_stack.currentWidget() is previous
        self.page_stack.removeWidget(previous)
        previous.close()
        previous.setParent(None)
        self.page_stack.insertWidget(stack_index, page)
        self._pages[key] = page
        if was_current:
            self.page_stack.setCurrentWidget(page)
        return previous

    def page(self, key: str) -> QWidget | None:
        return self._pages.get(str(key))

    def current_page_key(self) -> str:
        current = self.page_stack.currentWidget()
        return next((key for key, page in self._pages.items() if page is current), "")

    def show_page(self, key: str) -> None:
        normalized = str(key)
        page = self._pages.get(normalized)
        if page is None:
            raise KeyError(normalized)
        changed = self.page_stack.currentWidget() is not page
        self.page_stack.setCurrentWidget(page)
        navigation_item = self._navigation_items.get(normalized)
        if navigation_item is not None and self.navigation_list.currentItem() is not navigation_item:
            self.navigation_list.setCurrentItem(navigation_item)
        if changed:
            self.pageChanged.emit(normalized)
        workspace_active = normalized == "workspace"
        self.layers_action.setEnabled(workspace_active)
        self.view_menu.setEnabled(workspace_active)
        self.actions_menu.setEnabled(workspace_active)
        if not workspace_active:
            self.frame_properties_action.setEnabled(False)

    def set_page_visible(self, key: str, visible: bool) -> None:
        item = self._navigation_items.get(str(key))
        if item is None:
            raise KeyError(str(key))
        item.setHidden(not bool(visible))
        if not visible and self.current_page_key() == str(key):
            self.show_page("projects")

    def open_project_workspace(self, workspace: ProjectWorkspacePage | None = None) -> ProjectWorkspacePage:
        """Register or replace the non-sidebar workspace and display it."""
        page = workspace or ProjectWorkspacePage()
        if "workspace" in self._pages:
            self.replace_page("workspace", page)
        else:
            self.register_page("workspace", "Проект", page, navigation=False)
        page.selectionCountChanged.connect(
            lambda count: self.statusBar().showMessage(f"Выбрано кадров: {count:n}")
        )
        page.selectionCountChanged.connect(
            lambda count: self.frame_properties_action.setEnabled(count == 1)
        )
        self.statusBar().showMessage("Выбрано кадров: 0")
        self.show_page("workspace")
        return page

    def set_session_summary(self, text: str) -> None:
        self.session_label.setText(str(text) or "Нет активной сессии")

    def set_sync_status(self, status: str) -> None:
        labels = {
            "synchronized": "● Синхронизировано",
            "reconnecting": "● Переподключение…",
            "offline": "● Офлайн",
        }
        self.sync_status_label.setText(labels.get(str(status), str(status)))
        self.sync_status_label.setProperty("status", str(status))
        self.sync_status_label.setVisible(bool(status))
        self.sync_status_label.style().unpolish(self.sync_status_label)
        self.sync_status_label.style().polish(self.sync_status_label)

    def _on_navigation_row_changed(self, row: int) -> None:
        item = self.navigation_list.item(row)
        if item is None:
            return
        key = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not key:
            return
        self.navigationRequested.emit(key)
        self.show_page(key)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.close_guard is not None and not self.close_guard():
            event.ignore()
            return
        super().closeEvent(event)


__all__ = ["ProjectManagerShell"]

