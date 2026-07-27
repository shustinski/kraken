"""Project-manager desktop shell, intentionally free of composition logic."""

from __future__ import annotations

from collections.abc import Mapping

from PyQt6.QtCore import Qt, pyqtSignal
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

        self._register_default_pages()
        self.navigation_list.currentRowChanged.connect(self._on_navigation_row_changed)
        self.show_page("projects")

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
        self.statusBar().showMessage("Выбрано кадров: 0")
        self.show_page("workspace")
        return page

    def set_session_summary(self, text: str) -> None:
        self.session_label.setText(str(text) or "Нет активной сессии")

    def _on_navigation_row_changed(self, row: int) -> None:
        item = self.navigation_list.item(row)
        if item is None:
            return
        key = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not key:
            return
        self.navigationRequested.emit(key)
        self.show_page(key)


__all__ = ["ProjectManagerShell"]

