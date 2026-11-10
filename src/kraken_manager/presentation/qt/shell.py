"""Project-manager desktop shell, intentionally free of composition logic."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Callable

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from kraken_core.settings import QSettings
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QStackedWidget,
    QTabBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .pages import (
    AdministrationPage,
    MyWorkPage,
    PerformersPage,
    PluginsPage,
    ProjectCatalogPage,
    StatisticsPage,
)


class _ProjectTabBar(QTabBar):
    """Project tab bar that reports re-selection of the current tab.

    ``QTabBar.setCurrentIndex`` is a silent no-op when the index does not
    change, and a bar with tabs cannot be deselected (``-1`` is ignored).
    Re-selecting an already-current project tab after a section page has been
    shown must still re-activate the project page, so the same-index set is
    reported by re-emitting ``currentChanged``.
    """

    def setCurrentIndex(self, index: int) -> None:
        previous = self.currentIndex()
        super().setCurrentIndex(index)
        if index == previous and index >= 0:
            self.currentChanged.emit(index)


class ProjectManagerShell(QMainWindow):
    """Navigation and page registry for the future desktop composition root."""

    pageChanged = pyqtSignal(str)
    navigationRequested = pyqtSignal(str)
    layersRequested = pyqtSignal()
    workPlanningRequested = pyqtSignal()
    multilayerRequested = pyqtSignal()
    electricalRequested = pyqtSignal()
    qualityRequested = pyqtSignal()
    cellVisualModeChanged = pyqtSignal(str, str)
    preferencesRequested = pyqtSignal()
    reviewReturnRequested = pyqtSignal()
    framePropertiesRequested = pyqtSignal()
    project_tab_activated = pyqtSignal(object)
    project_tab_close_requested = pyqtSignal(str)
    addProjectRequested = pyqtSignal()

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
        self._selection_counts: dict[str, int] = {}
        self.close_guard: Callable[[], bool] | None = None

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.rail = self._build_rail()
        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("managerPageStack")
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addLayout(self._build_project_tab_row())
        content_layout.addWidget(self.page_stack, 1)
        root.addWidget(self.rail)
        root.addWidget(content, 1)
        self.setCentralWidget(central)

        self._session_summary_label = QLabel("Нет активной сессии")
        self._session_summary_label.setObjectName("sessionSummary")
        self._session_summary_label.setWordWrap(True)
        self._sync_status_label = QLabel("")
        self._sync_status_label.setObjectName("syncStatus")
        self._sync_status_label.setWordWrap(True)
        self._sync_status_label.hide()
        self.statusBar().addWidget(self._session_summary_label)
        self.statusBar().addWidget(self._sync_status_label)

        self._build_workspace_menus()

        self._register_default_pages()
        self.projectTabs.currentChanged.connect(self._on_project_tab_changed)
        self.projectTabs.tabCloseRequested.connect(self._on_project_tab_close_requested)
        self.add_project_button.clicked.connect(self._on_add_project_button_clicked)
        self.show_page("projects")

    def _build_workspace_menus(self) -> None:
        self._ui_settings = QSettings("Kraken", "KrakenHub")
        management = self.menuBar().addMenu("Управление")
        self.layers_action = QAction("Слои…", self)
        self.layers_action.setEnabled(False)
        self.layers_action.triggered.connect(self.layersRequested)
        management.addAction(self.layers_action)
        self.work_planning_action = QAction("План восстановления…", self)
        self.work_planning_action.triggered.connect(self.workPlanningRequested)
        management.addAction(self.work_planning_action)
        self.multilayer_action = QAction("Физические слои и контакты…", self)
        self.multilayer_action.triggered.connect(self.multilayerRequested)
        management.addAction(self.multilayer_action)
        self.electrical_action = QAction("Электрическая связность…", self)
        self.electrical_action.triggered.connect(self.electricalRequested)
        management.addAction(self.electrical_action)
        self.quality_action = QAction("Контроль качества…", self)
        self.quality_action.triggered.connect(self.qualityRequested)
        management.addAction(self.quality_action)

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
        self.preferences_action = QAction("Настройки…", self)
        self.preferences_action.triggered.connect(self.preferencesRequested.emit)
        self.view_menu.addAction(self.preferences_action)
        self.view_menu.addSeparator()
        self._visual_menus: dict[str, QMenu] = {}
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
            self._visual_menus[channel] = submenu
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

    def _build_rail(self) -> QWidget:
        rail = QFrame()
        rail.setObjectName("managerRail")
        rail.setFixedWidth(48)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(6)
        self.rail_actions: dict[str, QToolButton] = {}
        group = QButtonGroup(rail)
        # Qt 6 exclusive groups lock the checked button (setChecked(False) on
        # it is a no-op), but an active project tab must leave every section
        # button unchecked, so exclusivity is enforced by show_page() instead.
        group.setExclusive(False)
        for key, label in self.DEFAULT_NAVIGATION:
            button = QToolButton()
            button.setObjectName(f"railButton_{key}")
            button.setCheckable(True)
            button.setAutoRaise(True)
            button.setToolTip(str(label))
            button.setIcon(QIcon(self._rail_icon(key)))
            button.setIconSize(QSize(22, 22))
            button.clicked.connect(lambda _checked=False, section=key: self._on_rail_clicked(section))
            group.addButton(button)
            self.rail_actions[key] = button
            layout.addWidget(button)
        layout.addStretch(1)
        return rail

    def _on_rail_clicked(self, key: str) -> None:
        self.navigationRequested.emit(key)
        self.show_page(key)

    @staticmethod
    def _rail_icon(key: str) -> QPixmap:
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#647d9c"), 2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        if key == "projects":
            folder = QPainterPath()
            folder.moveTo(3, 8)
            folder.lineTo(9, 8)
            folder.lineTo(11, 10)
            folder.lineTo(21, 10)
            folder.lineTo(21, 18)
            folder.lineTo(3, 18)
            folder.closeSubpath()
            painter.drawPath(folder)
        elif key == "my_work":
            painter.drawEllipse(QPointF(12, 8), 3.5, 3.5)
            painter.drawArc(QRectF(5, 12, 14, 14), 180 * 16, 180 * 16)
        elif key == "statistics":
            painter.drawLine(QPointF(6, 19), QPointF(6, 11))
            painter.drawLine(QPointF(12, 19), QPointF(12, 5))
            painter.drawLine(QPointF(18, 19), QPointF(18, 13))
        elif key == "performers":
            painter.drawEllipse(QPointF(9, 8), 3, 3)
            painter.drawEllipse(QPointF(15, 8), 3, 3)
            painter.drawArc(QRectF(2, 12, 13, 13), 180 * 16, 180 * 16)
            painter.drawArc(QRectF(9, 12, 13, 13), 180 * 16, 180 * 16)
        elif key == "plugins":
            painter.drawRoundedRect(QRectF(7, 11, 10, 8), 2, 2)
            painter.drawLine(QPointF(10, 11), QPointF(10, 4))
            painter.drawLine(QPointF(14, 11), QPointF(14, 4))
        elif key == "administration":
            painter.drawEllipse(QPointF(12, 12), 4, 4)
            for angle in range(0, 360, 45):
                radians = math.radians(angle)
                painter.drawLine(
                    QPointF(12 + 6 * math.cos(radians), 12 + 6 * math.sin(radians)),
                    QPointF(12 + 9 * math.cos(radians), 12 + 9 * math.sin(radians)),
                )
        painter.end()
        return pixmap

    def _build_project_tab_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.projectTabs = _ProjectTabBar()
        self.projectTabs.setObjectName("projectTabs")
        self.projectTabs.setShape(QTabBar.Shape.RoundedNorth)
        self.projectTabs.setExpanding(False)
        self.projectTabs.setDrawBase(True)
        self.projectTabs.setMovable(False)
        self.projectTabs.setTabsClosable(False)
        self._tab_close_icon = self._paint_tab_close_icon()
        row.addWidget(self.projectTabs, 1)
        self.add_project_button = QPushButton()
        self.add_project_button.setObjectName("addProjectButton")
        self.add_project_button.setToolTip("Открыть проект")
        self.add_project_button.setFixedSize(38, 38)
        plus_pixmap = QPixmap(20, 20)
        plus_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(plus_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        plus_pen = QPen(QColor("#f5f9ff"), 2.5)
        plus_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(plus_pen)
        painter.drawLine(QPointF(4, 10), QPointF(16, 10))
        painter.drawLine(QPointF(10, 4), QPointF(10, 16))
        painter.end()
        self.add_project_button.setIcon(QIcon(plus_pixmap))
        self.add_project_button.setIconSize(plus_pixmap.size())
        self.add_project_button.setStyleSheet(
            """
            QPushButton#addProjectButton {
                padding: 0;
            }
            """
        )
        row.addWidget(self.add_project_button)
        return row

    @staticmethod
    def _paint_tab_close_icon() -> QIcon:
        pixmap = QPixmap(12, 12)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#647d9c"), 1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(2, 2), QPointF(10, 10))
        painter.drawLine(QPointF(10, 2), QPointF(2, 10))
        painter.end()
        return QIcon(pixmap)

    def _build_tab_close_button(self) -> QToolButton:
        button = QToolButton()
        button.setObjectName("projectTabCloseButton")
        button.setAutoRaise(True)
        button.setFixedSize(16, 16)
        button.setIcon(self._tab_close_icon)
        button.setIconSize(QSize(12, 12))
        button.setToolTip("Закрыть проект")
        return button

    def _register_default_pages(self) -> None:
        pages: Mapping[str, QWidget] = {
            "projects": ProjectCatalogPage(),
            "my_work": MyWorkPage(),
            "statistics": StatisticsPage(),
            "performers": PerformersPage(),
            "plugins": PluginsPage(),
            "administration": AdministrationPage(),
        }
        for key, page in pages.items():
            self.register_page(key, page)

    def register_page(self, key: str, page: QWidget) -> None:
        normalized = str(key).strip()
        if not normalized:
            raise ValueError("page key must not be empty")
        if normalized in self._pages:
            raise ValueError(f"page {normalized!r} is already registered")
        self._pages[normalized] = page
        self.page_stack.addWidget(page)

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
        for section, button in self.rail_actions.items():
            button.setChecked(section == normalized)
        if changed:
            self.pageChanged.emit(normalized)
        self._sync_chrome_for_current_page()
        if not normalized.startswith("project:"):
            self.project_tab_activated.emit(None)

    def set_page_visible(self, key: str, visible: bool) -> None:
        button = self.rail_actions[str(key)]
        button.setVisible(bool(visible))
        if not visible and self.current_page_key() == str(key):
            self.show_page("projects")

    def open_project_tab(self, project_id: str, name: str, page: QWidget) -> QWidget:
        key = f"project:{project_id}"
        existing = self._pages.get(key)
        if existing is not None:
            index = self._tab_index_for_project(project_id)
            if index >= 0:
                # _ProjectTabBar reports even the already-current index, so a
                # re-open always runs the activation sync.
                self.projectTabs.setCurrentIndex(index)
            return existing
        self.register_page(key, page)
        # The tab bar does not auto-activate the added tab; keep its signals
        # blocked until the tab data is in place, then activate it explicitly.
        self.projectTabs.blockSignals(True)
        try:
            index = self.projectTabs.addTab(str(name))
            self.projectTabs.setTabData(index, str(project_id))
            self.projectTabs.setTabToolTip(index, str(name))
        finally:
            self.projectTabs.blockSignals(False)
        close_button = self._build_tab_close_button()
        self.projectTabs.setTabButton(index, QTabBar.ButtonPosition.RightSide, close_button)
        close_button.clicked.connect(
            lambda _checked=False, tab_index=index: self.projectTabs.tabCloseRequested.emit(tab_index)
        )
        # The first added tab becomes current silently; _ProjectTabBar
        # re-emits currentChanged for that case, driving the activation.
        self.projectTabs.setCurrentIndex(index)
        if hasattr(page, "selectionCountChanged"):
            page.selectionCountChanged.connect(
                lambda count, pid=project_id: self._on_project_selection(pid, int(count))
            )
        return page

    def close_project_tab(self, project_id: str) -> None:
        index = self._tab_index_for_project(project_id)
        if index < 0:
            return
        key = f"project:{project_id}"
        page = self._pages.pop(key)
        was_current = self.page_stack.currentWidget() is page
        self.page_stack.removeWidget(page)
        page.close()
        page.setParent(None)
        self._selection_counts.pop(project_id, None)
        self.projectTabs.blockSignals(True)
        try:
            self.projectTabs.removeTab(index)
        finally:
            self.projectTabs.blockSignals(False)
        if was_current:
            if self.projectTabs.count() > 0:
                # removeTab already moved the bar (its signals were blocked);
                # _ProjectTabBar re-emits currentChanged for that index.
                self.projectTabs.setCurrentIndex(min(index, self.projectTabs.count() - 1))
            else:
                self.show_page("projects")

    def rename_project_tab(self, project_id: str, name: str) -> None:
        index = self._tab_index_for_project(project_id)
        if index >= 0:
            self.projectTabs.setTabText(index, str(name))

    def current_project_id(self) -> str | None:
        key = self.current_page_key()
        if key.startswith("project:"):
            return key[len("project:"):]
        return None

    def _sync_chrome_for_current_page(self) -> None:
        key = self.current_page_key()
        project_active = key.startswith("project:")
        self.layers_action.setEnabled(project_active)
        for menu in self._visual_menus.values():
            menu.setEnabled(project_active)
        self.actions_menu.setEnabled(project_active)
        if not project_active:
            self.frame_properties_action.setEnabled(False)

    def _tab_index_for_project(self, project_id: str) -> int:
        for index in range(self.projectTabs.count()):
            if self.projectTabs.tabData(index) == project_id:
                return index
        return -1

    def _on_project_tab_changed(self, index: int) -> None:
        if index < 0:
            self.project_tab_activated.emit(None)
            return
        project_id = str(self.projectTabs.tabData(index) or "")
        page = self._pages.get(f"project:{project_id}")
        if page is None:
            return
        self.page_stack.setCurrentWidget(page)
        for button in self.rail_actions.values():
            button.setChecked(False)
        self._sync_chrome_for_current_page()
        self._push_selection_count()
        self.project_tab_activated.emit(project_id)

    def _on_project_tab_close_requested(self, index: int) -> None:
        project_id = str(self.projectTabs.tabData(index) or "")
        if project_id:
            self.project_tab_close_requested.emit(project_id)

    def _on_add_project_button_clicked(self) -> None:
        self.addProjectRequested.emit()
        self.show_page("projects")

    def _on_project_selection(self, project_id: str, count: int) -> None:
        # Only the active project drives the status bar; an inactive
        # project's emissions neither update the bar nor its stored count.
        if self.current_project_id() != project_id:
            return
        self._selection_counts[project_id] = count
        self.statusBar().showMessage(f"Выбрано кадров: {count:n}")
        self.frame_properties_action.setEnabled(count == 1)

    def _push_selection_count(self) -> None:
        project_id = self.current_project_id()
        if project_id is None:
            return
        count = self._selection_counts.get(project_id, 0)
        self.statusBar().showMessage(f"Выбрано кадров: {count:n}")
        self.frame_properties_action.setEnabled(count == 1)

    def set_session_summary(self, text: str) -> None:
        self._session_summary_label.setText(str(text) or "Нет активной сессии")

    def set_sync_status(self, status: str) -> None:
        labels = {
            "synchronized": "● Синхронизировано",
            "reconnecting": "● Переподключение…",
            "offline": "● Офлайн",
        }
        label = self._sync_status_label
        label.setText(labels.get(str(status), str(status)))
        label.setProperty("status", str(status))
        label.setVisible(bool(status))
        label.style().unpolish(label)
        label.style().polish(label)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.close_guard is not None and not self.close_guard():
            event.ignore()
            return
        super().closeEvent(event)


__all__ = ["ProjectManagerShell"]
