"""Project-manager desktop shell, intentionally free of composition logic."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Callable

from PyQt6.QtCore import QByteArray, QPointF, QSize, Qt, pyqtSignal
from kraken_core.settings import QSettings
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtSvg import QSvgRenderer
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
        self.projectTabs.tabBarClicked.connect(self._on_project_tab_clicked)
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
            "none": "Нет",
            "time": "Время",
            "performer": "Исполнитель",
            "quality": "Качество",
            "status": "Статус",
            "thumbnail": "Миниатюра",
        }
        defaults = {"border": "status", "fill": "thumbnail"}
        for channel, title, modes in (
            ("border", "Рамка ячейки", ("none", "time", "performer", "quality", "status")),
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
            button.setIcon(self._rail_icon(key))
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
    def _gear_path() -> str:
        """Outline of an 8-tooth gear centred in the 24x24 icon box."""
        teeth, outer, inner = 8, 9.5, 7.0
        step = 2 * math.pi / teeth
        points: list[tuple[float, float]] = []
        for index in range(teeth):
            base = index * step - math.pi / 2
            for angle, radius in (
                (base - step * 0.30, inner),
                (base - step * 0.17, outer),
                (base + step * 0.17, outer),
                (base + step * 0.30, inner),
            ):
                points.append((12 + radius * math.cos(angle), 12 + radius * math.sin(angle)))
        head, *rest = points
        return f"M{head[0]:.2f} {head[1]:.2f}" + "".join(f"L{x:.2f} {y:.2f}" for x, y in rest) + "Z"

    @classmethod
    def _rail_icon_svg(cls, key: str) -> str:
        bodies = {
            # Folder with a tab.
            "projects": (
                '<path d="M3 7.5A1.5 1.5 0 0 1 4.5 6H9l2 2.5h8.5A1.5 1.5 0 0 1 21 10v8.5'
                'a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 18.5z"/>'
            ),
            # Clipboard with a check mark: "my tasks".
            "my_work": (
                '<rect x="5" y="4.5" width="14" height="17" rx="2"/>'
                '<rect x="9" y="2.5" width="6" height="4" rx="1"/>'
                '<path d="M9 13.5l2.2 2.2L15.5 11"/>'
            ),
            # Bar chart on axes.
            "statistics": (
                '<path d="M4 4v16h16"/>'
                '<path d="M8.5 16v-4M12.5 16V8M16.5 16v-6"/>'
            ),
            # Two people (head + shoulders).
            "performers": (
                '<circle cx="9" cy="8" r="3.5"/>'
                '<path d="M2.5 20c0-3.6 2.9-6 6.5-6s6.5 2.4 6.5 6"/>'
                '<path d="M15.5 4.6a3.5 3.5 0 0 1 0 6.8M18 14.4c2 .7 3.5 2.7 3.5 5.6"/>'
            ),
            # Puzzle piece: an add-on.
            "plugins": (
                '<path transform="translate(1 -1)" d="M4 8h4a2.2 2.2 0 1 1 4 0h4v2'
                'a2.2 2.2 0 1 1 0 4v6H4z"/>'
            ),
            # Gear.
            "administration": f'<path d="{cls._gear_path()}"/><circle cx="12" cy="12" r="3"/>',
        }
        return bodies.get(key, "")

    @classmethod
    def _rail_icon(cls, key: str) -> QIcon:
        icon = QIcon()
        body = cls._rail_icon_svg(key)
        for state, color in ((QIcon.State.Off, "#7f93ad"), (QIcon.State.On, "#4c9aff")):
            svg = (
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
                f'stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
                f"{body}</svg>"
            )
            renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
            for scale in (1, 2):
                pixmap = QPixmap(24 * scale, 24 * scale)
                pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                renderer.render(painter)
                painter.end()
                pixmap.setDevicePixelRatio(scale)
                icon.addPixmap(pixmap, QIcon.Mode.Normal, state)
        return icon

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
            self.projectTabs.setTabToolTip(index, str(name))

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

    def _on_project_tab_clicked(self, index: int) -> None:
        # Qt's mouse path bypasses _ProjectTabBar.setCurrentIndex, so a click
        # on the already-current tab never re-emits currentChanged. Re-drive
        # the activation only when a section page is showing; clicks on a
        # different tab are handled by Qt's own currentChanged.
        if (
            index >= 0
            and self.projectTabs.currentIndex() == index
            and not self.current_page_key().startswith("project:")
        ):
            self.projectTabs.setCurrentIndex(index)

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
