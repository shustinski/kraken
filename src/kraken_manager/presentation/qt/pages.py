"""Lightweight pages that expose signals and accept presentation models."""

from __future__ import annotations

from PyQt6.QtCore import QModelIndex, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QPushButton,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from .models import LayerListItem, LayerListModel, ProjectListItem, ProjectListModel
from .widgets import FrameMatrixView, FrameMatrixWidget


class _TitledPage(QWidget):
    """Common page chrome without application-service dependencies."""

    def __init__(self, title: str, description: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("managerPage")
        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(24, 20, 24, 24)
        self.root_layout.setSpacing(12)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("pageTitle")
        self.title_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.root_layout.addWidget(self.title_label)
        self.description_label = QLabel(description)
        self.description_label.setObjectName("pageDescription")
        self.description_label.setWordWrap(True)
        self.description_label.setVisible(bool(description))
        self.root_layout.addWidget(self.description_label)


class ProjectCatalogPage(_TitledPage):
    """Project catalog entry page; loading and commands are wired externally."""

    createRequested = pyqtSignal()
    refreshRequested = pyqtSignal()
    projectActivated = pyqtSignal(object)
    selectionChanged = pyqtSignal(object)
    renameRequested = pyqtSignal(object)
    archiveRequested = pyqtSignal(object)
    restoreRequested = pyqtSignal(object)
    deleteRequested = pyqtSignal(object)

    def __init__(
        self,
        model: ProjectListModel | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Проекты", "Локальные и общие проекты, доступные текущим сессиям.", parent)
        actions = QHBoxLayout()
        self.create_button = QPushButton("Создать проект")
        self.create_button.setObjectName("primaryAction")
        self.refresh_button = QPushButton("Обновить")
        self.rename_button = QPushButton("Переименовать")
        self.rename_button.hide()
        self.archive_button = QPushButton("В архив")
        self.restore_button = QPushButton("Восстановить")
        self.delete_button = QPushButton("Удалить")
        self.delete_button.setObjectName("deleteProjectButton")
        self.delete_button.setToolTip(
            "Удалить проект из Kraken, сохранив папки исходных и производных данных"
        )
        self.show_archived_check = QCheckBox("Показывать архивные")
        actions.addWidget(self.create_button)
        actions.addWidget(self.refresh_button)
        actions.addWidget(self.rename_button)
        actions.addWidget(self.archive_button)
        actions.addWidget(self.restore_button)
        actions.addWidget(self.delete_button)
        actions.addWidget(self.show_archived_check)
        actions.addStretch(1)
        self.root_layout.addLayout(actions)

        self.project_list = QListView()
        self.project_list.setObjectName("projectCatalogList")
        self.project_list.setAlternatingRowColors(True)
        self.project_list.setUniformItemSizes(True)
        self.project_list.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.project_model = model or ProjectListModel(parent=self)
        self.project_list.setModel(self.project_model)
        self.root_layout.addWidget(self.project_list, 1)

        self.empty_label = QLabel("Проектов пока нет")
        self.empty_label.setObjectName("emptyState")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.root_layout.addWidget(self.empty_label)
        self._sync_empty_state()

        self.create_button.clicked.connect(self.createRequested)
        self.refresh_button.clicked.connect(self.refreshRequested)
        self.show_archived_check.toggled.connect(self.refreshRequested)
        self.rename_button.clicked.connect(lambda: self.renameRequested.emit(self.selected_project()))
        self.archive_button.clicked.connect(lambda: self.archiveRequested.emit(self.selected_project()))
        self.restore_button.clicked.connect(lambda: self.restoreRequested.emit(self.selected_project()))
        self.delete_button.clicked.connect(lambda: self.deleteRequested.emit(self.selected_project()))
        self.project_list.doubleClicked.connect(self._activate_index)
        self.project_list.selectionModel().currentChanged.connect(self._selection_changed)
        self.project_model.modelReset.connect(self._sync_empty_state)
        self.project_model.rowsInserted.connect(self._sync_empty_state)
        self.project_model.rowsRemoved.connect(self._sync_empty_state)
        self._sync_selection_actions(None)

    def set_model(self, model: ProjectListModel) -> None:
        self.project_model = model
        self.project_list.setModel(model)
        self.project_list.selectionModel().currentChanged.connect(self._selection_changed)
        model.modelReset.connect(self._sync_empty_state)
        model.rowsInserted.connect(self._sync_empty_state)
        model.rowsRemoved.connect(self._sync_empty_state)
        self._sync_empty_state()

    def selected_project(self) -> ProjectListItem | None:
        return self.project_model.item_for_index(self.project_list.currentIndex())

    def _activate_index(self, index: QModelIndex) -> None:
        item = self.project_model.item_for_index(index)
        if item is not None:
            self.projectActivated.emit(item)

    def _selection_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        item = self.project_model.item_for_index(current)
        self._sync_selection_actions(item)
        self.selectionChanged.emit(item)

    def _sync_selection_actions(self, item: ProjectListItem | None) -> None:
        selected = item is not None
        archived = bool(item.archived) if item is not None else False
        self.rename_button.setEnabled(selected and not archived)
        self.archive_button.setEnabled(selected and not archived)
        self.restore_button.setEnabled(selected and archived)
        self.delete_button.setEnabled(selected)

    def _sync_empty_state(self, *_args: object) -> None:
        empty = self.project_model.rowCount() == 0
        self.project_list.setVisible(not empty)
        self.empty_label.setVisible(empty)


class ProjectWorkspacePage(_TitledPage):
    """Matrix workspace with Excel-style layer tabs."""

    layerActivated = pyqtSignal(object)
    addLayerRequested = pyqtSignal()
    addImageRepresentationRequested = pyqtSignal()
    addVectorRepresentationRequested = pyqtSignal()
    imageRepresentationChanged = pyqtSignal(str)
    vectorRepresentationChanged = pyqtSignal(str)
    selectionCountChanged = pyqtSignal(int)

    def __init__(
        self,
        layer_model: LayerListModel | None = None,
        matrix_view: FrameMatrixView | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Проект", parent=parent)
        self.title_label.hide()
        self.description_label.hide()

        self.image_representation_combo = QComboBox(self)
        self.image_representation_combo.setObjectName("imageRepresentationCombo")
        self.add_image_representation_button = QPushButton("Добавить", self)
        self.add_image_representation_button.setObjectName("addImageRepresentationButton")
        self.vector_representation_combo = QComboBox(self)
        self.vector_representation_combo.setObjectName("vectorRepresentationCombo")
        self.add_vector_representation_button = QPushButton("Добавить", self)
        self.add_vector_representation_button.setObjectName("addVectorRepresentationButton")
        # Compatibility controls remain alive for older automation, but
        # representation selection now lives in the modeless layer manager.
        for compatibility_control in (
            self.image_representation_combo,
            self.add_image_representation_button,
            self.vector_representation_combo,
            self.add_vector_representation_button,
        ):
            compatibility_control.hide()

        self.layer_model = layer_model or LayerListModel(parent=self)
        matrix_toolbar = QHBoxLayout()
        matrix_toolbar.setContentsMargins(0, 0, 0, 0)
        self.zoom_out_button = QPushButton("−")
        self.zoom_out_button.setObjectName("matrixZoomOutButton")
        self.zoom_out_button.setToolTip("Уменьшить")
        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.setObjectName("matrixZoomInButton")
        self.zoom_in_button.setToolTip("Увеличить")
        self.zoom_fit_button = QPushButton("Вписать")
        self.zoom_fit_button.setObjectName("matrixZoomFitButton")
        self.zoom_reset_button = QPushButton("1:1")
        self.zoom_reset_button.setObjectName("matrixZoomResetButton")
        self.matrix_lod_label = QLabel("LOD: cells")
        self.matrix_lod_label.setObjectName("matrixLodLabel")
        self.minimap_checkbox = QCheckBox("Мини-карта")
        self.minimap_checkbox.setObjectName("matrixMinimapCheck")
        self.minimap_checkbox.setChecked(True)
        self.clear_thumbnail_cache_button = QPushButton("Очистить миниатюры")
        self.clear_thumbnail_cache_button.setObjectName("clearThumbnailCacheButton")
        self.matrix_loading_label = QLabel("")
        self.matrix_loading_label.setObjectName("matrixLoadingLabel")
        for control in (
            self.zoom_out_button,
            self.zoom_in_button,
            self.zoom_fit_button,
            self.zoom_reset_button,
            self.matrix_lod_label,
            self.minimap_checkbox,
            self.clear_thumbnail_cache_button,
        ):
            matrix_toolbar.addWidget(control)
        matrix_toolbar.addStretch(1)
        matrix_toolbar.addWidget(self.matrix_loading_label)
        self.root_layout.addLayout(matrix_toolbar)

        self.matrix_view = matrix_view or FrameMatrixWidget()
        self.matrix_view.setMinimumSize(320, 240)
        matrix_host = QFrame()
        matrix_host.setObjectName("matrixHost")
        matrix_host_layout = QHBoxLayout(matrix_host)
        matrix_host_layout.setContentsMargins(0, 0, 0, 0)
        matrix_host_layout.addWidget(self.matrix_view, 1)
        self.matrix_minimap = QLabel("Мини-карта")
        self.matrix_minimap.setObjectName("matrixMinimap")
        self.matrix_minimap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.matrix_minimap.setFixedWidth(150)
        self.matrix_minimap.setStyleSheet("background:#111827; border:1px solid #475569; color:#94a3b8;")
        matrix_host_layout.addWidget(self.matrix_minimap)
        self.root_layout.addWidget(matrix_host, 1)

        layer_row = QHBoxLayout()
        layer_row.setContentsMargins(0, 0, 0, 0)
        self.layer_tabs = QTabBar()
        self.layer_tabs.setObjectName("layerTabs")
        self.layer_tabs.setShape(QTabBar.Shape.RoundedSouth)
        self.layer_tabs.setExpanding(False)
        self.layer_tabs.setDrawBase(True)
        layer_row.addWidget(self.layer_tabs, 1)
        self.add_layer_button = QPushButton()
        self.add_layer_button.setObjectName("addLayerButton")
        self.add_layer_button.setToolTip("Добавить слой")
        self.add_layer_button.setFixedSize(38, 38)
        plus_pixmap = QPixmap(20, 20)
        plus_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(plus_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#f5f9ff"), 2.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(4, 10), QPointF(16, 10))
        painter.drawLine(QPointF(10, 4), QPointF(10, 16))
        painter.end()
        self.add_layer_button.setIcon(QIcon(plus_pixmap))
        self.add_layer_button.setIconSize(plus_pixmap.size())
        self.add_layer_button.setStyleSheet(
            """
            QPushButton#addLayerButton {
                padding: 0;
            }
            """
        )
        layer_row.addWidget(self.add_layer_button)
        self.root_layout.addLayout(layer_row)

        self.layer_tabs.currentChanged.connect(self._activate_layer_tab)
        self.add_layer_button.clicked.connect(self.addLayerRequested)
        self.add_image_representation_button.clicked.connect(self.addImageRepresentationRequested)
        self.add_vector_representation_button.clicked.connect(self.addVectorRepresentationRequested)
        self.image_representation_combo.currentIndexChanged.connect(
            lambda: self.imageRepresentationChanged.emit(
                str(self.image_representation_combo.currentData() or "")
            )
        )
        self.vector_representation_combo.currentIndexChanged.connect(
            lambda: self.vectorRepresentationChanged.emit(
                str(self.vector_representation_combo.currentData() or "")
            )
        )
        self.matrix_view.selectionChanged.connect(self._show_selection_summary)
        self.zoom_out_button.clicked.connect(
            lambda: self.matrix_view.set_zoom_factor(max(self.matrix_view.MIN_ZOOM, self.matrix_view.zoom_factor() / 1.25))
        )
        self.zoom_in_button.clicked.connect(
            lambda: self.matrix_view.set_zoom_factor(min(self.matrix_view.MAX_ZOOM, self.matrix_view.zoom_factor() * 1.25))
        )
        self.zoom_fit_button.clicked.connect(self.matrix_view.zoom_to_fit)
        self.zoom_reset_button.clicked.connect(self.matrix_view.reset_zoom)
        self.matrix_view.lodChanged.connect(lambda lod: self.matrix_lod_label.setText(f"LOD: {lod}"))
        self.minimap_checkbox.toggled.connect(self.matrix_minimap.setVisible)
        if isinstance(self.matrix_view, FrameMatrixWidget):
            self.clear_thumbnail_cache_button.clicked.connect(self.matrix_view.clear_thumbnail_cache)
            self.matrix_view.loadingChanged.connect(
                lambda loading: self.matrix_loading_label.setText("Загрузка…" if loading else "")
            )
            self.matrix_view.errorOccurred.connect(
                lambda message: self.matrix_loading_label.setText(f"Ошибка: {message}")
            )
            self.matrix_view.viewportChanged.connect(self._update_minimap_summary)

    def set_project_title(self, name: str) -> None:
        project_name = str(name).strip()
        window = self.window()
        window.setWindowTitle(f"Kraken — {project_name}" if project_name else "Kraken")

    def set_layer_model(self, model: LayerListModel) -> None:
        self.layer_model = model
        self.sync_layer_tabs()

    def sync_layer_tabs(self) -> None:
        current_id = self.layer_tabs.tabData(self.layer_tabs.currentIndex())
        self.layer_tabs.blockSignals(True)
        while self.layer_tabs.count():
            self.layer_tabs.removeTab(0)
        selected = -1
        for row in range(self.layer_model.rowCount()):
            item = self.layer_model.item_for_index(self.layer_model.index(row, 0))
            if not isinstance(item, LayerListItem):
                continue
            index = self.layer_tabs.addTab(item.name)
            self.layer_tabs.setTabData(index, item.layer_id)
            self.layer_tabs.setTabToolTip(index, item.layer_type)
            if item.layer_id == current_id:
                selected = index
        if self.layer_tabs.count():
            self.layer_tabs.setCurrentIndex(max(0, selected))
        self.layer_tabs.blockSignals(False)

    def set_representations(
        self,
        *,
        images: list[tuple[str, str]] = (),
        vectors: list[tuple[str, str]] = (),
    ) -> None:
        self._replace_combo_items(self.image_representation_combo, images)
        self._replace_combo_items(self.vector_representation_combo, vectors)

    @staticmethod
    def _replace_combo_items(combo: QComboBox, items: list[tuple[str, str]]) -> None:
        combo.blockSignals(True)
        combo.clear()
        for identifier, name in items:
            combo.addItem(name, identifier)
        combo.blockSignals(False)

    def _activate_layer_tab(self, tab_index: int) -> None:
        layer_id = self.layer_tabs.tabData(tab_index)
        item = self.layer_model.layer_by_id(str(layer_id or ""))
        if isinstance(item, LayerListItem):
            self.layerActivated.emit(item)

    def _show_selection_summary(self, selection) -> None:
        self.selectionCountChanged.emit(sum(1 for _ in selection.coordinates()))

    def _update_minimap_summary(self, _visible_rect) -> None:
        width, height = self.matrix_view.matrix_size()
        self.matrix_minimap.setText(
            f"{width:n} × {height:n}\n{self.matrix_view.lod_level().value}"
        )


class _PlaceholderPage(_TitledPage):
    """Stable shell destination while a feature presenter is connected later."""

    def __init__(self, title: str, description: str, parent: QWidget | None = None) -> None:
        super().__init__(title, description, parent)
        self.content_host = QFrame()
        self.content_host.setObjectName("pageContentHost")
        self.content_layout = QVBoxLayout(self.content_host)
        self.placeholder_label = QLabel("Раздел готов к подключению данных")
        self.placeholder_label.setObjectName("emptyState")
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(self.placeholder_label, 1)
        self.root_layout.addWidget(self.content_host, 1)

    def set_content(self, widget: QWidget) -> None:
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            old_widget = item.widget()
            if old_widget is not None:
                old_widget.hide()
                old_widget.setParent(None)
        self.content_layout.addWidget(widget)


class MyWorkPage(_PlaceholderPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Моя работа", "Назначенные задания, проверки и сроки.", parent)


class StatisticsPage(_PlaceholderPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Статистика", "Метрики проектов и экспорт отчётов за период.", parent)


class PerformersPage(_PlaceholderPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Исполнители", "Пользователи GitLab и ручные исполнители.", parent)


class PluginsPage(_PlaceholderPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Плагины", "Доступные интеграции и фоновые задания.", parent)


class AdministrationPage(_PlaceholderPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Администрирование", "Хранилища, аккаунты, резервные копии и аудит.", parent)


__all__ = [
    "AdministrationPage",
    "MyWorkPage",
    "PerformersPage",
    "PluginsPage",
    "ProjectCatalogPage",
    "ProjectWorkspacePage",
    "StatisticsPage",
]
