"""Lightweight pages that expose signals and accept presentation models."""

from __future__ import annotations

from PyQt6.QtCore import QModelIndex, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .models import LayerListItem, LayerListModel, ProjectListItem, ProjectListModel
from .widgets import FrameMatrixView


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
        actions.addWidget(self.create_button)
        actions.addWidget(self.refresh_button)
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
        self.project_list.doubleClicked.connect(self._activate_index)
        self.project_list.selectionModel().currentChanged.connect(self._selection_changed)
        self.project_model.modelReset.connect(self._sync_empty_state)
        self.project_model.rowsInserted.connect(self._sync_empty_state)
        self.project_model.rowsRemoved.connect(self._sync_empty_state)

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
        self.selectionChanged.emit(self.project_model.item_for_index(current))

    def _sync_empty_state(self, *_args: object) -> None:
        empty = self.project_model.rowCount() == 0
        self.project_list.setVisible(not empty)
        self.empty_label.setVisible(empty)


class ProjectWorkspacePage(_TitledPage):
    """Three-pane layer/matrix/inspector workspace ready for presenter wiring."""

    layerActivated = pyqtSignal(object)
    addLayerRequested = pyqtSignal()
    imageRepresentationChanged = pyqtSignal(str)
    vectorRepresentationChanged = pyqtSignal(str)

    def __init__(
        self,
        layer_model: LayerListModel | None = None,
        matrix_view: FrameMatrixView | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Проект", parent=parent)
        self.description_label.hide()

        representation_row = QHBoxLayout()
        representation_row.addWidget(QLabel("Изображения:"))
        self.image_representation_combo = QComboBox()
        self.image_representation_combo.setObjectName("imageRepresentationCombo")
        representation_row.addWidget(self.image_representation_combo, 1)
        representation_row.addWidget(QLabel("Векторы:"))
        self.vector_representation_combo = QComboBox()
        self.vector_representation_combo.setObjectName("vectorRepresentationCombo")
        representation_row.addWidget(self.vector_representation_combo, 1)
        self.root_layout.addLayout(representation_row)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("projectWorkspaceSplitter")
        self.layer_panel = self._build_layer_panel(layer_model)
        self.matrix_view = matrix_view or FrameMatrixView()
        self.matrix_view.setMinimumSize(320, 240)
        self.inspector_panel = self._build_inspector_panel()
        self.splitter.addWidget(self.layer_panel)
        self.splitter.addWidget(self.matrix_view)
        self.splitter.addWidget(self.inspector_panel)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([220, 720, 260])
        self.root_layout.addWidget(self.splitter, 1)

        self.layer_list.doubleClicked.connect(self._activate_layer)
        self.add_layer_button.clicked.connect(self.addLayerRequested)
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

    def _build_layer_panel(self, model: LayerListModel | None) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sidePanel")
        panel.setMinimumWidth(180)
        panel.setMaximumWidth(340)
        layout = QVBoxLayout(panel)
        title = QLabel("Слои")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        self.layer_list = QListView()
        self.layer_list.setObjectName("layerList")
        self.layer_model = model or LayerListModel(parent=self)
        self.layer_list.setModel(self.layer_model)
        layout.addWidget(self.layer_list, 1)
        self.add_layer_button = QPushButton("Добавить слой")
        layout.addWidget(self.add_layer_button)
        return panel

    def _build_inspector_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sidePanel")
        panel.setMinimumWidth(210)
        panel.setMaximumWidth(380)
        layout = QVBoxLayout(panel)
        title = QLabel("Инспектор")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        self.inspector_summary = QLabel("Выберите кадр или область")
        self.inspector_summary.setWordWrap(True)
        self.inspector_summary.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.inspector_summary.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(self.inspector_summary, 1)
        return panel

    def set_project_title(self, name: str) -> None:
        self.title_label.setText(str(name) or "Проект")

    def set_layer_model(self, model: LayerListModel) -> None:
        self.layer_model = model
        self.layer_list.setModel(model)

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

    def _activate_layer(self, index: QModelIndex) -> None:
        item = self.layer_model.item_for_index(index)
        if isinstance(item, LayerListItem):
            self.layerActivated.emit(item)

    def _show_selection_summary(self, selection) -> None:
        if selection.is_empty:
            self.inspector_summary.setText("Выберите кадр или область")
            return
        rectangle_count = len(selection.rectangles)
        frame_count = sum(rectangle.frame_count for rectangle in selection.rectangles)
        self.inspector_summary.setText(
            f"Выбрано областей: {rectangle_count}\nКадров (до объединения): {frame_count:n}"
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
