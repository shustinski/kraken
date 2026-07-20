"""Small Qt list models used by the project-manager shell."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Generic, TypeVar

from PyQt6.QtCore import QAbstractListModel, QModelIndex, Qt
from PyQt6.QtGui import QColor


@dataclass(frozen=True, slots=True)
class ProjectListItem:
    """Presentation snapshot for one project in a catalog."""

    project_id: str
    name: str
    width: int
    height: int
    storage_label: str = ""
    status: str = "active"
    archived: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not str(self.project_id).strip():
            raise ValueError("project_id must not be empty")
        if not str(self.name).strip():
            raise ValueError("project name must not be empty")
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("project dimensions must be positive")
        object.__setattr__(self, "project_id", str(self.project_id))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))

    @property
    def subtitle(self) -> str:
        dimensions = f"{self.width} × {self.height}"
        return f"{dimensions} · {self.storage_label}" if self.storage_label else dimensions


@dataclass(frozen=True, slots=True)
class LayerListItem:
    """Presentation snapshot for a project layer."""

    layer_id: str
    name: str
    layer_type: str
    color: str = "#64748b"
    active: bool = True
    coverage: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not str(self.layer_id).strip():
            raise ValueError("layer_id must not be empty")
        if not str(self.name).strip():
            raise ValueError("layer name must not be empty")
        if self.coverage is not None and not 0.0 <= float(self.coverage) <= 1.0:
            raise ValueError("layer coverage must be between 0 and 1")
        object.__setattr__(self, "layer_id", str(self.layer_id))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "layer_type", str(self.layer_type))

    @property
    def subtitle(self) -> str:
        if self.coverage is None:
            return self.layer_type
        return f"{self.layer_type} · {self.coverage:.0%}"


class ProjectItemRole(IntEnum):
    ID = int(Qt.ItemDataRole.UserRole) + 1
    WIDTH = ID + 1
    HEIGHT = ID + 2
    STORAGE = ID + 3
    STATUS = ID + 4
    ARCHIVED = ID + 5
    ITEM = ID + 6


class LayerItemRole(IntEnum):
    ID = int(Qt.ItemDataRole.UserRole) + 1
    TYPE = ID + 1
    COLOR = ID + 2
    ACTIVE = ID + 3
    COVERAGE = ID + 4
    ITEM = ID + 5


ItemT = TypeVar("ItemT")


class _SnapshotListModel(QAbstractListModel, Generic[ItemT]):
    """Replaceable read-only snapshots; commands stay outside the Qt model."""

    def __init__(self, items: Iterable[ItemT] = (), parent=None) -> None:
        super().__init__(parent)
        self._items = list(items)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # type: ignore[override]
        return 0 if parent.isValid() else len(self._items)

    def items(self) -> tuple[ItemT, ...]:
        return tuple(self._items)

    def item_at(self, row: int) -> ItemT | None:
        return self._items[row] if 0 <= row < len(self._items) else None

    def item_for_index(self, index: QModelIndex) -> ItemT | None:
        return self.item_at(index.row()) if index.isValid() else None

    def replace_items(self, items: Iterable[ItemT]) -> None:
        prepared = list(items)
        self.beginResetModel()
        self._items = prepared
        self.endResetModel()

    set_items = replace_items

    def append_item(self, item: ItemT) -> None:
        row = len(self._items)
        self.beginInsertRows(QModelIndex(), row, row)
        self._items.append(item)
        self.endInsertRows()

    def remove_row(self, row: int) -> bool:
        if not 0 <= row < len(self._items):
            return False
        self.beginRemoveRows(QModelIndex(), row, row)
        self._items.pop(row)
        self.endRemoveRows()
        return True


class ProjectListModel(_SnapshotListModel[ProjectListItem]):
    """Project catalog model with stable role names for future delegates/QML."""

    IdRole = ProjectItemRole.ID
    WidthRole = ProjectItemRole.WIDTH
    HeightRole = ProjectItemRole.HEIGHT
    StorageRole = ProjectItemRole.STORAGE
    StatusRole = ProjectItemRole.STATUS
    ArchivedRole = ProjectItemRole.ARCHIVED
    ItemRole = ProjectItemRole.ITEM

    def data(self, index: QModelIndex, role: int = int(Qt.ItemDataRole.DisplayRole)):  # type: ignore[override]
        item = self.item_for_index(index)
        if item is None:
            return None
        if role == int(Qt.ItemDataRole.DisplayRole):
            return item.name
        if role == int(Qt.ItemDataRole.ToolTipRole):
            return f"{item.name}\n{item.subtitle}"
        values = {
            int(ProjectItemRole.ID): item.project_id,
            int(ProjectItemRole.WIDTH): item.width,
            int(ProjectItemRole.HEIGHT): item.height,
            int(ProjectItemRole.STORAGE): item.storage_label,
            int(ProjectItemRole.STATUS): item.status,
            int(ProjectItemRole.ARCHIVED): item.archived,
            int(ProjectItemRole.ITEM): item,
        }
        return values.get(int(role))

    def roleNames(self) -> dict[int, bytes]:  # type: ignore[override]
        return {
            int(ProjectItemRole.ID): b"projectId",
            int(ProjectItemRole.WIDTH): b"width",
            int(ProjectItemRole.HEIGHT): b"height",
            int(ProjectItemRole.STORAGE): b"storage",
            int(ProjectItemRole.STATUS): b"status",
            int(ProjectItemRole.ARCHIVED): b"archived",
            int(ProjectItemRole.ITEM): b"item",
        }

    def project_by_id(self, project_id: str) -> ProjectListItem | None:
        return next((item for item in self._items if item.project_id == str(project_id)), None)

    def upsert(self, item: ProjectListItem) -> None:
        for row, existing in enumerate(self._items):
            if existing.project_id != item.project_id:
                continue
            self._items[row] = item
            index = self.index(row, 0)
            self.dataChanged.emit(index, index)
            return
        self.append_item(item)


class LayerListModel(_SnapshotListModel[LayerListItem]):
    """Layer sidebar model independent of layer domain aggregates."""

    IdRole = LayerItemRole.ID
    TypeRole = LayerItemRole.TYPE
    ColorRole = LayerItemRole.COLOR
    ActiveRole = LayerItemRole.ACTIVE
    CoverageRole = LayerItemRole.COVERAGE
    ItemRole = LayerItemRole.ITEM

    def data(self, index: QModelIndex, role: int = int(Qt.ItemDataRole.DisplayRole)):  # type: ignore[override]
        item = self.item_for_index(index)
        if item is None:
            return None
        if role == int(Qt.ItemDataRole.DisplayRole):
            return item.name
        if role == int(Qt.ItemDataRole.ToolTipRole):
            return f"{item.name}\n{item.subtitle}"
        if role == int(Qt.ItemDataRole.DecorationRole):
            color = QColor(item.color)
            return color if color.isValid() else QColor("#64748b")
        values = {
            int(LayerItemRole.ID): item.layer_id,
            int(LayerItemRole.TYPE): item.layer_type,
            int(LayerItemRole.COLOR): item.color,
            int(LayerItemRole.ACTIVE): item.active,
            int(LayerItemRole.COVERAGE): item.coverage,
            int(LayerItemRole.ITEM): item,
        }
        return values.get(int(role))

    def roleNames(self) -> dict[int, bytes]:  # type: ignore[override]
        return {
            int(LayerItemRole.ID): b"layerId",
            int(LayerItemRole.TYPE): b"layerType",
            int(LayerItemRole.COLOR): b"color",
            int(LayerItemRole.ACTIVE): b"active",
            int(LayerItemRole.COVERAGE): b"coverage",
            int(LayerItemRole.ITEM): b"item",
        }

    def layer_by_id(self, layer_id: str) -> LayerListItem | None:
        return next((item for item in self._items if item.layer_id == str(layer_id)), None)

    def upsert(self, item: LayerListItem) -> None:
        for row, existing in enumerate(self._items):
            if existing.layer_id != item.layer_id:
                continue
            self._items[row] = item
            index = self.index(row, 0)
            self.dataChanged.emit(index, index)
            return
        self.append_item(item)


__all__ = [
    "LayerItemRole",
    "LayerListItem",
    "LayerListModel",
    "ProjectItemRole",
    "ProjectListItem",
    "ProjectListModel",
]

