"""Role dependency graph: rectangles, inclusion arrows, and grant controls."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from kraken_manager.domain.identity import ProjectRole
from kraken_manager.domain.roles import CATALOG

_POSITIONS = {
    "admin": (340.0, 24.0),
    "maintainer": (340.0, 150.0),
    "sewer": (40.0, 300.0),
    "elementer": (340.0, 300.0),
    "viewer": (640.0, 300.0),
    "corrector": (340.0, 450.0),
}
_COLORS = {
    "admin": "#9b4055",
    "maintainer": "#9a6324",
    "elementer": "#73559b",
    "corrector": "#4d738a",
    "sewer": "#278b78",
    "viewer": "#4c5350",
}


class _RoleNode(QGraphicsRectItem):
    def __init__(self, role: ProjectRole, select: Callable[[ProjectRole], None]) -> None:
        super().__init__(0, 0, 190, 72)
        self.role = role
        self._select = select
        definition = CATALOG.roles[role.value]
        self.setFlag(self.GraphicsItemFlag.ItemIsSelectable, True)
        self.setBrush(QColor(_COLORS.get(role.value, "#4c5350")))
        self.setPen(QPen(QColor("#101210"), 1.5))
        title = QGraphicsSimpleTextItem(definition.title, self)
        title.setBrush(QColor("#f3f4f6"))
        title.setPos(12, 10)
        subtitle = QGraphicsSimpleTextItem(role.value, self)
        subtitle.setBrush(QColor("#c3c8c3"))
        subtitle.setPos(12, 38)
        for x in (-5.0, 185.0):
            port = QGraphicsEllipseItem(x, 31.0, 10.0, 10.0, self)
            port.setBrush(QColor("#facc15"))
            port.setPen(QPen(QColor("#111827"), 1.0))

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().mousePressEvent(event)
        if event.button() is Qt.MouseButton.LeftButton:
            self._select(self.role)

    def set_marked(self, *, held: bool, selected: bool, dimmed: bool) -> None:
        color = QColor("#f59e0b" if selected else "#86efac" if held else "#101210")
        self.setPen(QPen(color, 3 if selected or held else 1.5))
        self.setOpacity(0.35 if dimmed else 1.0)


class RoleGraphView(QGraphicsView):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.graph_scene = QGraphicsScene(self)
        self.setScene(self.graph_scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._nodes: dict[str, _RoleNode] = {}
        self._edges: list[tuple[QGraphicsPathItem, _RoleNode, _RoleNode]] = []
        self._select = lambda _role: None
        self.rebuild(self._select)

    def rebuild(self, select: Callable[[ProjectRole], None]) -> None:
        self._select = select
        self.graph_scene.clear()
        self._paint_grid()
        self._nodes.clear()
        self._edges.clear()
        for role in ProjectRole:
            node = _RoleNode(role, select)
            node.setPos(*_POSITIONS.get(role.value, (40.0, 40.0)))
            self.graph_scene.addItem(node)
            self._nodes[role.value] = node
        for role_id, definition in CATALOG.roles.items():
            source = self._nodes.get(role_id)
            if source is None:
                continue
            for included in definition.includes:
                target = self._nodes.get(included)
                if target is None:
                    continue
                edge = QGraphicsPathItem()
                edge.setPen(QPen(QColor("#9ca39c"), 2))
                edge.setZValue(-1)
                self.graph_scene.addItem(edge)
                self._edges.append((edge, source, target))
        self.refresh_edges()
        self.graph_scene.setSceneRect(self.graph_scene.itemsBoundingRect().adjusted(-80, -40, 80, 80))

    def _paint_grid(self) -> None:
        self.graph_scene.setBackgroundBrush(QColor("#1b1d1b"))

    def drawBackground(self, painter: QPainter, rect) -> None:  # type: ignore[no-untyped-def]
        painter.fillRect(rect, QColor("#1b1d1b"))
        for spacing, color in ((20, "#272a27"), (100, "#343834")):
            pen = QPen(QColor(color), 0)
            painter.setPen(pen)
            left = int(rect.left()) - (int(rect.left()) % spacing)
            top = int(rect.top()) - (int(rect.top()) % spacing)
            for x in range(left, int(rect.right()) + spacing, spacing):
                painter.drawLine(float(x), rect.top(), float(x), rect.bottom())
            for y in range(top, int(rect.bottom()) + spacing, spacing):
                painter.drawLine(rect.left(), float(y), rect.right(), float(y))

    def refresh_edges(self) -> None:
        for edge, source, target in self._edges:
            source_rect = source.sceneBoundingRect()
            target_rect = target.sceneBoundingRect()
            start = QPointF(source_rect.center().x(), source_rect.bottom())
            end = QPointF(target_rect.center().x(), target_rect.top())
            path = QPainterPath(start)
            path.cubicTo(
                start.x(),
                start.y() + 40,
                end.x(),
                end.y() - 40,
                end.x(),
                end.y(),
            )
            edge.setPath(path)

    def mark(self, held: set[str], selected: str) -> None:
        closure = set(CATALOG.closure({selected})) if selected else set()
        for role_id, node in self._nodes.items():
            node.set_marked(
                held=role_id in held,
                selected=role_id == selected,
                dimmed=bool(selected) and role_id not in closure and role_id != selected,
            )


class RoleManagementDialog(QDialog):
    """Assign and revoke roles on the same graph that shows what each role includes."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        project_name: str,
        principals: list[Any],
        roles_for: Callable[[Any], frozenset[ProjectRole]],
        change_role: Callable[[Any, ProjectRole, bool], None],
        acting_roles: list[ProjectRole],
        on_acting_role: Callable[[ProjectRole], None],
    ) -> None:
        super().__init__(parent)
        self._principals = principals
        self._roles_for = roles_for
        self._change_role = change_role
        self._on_acting_role = on_acting_role
        self._selected_role = ProjectRole.VIEWER
        self.setWindowTitle(f"Роли — {project_name}")
        self.resize(980, 640)
        root = QVBoxLayout(self)
        hint = QLabel(
            "Прямоугольник — роль, стрелка — «включает возможности». "
            "Назначать роли может только тот, кому это разрешено. "
            "Если отозвать сопровождающего, роли, которые он выдал, снимаются вместе с ним.",
            self,
        )
        hint.setWordWrap(True)
        root.addWidget(hint)
        splitter = QSplitter(self)
        self.people = QListWidget(splitter)
        for principal in principals:
            label = getattr(principal, "display_name", str(principal))
            email = getattr(principal, "email", None)
            if email:
                label = f"{label} ({email})"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, principal)
            self.people.addItem(item)
        graph_host = QWidget(splitter)
        graph_layout = QVBoxLayout(graph_host)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        self.graph = RoleGraphView(graph_host)
        self.graph.rebuild(self._select_role)
        graph_layout.addWidget(self.graph)
        self.detail = QLabel(graph_host)
        self.detail.setWordWrap(True)
        graph_layout.addWidget(self.detail)
        splitter.addWidget(self.people)
        splitter.addWidget(graph_host)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Действовать как", self))
        self.acting = QComboBox(self)
        for role in acting_roles:
            self.acting.addItem(CATALOG.roles[role.value].title, role)
        if self.acting.count() == 0:
            self.acting.addItem("Нет роли", None)
        self.acting.currentIndexChanged.connect(self._acting_changed)
        controls.addWidget(self.acting)
        self.assign_button = QPushButton("Назначить", self)
        self.revoke_button = QPushButton("Отозвать", self)
        self.assign_button.clicked.connect(lambda: self._apply(True))
        self.revoke_button.clicked.connect(lambda: self._apply(False))
        controls.addWidget(self.assign_button)
        controls.addWidget(self.revoke_button)
        close = QPushButton("Закрыть", self)
        close.clicked.connect(self.accept)
        controls.addWidget(close)
        root.addLayout(controls)
        self.people.currentRowChanged.connect(lambda _row: self._refresh())
        if self.people.count():
            self.people.setCurrentRow(0)
        self._acting_changed()
        self._refresh()

    def _current_principal(self) -> Any | None:
        item = self.people.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _select_role(self, role: ProjectRole) -> None:
        self._selected_role = role
        self._refresh()

    def _acting_changed(self) -> None:
        role = self.acting.currentData()
        if isinstance(role, ProjectRole):
            self._on_acting_role(role)

    def _held(self) -> set[str]:
        principal = self._current_principal()
        if principal is None:
            return set()
        return {role.value for role in self._roles_for(principal)}

    def _refresh(self) -> None:
        held = self._held()
        self.graph.mark(held, self._selected_role.value)
        definition = CATALOG.roles[self._selected_role.value]
        includes = ", ".join(
            CATALOG.roles[role].title for role in sorted(CATALOG.closure({definition.role_id}) - {definition.role_id})
        ) or "нет"
        grants = ", ".join(CATALOG.roles[role].title for role in sorted(definition.grants)) or "не назначает роли"
        state = "назначена" if definition.role_id in held else "не назначена"
        self.detail.setText(
            f"{definition.title}: {state}. Включает: {includes}. Может назначать: {grants}."
        )
        self.revoke_button.setEnabled(definition.role_id in held)
        self.assign_button.setEnabled(definition.role_id not in held)

    def _notify(self, message: str) -> None:
        QMessageBox.warning(self, "Недостаточно прав", message)

    def _apply(self, enabled: bool) -> None:
        principal = self._current_principal()
        if principal is None:
            self._notify("Выберите участника.")
            return
        role = self._selected_role
        if not enabled and CATALOG.roles[role.value].grants:
            answer = QMessageBox.question(
                self,
                "Отозвать роль",
                "Вместе с этой ролью будут сняты роли, которые участник выдал другим.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            self._change_role(principal, role, enabled)
        except Exception as exc:
            self._notify(str(exc))
            return
        self._refresh()


__all__ = ["RoleManagementDialog"]
