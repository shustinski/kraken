"""User/project role matrix using the shared Kraken role catalog."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QLabel, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from kraken_manager.domain.common import ProjectId
from kraken_manager.domain.identity import ProjectRole
from kraken_manager.domain.roles import CATALOG
from kraken_manager.presentation.qt.role_management import RoleManagementDialog

from .project_roles import all_principals, cascade_preview, set_roles, snapshot


class RolesPanel(QWidget):
    def __init__(self, admin) -> None:
        super().__init__(admin.window)
        self.admin = admin
        self.people = []
        self.projects = []
        self.assignments = {}
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Двойной щелчок по ячейке — изменить набор ролей."))
        self.table = QTableWidget()
        self.table.setObjectName("adminRoleMatrix")
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self.edit)
        layout.addWidget(self.table)
        graph = QPushButton("Граф ролей выбранного проекта…")
        graph.clicked.connect(self.graph)
        layout.addWidget(graph)

    def reload(self) -> None:
        services, accounts = self.admin._services, self.admin._accounts
        if services is None or accounts is None or not hasattr(services.identities, "list"):
            return
        self.people = all_principals(services, accounts)
        self.projects = services.list_projects(include_archived=True)
        # One batched ACL read, independent of matrix cell count.
        import sqlalchemy as sa
        with services.engine.connect() as connection:
            rows = connection.execute(sa.select(services.identities.acl).where(
                services.identities.acl.c.revoked_at.is_(None),
            )).mappings().all()
        self.assignments = {}
        for row in rows:
            self.assignments.setdefault((str(row["principal_id"]), str(row["project_id"])), set()).add(row["role"])
        self.table.setRowCount(len(self.people))
        self.table.setColumnCount(len(self.projects) + 1)
        self.table.setHorizontalHeaderLabels(["Пользователь", *(project["name"] for project in self.projects)])
        for row, person in enumerate(self.people):
            self.table.setItem(row, 0, QTableWidgetItem(
                f"{person.display_name} ({person.subject})" + (" — отключён" if not person.active else ""),
            ))
            for column, project in enumerate(self.projects, 1):
                roles = self.assignments.get((str(person.id), str(project["project_id"])), set())
                cell = QTableWidgetItem(", ".join(CATALOG.roles[role].title for role in sorted(roles)) or "Нет ролей")
                cell.setData(Qt.ItemDataRole.UserRole, (str(person.id), str(project["project_id"])))
                self.table.setItem(row, column, cell)

    def _save(self, project_id, person, desired, revision, assignments) -> None:
        lost = cascade_preview(assignments, person, desired)
        if lost:
            names = {str(item.id): item.display_name for item in self.people}
            text = "Будут также отозваны:\n" + "\n".join(
                f"{names.get(holder, holder)} — {CATALOG.roles[role].title}" for holder, role in lost
            )
            if QMessageBox.question(self, "Каскадный отзыв", text) != QMessageBox.StandardButton.Yes:
                return
        try:
            set_roles(self.admin._services, self.admin._accounts, project_id=project_id, principal=person,
                      desired=desired, expected_revision=revision, expected_snapshot=assignments)
        finally:
            self.admin.reload()

    def edit(self, row: int, column: int) -> None:
        if column <= 0 or row < 0:
            return
        person = self.people[row]
        project_id = str(self.projects[column - 1]["project_id"])
        services = self.admin._services
        try:
            assignments = snapshot(services.identities.assignments_for(ProjectId(project_id)))
            state = services.project_roles(project_id, str(person.id))
            held = set(state["roles"])
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Роли — {person.display_name}")
            layout = QVBoxLayout(dialog)
            boxes = {}
            for role in ProjectRole:
                box = QCheckBox(CATALOG.roles[role.value].title)
                box.setChecked(role.value in held)
                boxes[role] = box
                layout.addWidget(box)
            inherited = QLabel()
            inherited.setWordWrap(True)
            def refresh():
                selected = {role.value for role, box in boxes.items() if box.isChecked()}
                inherited.setText("Включённые возможности ролей: " + (
                    ", ".join(CATALOG.roles[role].title for role in sorted(CATALOG.closure(selected) - selected)) or "нет"
                ))
            for box in boxes.values():
                box.toggled.connect(refresh)
            refresh()
            layout.addWidget(inherited)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self._save(project_id, person, frozenset(role for role, box in boxes.items() if box.isChecked()),
                           state["revision"], assignments)
        except Exception as exc:
            self.admin._report("Не удалось изменить роли", exc)

    def graph(self) -> None:
        column = self.table.currentColumn()
        if column <= 0:
            return
        project = self.projects[column - 1]
        project_id = str(project["project_id"])
        services = self.admin._services
        def roles(person):
            return frozenset(ProjectRole(value) for value in services.project_roles(project_id, str(person.id))["roles"])
        def change(person, role, enabled):
            state = services.project_roles(project_id, str(person.id))
            assignments = snapshot(services.identities.assignments_for(ProjectId(project_id)))
            desired = set(ProjectRole(value) for value in state["roles"])
            desired.add(role) if enabled else desired.discard(role)
            self._save(project_id, person, frozenset(desired), state["revision"], assignments)
        dialog = RoleManagementDialog(self, project_name=project["name"], principals=self.people,
                                      roles_for=roles, change_role=change, acting_roles=[], on_acting_role=lambda _: None)
        dialog.acting.hide()
        dialog.exec()
