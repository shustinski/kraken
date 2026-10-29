"""Physical layers, explicit control points and independent contact review."""
from __future__ import annotations

import json
from uuid import uuid4

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)

from kraken_hub.presentation.qt.parts.layer_create_thread import _BackgroundCallThread
from kraken_manager.domain.artifacts import ArtifactScope
from kraken_manager.domain.project import LayerType


class MultilayerDialog(QDialog):
    def __init__(self, service, principal, project_id, parent=None, *, language="Русский", background_workers=None):
        super().__init__(parent)
        self.service, self.principal, self.project_id = service, principal, str(project_id)
        self.language, self._background_workers = language, background_workers
        self._worker = None
        self._pending = None
        self.snapshot = None
        self.layers, self.versions = [], {}
        self.setWindowTitle(self._t("Физические слои и контакты", "Physical layers and contacts"))
        self.resize(1050, 760)
        root = QVBoxLayout(self)
        self.summary = QLabel()
        root.addWidget(self.summary)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self._build_stack()
        self._build_calibration()
        self._build_registration()
        self._build_contacts()
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.status)
        row = QHBoxLayout()
        self.refresh_button = self._button(row, "Обновить", "Refresh", self.refresh)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.rejected.connect(self.reject)
        row.addWidget(self.buttons)
        root.addLayout(row)
        self.refresh()

    def _t(self, ru, en):
        return en if self.language == "English" else ru

    def _tab(self, ru, en):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.tabs.addTab(page, self._t(ru, en))
        return layout

    def _button(self, row, ru, en, callback):
        button = QPushButton(self._t(ru, en))
        button.clicked.connect(lambda _checked=False: self._invoke(callback))
        row.addWidget(button)
        return button

    def _invoke(self, callback):
        try:
            callback()
        except Exception as exc:
            self.status.setText(self._t("Ошибка: ", "Error: ") + str(exc))

    def _form(self, layout, rows):
        form = QFormLayout()
        for ru, en, control in rows:
            form.addRow(self._t(ru, en), control)
        layout.addLayout(form)

    def _build_stack(self):
        layout = self._tab("Стек", "Stack")
        layout.addWidget(QLabel(self._t("Задайте физический Z и толщину в метрах. Порядок кадров не определяет физический стек.",
                                       "Specify physical Z and thickness in metres. Frame order does not establish a physical stack.")))
        self.stack = QTableWidget(0, 5)
        self.stack.setHorizontalHeaderLabels([self._t("Включить", "Include"), self._t("Слой", "Layer"),
            "Z (m)", self._t("Толщина (m)", "Thickness (m)"), self._t("Проводящий", "Conductive")])
        self.stack.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.stack)
        row = QHBoxLayout()
        self.stack_button = self._button(row, "Сохранить физический стек", "Save physical stack", self.save_stack)
        layout.addLayout(row)

    def _build_calibration(self):
        layout = self._tab("Калибровка", "Calibration")
        self.calibration_layer, self.calibration_version = QComboBox(), QComboBox()
        self.scale_x, self.scale_y = QLineEdit(), QLineEdit()
        self.origin_x, self.origin_y = QLineEdit("0"), QLineEdit("0")
        self.y_axis = QComboBox()
        self.y_axis.addItem(self._t("Вниз", "Down"), "down")
        self.y_axis.addItem(self._t("Вверх", "Up"), "up")
        self.uncertainty = QLineEdit()
        self.uncertainty.setPlaceholderText(self._t("Пусто = неизвестно", "Blank = unknown"))
        self._form(layout, [
            ("Слой", "Layer", self.calibration_layer),
            ("Версия вложения слоя", "Layer attachment revision", self.calibration_version),
            ("Метров на пиксель X", "Metres per pixel X", self.scale_x),
            ("Метров на пиксель Y", "Metres per pixel Y", self.scale_y),
            ("Физическое начало X (m)", "Physical origin X (m)", self.origin_x),
            ("Физическое начало Y (m)", "Physical origin Y (m)", self.origin_y),
            ("Направление пиксельной Y", "Pixel Y direction", self.y_axis),
            ("Заявленная неопределённость (m)", "Declared uncertainty (m)", self.uncertainty),
        ])
        row = QHBoxLayout()
        self.calibrate_button = self._button(row, "Сохранить калибровку", "Save calibration", self.calibrate)
        layout.addLayout(row)
        layout.addStretch(1)
        self.calibration_layer.currentIndexChanged.connect(self._calibration_selected)

    def _build_registration(self):
        layout = self._tab("Совмещение", "Registration")
        self.source_layer, self.target_layer = QComboBox(), QComboBox()
        self.tolerance = QLineEdit()
        self._form(layout, [("Исходный слой", "Source layer", self.source_layer),
                            ("Целевой слой", "Target layer", self.target_layer),
                            ("Максимальная невязка (m)", "Maximum control residual (m)", self.tolerance)])
        self.points = QTableWidget(0, 5)
        self.points.setHorizontalHeaderLabels(["Source X (px)", "Source Y (px)", "Target X (px)", "Target Y (px)",
                                               self._t("Неопределённость (m)", "Uncertainty (m)")])
        layout.addWidget(self.points, 1)
        row = QHBoxLayout()
        self._button(row, "Добавить контрольную точку", "Add control point", self.add_point)
        self._button(row, "Удалить выбранную", "Remove selected", lambda: self.points.removeRow(self.points.currentRow()))
        self.register_button = self._button(row, "Рассчитать affine", "Fit affine", self.register)
        layout.addLayout(row)
        self.fit_summary = QLabel()
        self.fit_summary.setWordWrap(True)
        layout.addWidget(self.fit_summary)

    def _build_contacts(self):
        layout = self._tab("Контакты", "Contacts")
        layout.addWidget(QLabel(self._t("Совмещение не доказывает электрическое соединение. Подтверждение требует независимой версии слоя контактов и ревью.",
                                       "Registration does not prove an electrical connection. Confirmation requires an independent contact-layer revision and review.")))
        self.registration, self.contact_layer = QComboBox(), QComboBox()
        self.contact_x, self.contact_y, self.contact_radius = QLineEdit(), QLineEdit(), QLineEdit()
        self._form(layout, [("Ревизия совмещения", "Registration revision", self.registration),
            ("Слой контактов", "Contact layer", self.contact_layer),
            ("X в целевой физической системе (m)", "Target physical X (m)", self.contact_x),
            ("Y в целевой физической системе (m)", "Target physical Y (m)", self.contact_y),
            ("Радиус наблюдения (m)", "Observation radius (m)", self.contact_radius)])
        row = QHBoxLayout()
        self.propose_button = self._button(row, "Добавить кандидат", "Propose contact", self.propose_contact)
        layout.addLayout(row)
        self.contacts = QTableWidget(0, 4)
        self.contacts.setHorizontalHeaderLabels([self._t("Контакт", "Contact"), self._t("Проводящие слои", "Conductor layers"),
                                                self._t("Результат", "Effective state"), self._t("Ревьюер", "Reviewer")])
        self.contacts.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.contacts.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.contacts.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.contacts, 1)
        self.evidence, self.note = QComboBox(), QLineEdit()
        self._form(layout, [("Независимое доказательство слоя контактов", "Independent contact-layer evidence", self.evidence),
                            ("Обоснование ревью", "Review rationale", self.note)])
        row = QHBoxLayout()
        self.confirm_button = self._button(row, "Подтвердить контакт", "Confirm contact", lambda: self.review("confirmed"))
        self.reject_button = self._button(row, "Отклонить контакт", "Reject contact", lambda: self.review("rejected"))
        layout.addLayout(row)
        self.contact_layer.currentIndexChanged.connect(self._contact_versions)
        self.contacts.itemSelectionChanged.connect(self._contact_selected)

    def _run(self, call, loaded):
        if self._worker is not None:
            return
        self.tabs.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.buttons.setEnabled(False)
        self.status.setText(self._t("Выполнение…", "Working…"))
        worker = _BackgroundCallThread(call, parent=self)
        self._worker = worker
        if self._background_workers is not None:
            self._background_workers.add(worker)

        def succeeded(result):
            try:
                loaded(result)
                self.status.setText(self._t("Сохранено / загружено", "Saved / loaded"))
            except Exception as exc:
                self.status.setText(str(exc))

        def finished():
            self._worker = None
            self.tabs.setEnabled(True)
            self.refresh_button.setEnabled(True)
            self.buttons.setEnabled(True)
            if self._background_workers is not None:
                self._background_workers.discard(worker)

        worker.succeeded.connect(succeeded)
        worker.failed.connect(lambda error: self.status.setText(self._t("Ошибка: ", "Error: ") + error))
        worker.finished.connect(finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def reject(self):
        if self._worker is None:
            super().reject()

    def closeEvent(self, event):
        if self._worker is not None:
            event.ignore()
        else:
            super().closeEvent(event)

    def refresh(self):
        def read():
            model = self.service.get_multilayer_model(self.project_id)
            layers = list(self.service.list_layers(self.project_id))
            versions = {}
            for series in self.service.list_artifact_series(self.project_id):
                if series.scope is ArtifactScope.LAYER_ATTACHMENT and not series.archived:
                    versions.setdefault(str(series.layer_id), []).extend(
                        v for v in self.service.artifact_versions(self.project_id, series.id) if v.blob is not None)
            return model, layers, versions

        def loaded(result):
            self.snapshot, self.layers, self.versions = result
            self._render()

        self._run(read, loaded)

    def _names(self):
        return {str(layer.id): layer.name for layer in self.layers}

    @staticmethod
    def _fill(combo, rows):
        selected = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for label, value in rows:
            combo.addItem(label, value)
        index = combo.findData(selected)
        combo.setCurrentIndex(index if index >= 0 else (0 if combo.count() else -1))
        combo.blockSignals(False)

    def _render(self):
        model = self.snapshot["model"]
        self.summary.setText(self._t("Физическая ревизия", "Physical revision") + f" {model['revision']} · " +
                             self._t("Единицы: метры. Автопересчёт отключён.", "Units: metres. Automatic recalculation disabled."))
        names = self._names()
        named = [(layer.name, str(layer.id)) for layer in self.layers]
        for combo in (self.calibration_layer, self.source_layer, self.target_layer):
            self._fill(combo, named)
        self._fill(self.contact_layer, [(layer.name, str(layer.id)) for layer in self.layers if layer.type is LayerType.CONTACT])
        stack = {layer["layer_id"]: layer for layer in model["layers"]}
        self.stack.setRowCount(len(self.layers))
        for row, layer in enumerate(self.layers):
            physical = stack.get(str(layer.id))
            included, conductive = QCheckBox(), QCheckBox()
            included.setChecked(physical is not None)
            conductive.setChecked(bool(physical and physical["conductive"]))
            self.stack.setCellWidget(row, 0, included)
            name = QTableWidgetItem(layer.name)
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            name.setData(Qt.ItemDataRole.UserRole, str(layer.id))
            self.stack.setItem(row, 1, name)
            for column, key in ((2, "z_m"), (3, "thickness_m")):
                self.stack.setItem(row, column, QTableWidgetItem(str(physical[key]) if physical else ""))
            self.stack.setCellWidget(row, 4, conductive)
        self._calibration_selected()
        registrations = model["registrations"]
        states = {(r["id"], r["revision"]): r for r in self.snapshot["assessment"]["registrations"]}
        self._fill(self.registration, [
            (f"{names.get(r['source']['layer_id'], '?')} → {names.get(r['target']['layer_id'], '?')} · "
             f"r{r['revision']} · {r['fit']['status']} · " + ("current" if states[r['id'], r['revision']]["current"] else "stale"),
             (r["id"], r["revision"])) for r in registrations
        ])
        if registrations:
            fit = registrations[-1]["fit"]
            self.fit_summary.setText(f"{fit['status']}: {fit['reason']}\nRMS={fit['rms_m']} m; max={fit['max_residual_m']} m; "
                                     f"condition={fit['design_condition']}; control bound={fit['control_error_bound_m']} m\n" +
                                     self._t("Оценка ошибки относится только к контрольным точкам.", "Error estimate applies only at supplied control points."))
            self.registration.setCurrentIndex(self.registration.count() - 1)
        contact_states = {c["id"]: c for c in self.snapshot["assessment"]["contacts"]}
        self.contacts.blockSignals(True)
        self.contacts.setRowCount(len(model["contacts"]))
        for row, contact in enumerate(model["contacts"]):
            values = [contact["id"][:8], f"{names[contact['source_layer_id']]} → {names[contact['target_layer_id']]}",
                      contact_states[contact["id"]]["effective_state"], contact["reviewed_by"] or "—"]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, contact["id"])
                self.contacts.setItem(row, column, item)
        if self.contacts.rowCount():
            self.contacts.selectRow(self.contacts.rowCount() - 1)
        self.contacts.blockSignals(False)
        self._contact_selected()
        self._contact_versions()

    def _calibration_selected(self):
        layer = self.calibration_layer.currentData()
        self._fill(self.calibration_version, [(f"{v.filename} · {str(v.id)[:8]} · SHA256 {v.sha256[:12]}", str(v.id))
                                              for v in self.versions.get(layer, [])])
        calibration = next((c for c in self.snapshot["model"]["calibrations"] if c["layer_id"] == layer), None) if self.snapshot else None
        if calibration:
            self.scale_x.setText(str(calibration["pixel_size_m"][0]))
            self.scale_y.setText(str(calibration["pixel_size_m"][1]))
            self.origin_x.setText(str(calibration["origin_m"][0]))
            self.origin_y.setText(str(calibration["origin_m"][1]))
            self.y_axis.setCurrentIndex(self.y_axis.findData(calibration["y_axis"]))
            self.uncertainty.setText("" if calibration["uncertainty_m"] is None else str(calibration["uncertainty_m"]))
            self.calibration_version.setCurrentIndex(self.calibration_version.findData(calibration["artifact"]["version_id"]))
        else:
            self.scale_x.clear()
            self.scale_y.clear()
            self.origin_x.setText("0")
            self.origin_y.setText("0")
            self.y_axis.setCurrentIndex(self.y_axis.findData("down"))
            self.uncertainty.clear()

    def _contact_versions(self):
        self._fill(self.evidence, [(f"{v.filename} · {str(v.id)[:8]} · SHA256 {v.sha256[:12]}", str(v.id))
                                  for v in self.versions.get(self.contact_layer.currentData(), [])])

    def _contact_selected(self):
        row = self.contacts.currentRow()
        if row >= 0 and self.contacts.item(row, 0) and self.snapshot:
            identifier = self.contacts.item(row, 0).data(Qt.ItemDataRole.UserRole)
            contact = next(c for c in self.snapshot["model"]["contacts"] if c["id"] == identifier)
            self.contact_layer.setCurrentIndex(self.contact_layer.findData(contact["contact_layer_id"]))

    def _send(self, operation, arguments):
        if self._worker is not None or self.snapshot is None:
            return
        signature = json.dumps([operation, arguments], sort_keys=True)
        if self._pending is None or self._pending[0] != signature:
            self._pending = (signature, str(uuid4()))
        key, expected = self._pending[1], self.snapshot["model"]["revision"]

        def saved(result):
            self.snapshot = result
            self._pending = None
            self._render()

        self._run(lambda: self.service.mutate_multilayer_model(
            project_id=self.project_id, principal=self.principal, expected_revision=expected,
            operation=operation, arguments=arguments, idempotency_key=key,
        ), saved)

    def save_stack(self):
        layer_by_id = {str(layer.id): layer for layer in self.layers}
        layers = []
        for row in range(self.stack.rowCount()):
            if not self.stack.cellWidget(row, 0).isChecked():
                continue
            identifier = self.stack.item(row, 1).data(Qt.ItemDataRole.UserRole)
            layers.append({"layer_id": identifier, "layer_type": layer_by_id[identifier].type.value,
                           "z_m": float(self.stack.item(row, 2).text()), "thickness_m": float(self.stack.item(row, 3).text()),
                           "conductive": self.stack.cellWidget(row, 4).isChecked()})
        self._send("set_stack", {"layers": sorted(layers, key=lambda layer: layer["z_m"])})

    def calibrate(self):
        self._send("calibrate", {"layer_id": self.calibration_layer.currentData(),
            "artifact_version_id": self.calibration_version.currentData(),
            "pixel_size_m": [float(self.scale_x.text()), float(self.scale_y.text())],
            "origin_m": [float(self.origin_x.text()), float(self.origin_y.text())], "y_axis": self.y_axis.currentData(),
            "uncertainty_m": float(self.uncertainty.text()) if self.uncertainty.text().strip() else None})

    def add_point(self):
        row = self.points.rowCount()
        self.points.insertRow(row)
        for column in range(5):
            self.points.setItem(row, column, QTableWidgetItem(""))

    def register(self):
        points = []
        for row in range(self.points.rowCount()):
            values = [self.points.item(row, c).text().strip() for c in range(5)]
            points.append({"source_xy": [float(v) for v in values[:2]], "target_xy": [float(v) for v in values[2:4]],
                           "uncertainty_m": float(values[4]) if values[4] else None})
        self._send("register_affine", {"source_layer_id": self.source_layer.currentData(),
            "target_layer_id": self.target_layer.currentData(), "control_points": points,
            "max_error_m": float(self.tolerance.text())})

    def propose_contact(self):
        key = self.registration.currentData()
        reg = next(r for r in self.snapshot["model"]["registrations"] if (r["id"], r["revision"]) == key)
        self._send("propose_contact", {"source_layer_id": reg["source"]["layer_id"],
            "target_layer_id": reg["target"]["layer_id"], "contact_layer_id": self.contact_layer.currentData(),
            "registration_id": reg["id"], "registration_revision": reg["revision"],
            "position_target_m": [float(self.contact_x.text()), float(self.contact_y.text())],
            "radius_m": float(self.contact_radius.text())})

    def review(self, decision):
        row = self.contacts.currentRow()
        if row < 0:
            raise ValueError(self._t("Выберите контакт", "Select a contact"))
        evidence = self.evidence.currentData()
        self._send("review_contact", {"contact_id": self.contacts.item(row, 0).data(Qt.ItemDataRole.UserRole),
            "decision": decision, "evidence_version_ids": [evidence] if evidence else [], "review_note": self.note.text()})
