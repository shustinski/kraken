"""Review binary conductor evidence and explicit connections; no proximity inference."""
from __future__ import annotations

import json
from uuid import uuid4

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from kraken_hub.multilayer_dialog import MultilayerDialog
from kraken_manager.domain.artifacts import ArtifactScope


class ElectricalDialog(QDialog):
    _t = MultilayerDialog._t
    _button = MultilayerDialog._button
    _invoke = MultilayerDialog._invoke
    _run = MultilayerDialog._run
    _fill = staticmethod(MultilayerDialog._fill)
    _form = MultilayerDialog._form

    def __init__(self, service, principal, project_id, parent=None, *, language="Русский", background_workers=None):
        super().__init__(parent)
        self.service, self.principal, self.project_id = service, principal, str(project_id)
        self.language, self._background_workers = language, background_workers
        self._worker = self._pending = None
        self.snapshot = self.physical = None
        self.layers, self.versions, self.components = [], {}, {}
        self.setWindowTitle(self._t("Электрическая связность: доказательства", "Electrical connectivity: evidence"))
        self.resize(1100, 800)
        root = QVBoxLayout(self)
        root.addWidget(QLabel(self._t("Близость и совпадение рисунков не создают электрическую связь. Экспорт содержит только частичный проверенный граф.",
            "Proximity and overlapping drawings do not establish connections. Export contains only a partial reviewed graph.")))
        self.summary = QLabel()
        root.addWidget(self.summary)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self._build_components()
        self._build_bindings()
        self._build_export()
        self._build_devices()
        self.status = QLabel()
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        row = QHBoxLayout()
        self.refresh_button = self._button(row, "Обновить", "Refresh", self.refresh)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.rejected.connect(self.reject)
        row.addWidget(self.buttons)
        root.addLayout(row)
        self.refresh()

    def _tab(self, ru, en):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.tabs.addTab(page, self._t(ru, en))
        return layout

    def _build_components(self):
        layout = self._tab("Материал и маски", "Material and masks")
        self.layer, self.mask, self.adjacency, self.foreground = QComboBox(), QComboBox(), QComboBox(), QComboBox()
        self.adjacency.addItem(self._t("4: только стороны пикселей", "4: pixel sides only"), 4)
        self.adjacency.addItem(self._t("8: включая диагонали", "8: includes diagonals"), 8)
        self.foreground.addItem(self._t("255: белый материал", "255: white material"), 255)
        self.foreground.addItem(self._t("0: чёрный материал", "0: black material"), 0)
        self.adjacency.setCurrentIndex(-1)
        self.foreground.setCurrentIndex(-1)
        self.nodata = QLineEdit()
        self.nodata.setPlaceholderText(self._t("Пусто = только 0/255; прозрачность всегда неизвестна", "Blank = only 0/255; transparency is always unknown"))
        self._form(layout, [("Слой", "Layer", self.layer), ("Калиброванная версия binary-маски", "Calibrated binary-mask revision", self.mask),
            ("Смежность пикселей", "Pixel adjacency", self.adjacency), ("Материал", "Foreground material", self.foreground),
            ("Маркер неизвестных пикселей (1..254)", "Nodata marker (1..254)", self.nodata)])
        row = QHBoxLayout()
        self.extract_button = self._button(row, "Выделить компоненты", "Extract components", self.extract)
        layout.addLayout(row)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([self._t("Слой", "Layer"), self._t("Компонент", "Component"),
            self._t("Пикселей", "Pixels"), self._t("Площадь (m²)", "Area (m²)"), self._t("Доказанность", "Evidence state")])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)
        self.material_evidence, self.material_note, self.net_name = QComboBox(), QLineEdit(), QLineEdit()
        self.diagonal_review = QCheckBox(self._t("Диагональные перемычки независимо проверены", "Diagonal bridges independently reviewed"))
        self._form(layout, [("Независимое наблюдение материала", "Independent material observation", self.material_evidence),
            ("Обоснование", "Rationale", self.material_note), ("Каноническое имя сети (необязательно)", "Canonical net identity (optional)", self.net_name)])
        layout.addWidget(self.diagonal_review)
        row = QHBoxLayout()
        self.confirm_material = self._button(row, "Подтвердить материал", "Confirm material", lambda: self.review_material("confirmed"))
        self.reject_material = self._button(row, "Отклонить материал", "Reject material", lambda: self.review_material("rejected"))
        layout.addLayout(row)
        self.layer.currentIndexChanged.connect(self._layer_selected)
        self.table.itemSelectionChanged.connect(self._material_selected)

    def _build_bindings(self):
        layout = self._tab("Явные соединения и выводы", "Explicit connections and terminals")
        self.contact, self.contact_source, self.contact_target = QComboBox(), QComboBox(), QComboBox()
        self.contact_note = QLineEdit()
        self._form(layout, [("Подтверждённый физический контакт", "Reviewed physical contact", self.contact),
            ("Явно выбранный исходный компонент", "Explicit source component", self.contact_source),
            ("Явно выбранный целевой компонент", "Explicit target component", self.contact_target),
            ("Обоснование привязки", "Binding rationale", self.contact_note)])
        row = QHBoxLayout()
        self.bind_contact_button = self._button(row, "Связать контакт", "Bind contact", self.bind_contact)
        self.retract_contact_button = self._button(row, "Отозвать привязку", "Retract binding", self.retract_contact)
        layout.addLayout(row)
        self.terminal_component, self.terminal_evidence, self.direction = QComboBox(), QComboBox(), QComboBox()
        for text in ("input", "output", "inout"):
            self.direction.addItem(text, text)
        self.direction.setCurrentIndex(-1)
        self.instance, self.cell, self.port, self.terminal_note = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        self.instance.setPlaceholderText(self._t("Пусто = порт верхнего уровня", "Blank = top-level port"))
        self._form(layout, [("Компонент вывода", "Terminal component", self.terminal_component),
            ("Имя экземпляра", "Instance name", self.instance), ("Явно заданный тип ячейки", "Explicit cell type", self.cell),
            ("Имя вывода", "Port name", self.port), ("Направление", "Direction", self.direction),
            ("Независимое наблюдение вывода", "Independent terminal observation", self.terminal_evidence),
            ("Обоснование вывода", "Terminal rationale", self.terminal_note)])
        row = QHBoxLayout()
        self.bind_terminal_button = self._button(row, "Привязать вывод", "Bind terminal", self.bind_terminal)
        self.retract_terminal_button = self._button(row, "Отозвать вывод", "Retract terminal", self.retract_terminal)
        layout.addLayout(row)
        self.contact.currentIndexChanged.connect(self._contact_selected)
        self.terminal_component.currentIndexChanged.connect(self._terminal_selected)

    def _build_export(self):
        layout = self._tab("Граф и EDIF", "Graph and EDIF")
        self.diagnostics = QPlainTextEdit()
        self.diagnostics.setReadOnly(True)
        layout.addWidget(self.diagnostics, 1)
        row = QHBoxLayout()
        self.export_button = self._button(row, "Сформировать частичный EDIF", "Generate partial EDIF", self.export)
        layout.addLayout(row)
        self.export_text = QPlainTextEdit()
        self.export_text.setReadOnly(True)
        layout.addWidget(self.export_text, 1)

    def _build_devices(self):
        layout = self._tab("Кандидаты MOS", "MOS candidates")
        layout.addWidget(QLabel(self._t("Геометрия предлагает кандидата. Полярность, source/drain и независимое доказательство задаёт инженер; bulk остаётся неизвестным.",
            "Geometry proposes a candidate. The engineer assigns polarity, source/drain and independent evidence; bulk remains unknown.")))
        self.device_candidate = QComboBox()
        layout.addWidget(self.device_candidate)
        self.device_summary = QPlainTextEdit()
        self.device_summary.setReadOnly(True)
        layout.addWidget(self.device_summary, 1)
        self.device_instance, self.device_note = QLineEdit(), QLineEdit()
        self.device_kind, self.device_source, self.device_drain = QComboBox(), QComboBox(), QComboBox()
        self.device_kind.addItem(self._t("Выберите полярность", "Select polarity"), None)
        for kind in ("NMOS", "PMOS"):
            self.device_kind.addItem(kind, kind)
        self.device_evidence, self.device_evidence_kind = QComboBox(), QComboBox()
        self.device_evidence_kind.addItem(self._t("Выберите категорию доказательства", "Select evidence category"), None)
        for ru, en, value in (("Наблюдение имплантации", "Implant observation", "implant_observation"),
                              ("Документация процесса", "Process documentation", "process_documentation"),
                              ("Инженерная аннотация устройства", "Engineer device annotation", "engineer_device_annotation")):
            self.device_evidence_kind.addItem(self._t(ru, en), value)
        self._form(layout, [("Имя экземпляра", "Instance name", self.device_instance),
            ("Тип по независимому ревью", "Kind from independent review", self.device_kind),
            ("Явно выбранный source", "Explicit source assignment", self.device_source),
            ("Явно выбранный drain", "Explicit drain assignment", self.device_drain),
            ("Независимое доказательство типа", "Independent kind evidence", self.device_evidence),
            ("Категория доказательства", "Evidence category", self.device_evidence_kind),
            ("Обоснование устройства", "Device review rationale", self.device_note)])
        row = QHBoxLayout()
        self.realize_device_button = self._button(row, "Подтвердить явную модель MOS", "Confirm explicit MOS model", self.realize_device)
        self.retract_device_button = self._button(row, "Отозвать модель устройства", "Retract device model", self.retract_device)
        layout.addLayout(row)
        self.device_candidate.currentIndexChanged.connect(self._device_selected)

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
            snapshot = self.service.get_electrical_model(self.project_id)
            physical = self.service.get_multilayer_model(self.project_id)
            layers = list(self.service.list_layers(self.project_id))
            versions = {}
            for series in self.service.list_artifact_series(self.project_id):
                if series.scope is ArtifactScope.LAYER_ATTACHMENT and not series.archived:
                    versions.setdefault(str(series.layer_id), []).extend(v for v in self.service.artifact_versions(
                        self.project_id, series.id) if v.blob is not None)
            return snapshot, physical, layers, versions

        def loaded(result):
            self.snapshot, self.physical, self.layers, self.versions = result
            self._render()

        self._run(read, loaded)

    def _version_rows(self, layer_id, excluded_hash=None):
        return [(f"{v.filename} · {str(v.id)[:8]} · SHA256 {v.sha256[:12]}", str(v.id))
                for v in self.versions.get(layer_id, []) if v.sha256 != excluded_hash]

    def _component_rows(self, layer_id=None):
        names = {str(layer.id): layer.name for layer in self.layers}
        return [(f"{names[e['layer_id']]} · {c['pixel_count']} px · {identifier[:8]}", identifier)
                for identifier, (c, e) in self.components.items() if layer_id is None or e["layer_id"] == layer_id]

    def _render(self):
        graph = self.snapshot["graph"]
        self.export_text.clear()
        self.summary.setText(self._t("Частичный граф", "Partial graph") +
            f" · r{self.snapshot['graph_electrical_revision']} · nets={len(graph['nets'])}; connections={len(graph['connections'])}; "
            f"unknown={len(graph['unknowns'])}; conflicts={len(graph['conflicts'])}")
        self._fill(self.layer, [(layer.name, str(layer.id)) for layer in self.layers])
        self._layer_selected()
        previous = self._selected_component()
        selected_id = previous[0]["id"] if previous else None
        self.components = {c["id"]: (c, e) for e in self.snapshot["electrical"]["extractions"] for c in e["components"]}
        nodes = {node["component_id"]: node for node in graph["nodes"]}
        names = {str(layer.id): layer.name for layer in self.layers}
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.components))
        for row, (identifier, (component, extraction)) in enumerate(self.components.items()):
            node = nodes.get(identifier, {})
            values = [names[extraction["layer_id"]], identifier[:8], str(component["pixel_count"]),
                      str(node.get("area_m2", "?")), node.get("state", "historical")]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, identifier)
                self.table.setItem(row, column, item)
            if identifier == selected_id:
                self.table.selectRow(row)
        if self.table.rowCount() and not self.table.selectedItems():
            self.table.selectRow(0)
        self.table.blockSignals(False)
        self._material_selected()
        self._fill(self.terminal_component, self._component_rows())
        assessments = {c["id"]: c for c in self.physical["assessment"]["contacts"]}
        self._fill(self.contact, [(f"{names[c['source_layer_id']]} → {names[c['target_layer_id']]} · {c['id'][:8]}", c["id"])
            for c in self.physical["model"]["contacts"] if assessments[c["id"]]["electrical_connectivity"] == "engineer_confirmed"])
        self._contact_selected()
        self._terminal_selected()
        self.diagnostics.setPlainText(json.dumps({"connections": graph["connections"], "conflicts": graph["conflicts"],
                                                   "unknowns": graph["unknowns"], "devices": graph.get("devices", [])}, ensure_ascii=False, indent=2))
        self._fill(self.device_candidate, [(f"{names.get(c['gate_layer_id'], '?')} → {names.get(c['diffusion_layer_id'], '?')} · "
            f"{c['id'][:8]} · {c['state']}", c["id"]) for c in self.snapshot.get("device_candidates", [])])
        self._device_selected()

    def _layer_selected(self):
        calibration = next((c for c in self.physical["model"]["calibrations"] if c["layer_id"] == self.layer.currentData()), None) if self.physical else None
        identifier = calibration["artifact"]["version_id"] if calibration else None
        self._fill(self.mask, [(label, value) for label, value in self._version_rows(self.layer.currentData()) if value == identifier])

    def _selected_component(self):
        row = self.table.currentRow()
        identifier = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) if row >= 0 and self.table.item(row, 0) else None
        return self.components.get(identifier)

    def _material_selected(self):
        selected = self._selected_component()
        self._fill(self.material_evidence, self._version_rows(selected[1]["layer_id"], selected[1]["mask"]["sha256"]) if selected else [])

    def _contact_selected(self):
        contact = next((c for c in self.physical["model"]["contacts"] if c["id"] == self.contact.currentData()), None) if self.physical else None
        self._fill(self.contact_source, self._component_rows(contact["source_layer_id"]) if contact else [])
        self._fill(self.contact_target, self._component_rows(contact["target_layer_id"]) if contact else [])

    def _terminal_selected(self):
        selected = self.components.get(self.terminal_component.currentData())
        self._fill(self.terminal_evidence, self._version_rows(selected[1]["layer_id"], selected[1]["mask"]["sha256"]) if selected else [])

    def _device_selected(self):
        candidate = next((c for c in self.snapshot.get("device_candidates", []) if c["id"] == self.device_candidate.currentData()), None) if self.snapshot else None
        self.device_summary.setPlainText(json.dumps(candidate or {"state": "no_observed_supported_candidate"}, ensure_ascii=False, indent=2))
        ids = set(candidate["diffusion_terminal_candidates"]) if candidate else set()
        rows = [(label, identifier) for label, identifier in self._component_rows() if identifier in ids]
        self._fill(self.device_source, [(self._t("Выберите source", "Select source"), None), *rows])
        self._fill(self.device_drain, [(self._t("Выберите drain", "Select drain"), None), *rows])
        evidence = []
        if candidate:
            excluded = {candidate["gate_mask"]["sha256"], candidate["diffusion_mask"]["sha256"]}
            for layer_id in (candidate["gate_layer_id"], candidate["diffusion_layer_id"]):
                evidence.extend((f"{v.filename} · {str(v.id)[:8]} · SHA256 {v.sha256[:12]}", str(v.id))
                                for v in self.versions.get(layer_id, []) if v.sha256 not in excluded)
        self._fill(self.device_evidence, [(self._t("Выберите доказательство", "Select evidence"), None), *evidence])

    def _send(self, operation, arguments):
        if self.snapshot is None or self._worker is not None:
            return
        physical_revision = self.snapshot["physical_model_revision"]
        signature = json.dumps([operation, arguments, physical_revision], sort_keys=True)
        if self._pending is None or self._pending[0] != signature:
            self._pending = (signature, str(uuid4()))
        key, revision = self._pending[1], self.snapshot["electrical"]["revision"]

        def command():
            result = self.service.mutate_electrical_model(project_id=self.project_id, principal=self.principal,
                expected_revision=revision, expected_physical_revision=physical_revision,
                operation=operation, arguments=arguments, idempotency_key=key)
            return result, self.service.get_multilayer_model(self.project_id)

        def saved(result):
            self.snapshot, self.physical = result
            self._pending = None
            self._render()

        self._run(command, saved)

    def extract(self):
        self._send("extract_conductors", {"layer_id": self.layer.currentData(), "mask_version_id": self.mask.currentData(),
            "connectivity": self.adjacency.currentData(), "foreground_value": self.foreground.currentData(),
            "nodata_value": int(self.nodata.text()) if self.nodata.text().strip() else None})

    def review_material(self, decision):
        selected = self._selected_component()
        if selected is None:
            raise ValueError(self._t("Выберите компонент", "Select a component"))
        evidence = self.material_evidence.currentData()
        self._send("review_components", {"component_ids": [selected[0]["id"]], "decision": decision,
            "evidence_version_ids": [evidence] if decision == "confirmed" and evidence else [], "note": self.material_note.text(),
            "declared_net_name": self.net_name.text().strip() or None, "accepts_diagonal_bridges": self.diagonal_review.isChecked()})

    def bind_contact(self):
        self._send("bind_contact", {"contact_id": self.contact.currentData(), "source_component_id": self.contact_source.currentData(),
            "target_component_id": self.contact_target.currentData(), "note": self.contact_note.text()})

    def retract_contact(self):
        self._send("retract_contact_binding", {"contact_id": self.contact.currentData(), "note": self.contact_note.text()})

    def bind_terminal(self):
        evidence = self.terminal_evidence.currentData()
        self._send("bind_terminal", {"component_id": self.terminal_component.currentData(), "port": self.port.text(),
            "direction": self.direction.currentData(), "instance": self.instance.text().strip() or None,
            "cell": self.cell.text().strip() or None, "evidence_version_ids": [evidence] if evidence else [], "note": self.terminal_note.text()})

    def retract_terminal(self):
        self._send("retract_terminal", {"instance": self.instance.text().strip() or None, "port": self.port.text(),
                                         "note": self.terminal_note.text()})

    def export(self):
        def loaded(result):
            self.export_text.setPlainText(result["content"])
            self.summary.setText(self.summary.text() + " · SHA256 " + result["sha256"])

        self._run(lambda: self.service.export_electrical_model(self.project_id), loaded)

    def realize_device(self):
        evidence = self.device_evidence.currentData()
        self._send("realize_device", {"candidate_id": self.device_candidate.currentData(), "instance": self.device_instance.text(),
            "kind": self.device_kind.currentData(), "source_component_id": self.device_source.currentData(),
            "drain_component_id": self.device_drain.currentData(), "evidence_version_ids": [evidence] if evidence else [],
            "evidence_kind": self.device_evidence_kind.currentData(), "note": self.device_note.text()})

    def retract_device(self):
        self._send("retract_device", {"instance": self.device_instance.text(), "note": self.device_note.text()})
