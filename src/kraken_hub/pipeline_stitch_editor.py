"""Layer-step editor uses geometry and producer ports, never fake frame IDs."""
from uuid import uuid4

from PyQt6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QSpinBox, QVBoxLayout
from kraken_manager.domain.pipeline import LayerPipelineDefinition, PipelineResources
from kraken_manager.domain.pipeline_layer import PipelineLayerContent, PipelineLayerOperation


def _port_index(combo, value):
    # Qt's QVariant comparison does not reliably find Python tuple item data.
    for index in range(combo.count()):
        stored = combo.itemData(index)
        if stored is not None and tuple(stored) == tuple(value):
            return index
    return -1


def edit_stitch_step(parent, snapshot, existing=None):
    plans = [p for p in snapshot["model"]["stitches"] if p["layer_id"] == parent.layer_id]
    if not plans:
        parent.status.setText(parent._t("Сначала создайте геометрию сшивки слоя", "Create layer stitching geometry first"))
        return
    dialog = QDialog(parent)
    dialog.setObjectName("pipelineStitchEditor")
    dialog.setWindowTitle(parent._t("Сшивка в технологическом процессе", "Stitch workflow step"))
    layout = QVBoxLayout(dialog)
    form = QFormLayout()
    title = QLineEdit(existing["title"] if existing else parent._t("Мозаика слоя", "Layer mosaic"))
    geometry, producer, mode = QComboBox(), QComboBox(), QComboBox()
    geometry.setObjectName("pipelineStitchGeometry")
    producer.setObjectName("pipelineStitchProducer")
    for p in plans:
        geometry.addItem(f"r{p['revision']} · {len(p['tiles'])} frames · {p['id'][:8]}", (p["id"], p["revision"]))
    geometry.setCurrentIndex(geometry.count()-1)
    producer.addItem(parent._t("Исходные кадры геометрии", "Original geometry images"), None)
    for op in parent._operations:
        if op.get("kind") != "layer" and op["output_kind"] in {"binary_mask", "confidence", "vector"}:
            producer.addItem(op["title"], (op["id"], op["output_kind"]))
    # An existing representation binding remains explicit even when it is not a
    # producer port offered by this editor. Opening and saving must preserve it.
    existing_content = existing.get("content") if existing else None
    if existing_content and existing_content.get("representation_id"):
        producer.addItem(parent._t("Закреплённое представление: ", "Bound representation: ") +
            existing_content["representation_id"], (existing_content["representation_id"], existing_content["kind"], "representation"))
    for label, value in (("RGB", "rgb"), (parent._t("Серый", "Gray"), "gray"), (parent._t("Метки", "Labels"), "labels")):
        mode.addItem(label, value)
    memory = QSpinBox()
    memory.setRange(0, 1024*1024)
    memory.setValue(512 if existing is None else existing["resources"]["memory_mb"])
    approval, quality = QCheckBox(), QCheckBox()
    quality.setChecked(True if existing is None else existing["quality_gate"])
    approval.setChecked(False if existing is None else existing["approval_required"])
    for ru, en, widget in (("Название", "Title", title), ("Геометрия", "Geometry", geometry),
        ("Содержимое", "Content", producer), ("Пиксели", "Pixels", mode), ("RAM, MB", "RAM, MB", memory),
        ("Подтверждение инженера", "Engineer approval", approval), ("Контроль качества входов", "Input quality gate", quality)):
        form.addRow(parent._t(ru, en), widget)
    layout.addLayout(form)
    status = QLabel()
    status.setObjectName("pipelineStitchValidation")
    status.setWordWrap(True)
    layout.addWidget(status)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    layout.addWidget(buttons)

    def source_changed():
        selected = producer.currentData()
        mode.setEnabled(not selected or selected[1] != "binary_mask")
        if selected and selected[1] != "vector":
            mode.setCurrentIndex(mode.findData("labels" if selected[1] == "binary_mask" else "gray"))

    producer.currentIndexChanged.connect(source_changed)
    if existing:
        geometry.setCurrentIndex(_port_index(geometry, (existing["stitch_id"], existing["stitch_revision"])))
        content = existing.get("content")
        if content and content.get("producer_id"):
            value = (content["producer_id"], content["kind"])
            index = _port_index(producer, value)
            if index < 0:
                producer.addItem(parent._t("Недоступный этап: ", "Unavailable step: ") + content["producer_id"], value)
                index = producer.count() - 1
            producer.setCurrentIndex(index)
        elif content and content.get("representation_id"):
            producer.setCurrentIndex(producer.count() - 1)
        mode.setCurrentIndex(mode.findData(existing["parameters"].get("mode", "rgb")))
        source_changed()

    def accept():
        try:
            if geometry.currentData() is None:
                raise ValueError(parent._t("Выберите доступную геометрию", "Select available geometry"))
            stitch_id, revision = geometry.currentData()
            selected = producer.currentData()
            dependencies = list(existing.get("dependencies", ())) if existing else []
            is_producer = selected is not None and len(selected) == 2
            if is_producer and selected[0] not in dependencies:
                dependencies.append(selected[0])
            parameters = dict(existing["parameters"]) if existing else {"block_size": 256}
            parameters["mode"] = mode.currentData()
            resources = dict(existing["resources"]) if existing else {"cpu": 1, "gpu": 0}
            resources["memory_mb"] = memory.value()
            operation = PipelineLayerOperation(existing["id"] if existing else str(uuid4()), title.text(), "layer.stitch.v1",
                existing["algorithm_version"] if existing else "1.0.0",
                stitch_id, revision, dependencies=tuple(dependencies), content=None if selected is None else
                PipelineLayerContent(selected[1], **{"producer_id" if is_producer else "representation_id": selected[0]}), parameters=parameters,
                resources=PipelineResources(**resources), approval_required=approval.isChecked(), quality_gate=quality.isChecked())
            value = operation.to_dict()
            operations = [value if old["id"] == value["id"] else old for old in parent._operations] if existing else [*parent._operations, value]
            LayerPipelineDefinition.from_dict(parent._definition(operations))
        except (ValueError, TypeError) as exc:
            status.setText(str(exc))
            return
        parent._operations, parent._dirty = operations, True
        parent._redraw()
        dialog.accept()

    buttons.accepted.connect(accept)
    buttons.rejected.connect(dialog.reject)
    dialog.exec()
