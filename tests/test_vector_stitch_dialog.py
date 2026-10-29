"""Shown Qt vector producer binding and bounded mosaic uncertainty display."""
from uuid import uuid4

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QComboBox, QDialogButtonBox
from shapely import box

from kraken_hub.presentation.qt.computed_stitch_dialog import MosaicBlockDialog
from kraken_hub.presentation.qt.layer_pipeline_dialog import LayerPipelineDialog
from kraken_hub.presentation.qt.pipeline_stitch_editor import edit_stitch_step
from kraken_manager.domain.multilayer import MultilayerModel
from kraken_manager.domain.pipeline import PipelineOperation
from kraken_manager.domain.project import RepresentationKind
from kraken_manager.infrastructure.mosaic import write_mosaic
from kraken_manager.infrastructure.stitching import StitchRenderer
from kraken_manager.infrastructure.stitch_vectors import VectorStitchIndex
from tests.test_stitch_plan_workflow import geometry as geometry


def test_shown_editor_selects_actual_vector_producer_and_saves_typed_edge(qtbot, geometry):
    local, session, project, layer, versions, args, command = geometry
    command("plan_layer_stitch", args)
    representation = local._projection(project.id).get_artifact_series(versions[0].series_id).representation_id
    vector = local.create_representation(principal=session.principal, project=local.get_project(project.id),
        layer=local.list_layers(project.id)[0], name="Vectors", kind=RepresentationKind.VECTOR,
        source_image_representation_id=representation, active=False, idempotency_key="vectors")
    producer = PipelineOperation(str(uuid4()), "Contour producer", "frames.vectorize.v1", "0.9.10",
        str(representation), str(vector.id), "image", "vector", ((1, 1), (2, 1)))
    dialog = LayerPipelineDialog(local, session.principal, project.id, layer.id)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog._worker is None)
    dialog._operations = [producer.to_dict()]
    dialog.capacity_memory.setValue(512)
    observed = []
    def accept():
        editor = QApplication.activeModalWidget()
        content = editor.findChild(QComboBox, "pipelineStitchProducer")
        values = [content.itemData(i) for i in range(content.count())]
        observed.extend(values)
        content.setCurrentIndex(1)
        editor.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
        if editor.isVisible():
            editor.reject()
    QTimer.singleShot(20, accept)
    edit_stitch_step(dialog, local.get_multilayer_model(project.id))
    assert (producer.id, "vector") in observed
    assert len(dialog._operations) == 2
    child = dialog._operations[-1]
    assert child["content"] == {"kind": "vector", "producer_id": producer.id} and child["dependencies"] == [producer.id]
    dialog._dirty = True
    dialog.save_definition()
    qtbot.waitUntil(lambda: dialog._worker is None)
    assert local.get_layer_pipeline(project.id, layer.id)["definition"]["operations"][-1] == child
    dialog.close()


def test_shown_block_viewer_marks_vector_disagreement_as_unverified(qtbot, geometry, tmp_path):
    _, _, _, _, _, args, command = geometry
    plan = MultilayerModel.from_dict(command("plan_layer_stitch", args)).stitches[-1]
    renderer = StitchRenderer(plan)
    bindings = [{"frame_id": tile.frame_id, "artifact_version_id": tile.calibration.artifact.version_id,
        "sha256": tile.calibration.artifact.sha256} for tile in plan.tiles]
    path = tmp_path / "mosaic.sqlite3"
    with VectorStitchIndex(tmp_path / "index.sqlite3", renderer) as vectors:
        for tile, binding in zip(plan.tiles, bindings):
            vectors.add(tile, binding, [("conductor", box(2, 2, 10, 8))])
        write_mosaic(path, renderer, lambda tile: np.zeros((tile.height, tile.width, 3), dtype=np.uint8),
            job_id="shown-vector", content_inputs=bindings, vector_index=vectors)
    dialog = MosaicBlockDialog(path)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog._worker is None)
    assert not dialog.image.pixmap().isNull()
    assert dialog._metadata["vector_stitch"]["conflict_blocks"] > 0
    assert "красные" in dialog.status.text() and "не подтверждена" in dialog.status.text()
    dialog.close()
