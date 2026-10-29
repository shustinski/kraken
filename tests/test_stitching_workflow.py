from __future__ import annotations

from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication, QFileDialog, QMainWindow
from PyQt6.QtTest import QTest

from kraken_manager.domain.artifacts import ArtifactScope
from kraken_manager.domain.stitching import StitchingKind, stitching_series_id
from kraken_server.services import CommandContext
from test_server_workspace_inputs_parts.server_layer import _server_layer


@pytest.mark.parametrize("kind", list(StitchingKind))
def test_stitching_accepts_arbitrary_files_keeps_variants_and_selects_primary(tmp_path, kind):
    server, local, session, project, layer, *_ = _server_layer(tmp_path)
    series_id = stitching_series_id(layer.id, kind)
    with pytest.raises(ValueError, match="Сначала загрузите"):
        local.set_layer_stitching_status(
            principal=session.principal, project_id=project.id, layer_id=layer.id, kind=kind.value, completed=True
        )
    series = local.create_artifact_series(
        principal=session.principal,
        project_id=project.id,
        layer_id=layer.id,
        scope=ArtifactScope.LAYER_ATTACHMENT,
        name=kind.title,
        series_id=series_id,
        idempotency_key="stitching-series",
    )
    assert series.id == series_id
    versions = []
    for index in range(2):
        path = tmp_path / f"variant-{index}.arbitrary"
        path.write_bytes(b"external stitching\x00" + bytes([index]))
        versions.append(
            local.add_managed_artifact_version(
                principal=session.principal,
                project_id=project.id,
                series_id=series.id,
                source=path,
                idempotency_key=f"upload-{index}",
            )
        )
    local.activate_artifact_version(
        principal=session.principal,
        project_id=project.id,
        series_id=series.id,
        version_id=versions[0].id,
        idempotency_key="primary",
    )
    assert local.active_artifact_version(project.id, series.id).id == versions[0].id
    assert len(local.artifact_versions(project.id, series.id)) == 2
    event = local.set_layer_stitching_status(
        principal=session.principal,
        project_id=project.id,
        layer_id=layer.id,
        kind=kind.value,
        completed=True,
        version_id=str(versions[0].id),
    )
    assert event.payload["completed"] is True
    other_kind = next(k for k in StitchingKind if k is not kind)
    with pytest.raises(ValueError, match="Сначала загрузите"):
        local.set_layer_stitching_status(
            principal=session.principal,
            project_id=project.id,
            layer_id=layer.id,
            kind=other_kind.value,
            completed=True,
            version_id=str(versions[0].id),
        )
    assert local.read_project_blob(project.id, versions[0].blob.sha256) == b"external stitching\x00\x00"

    server.uow_factory = local._uow(str(project.id))
    server.profiles = local.profiles
    revision = server.events.current_revision(f"layer-pipeline:{layer.id}")
    result = server.append_pipeline_event(
        str(project.id),
        {
            "event_type": "LayerStitchingStatusChangedV1",
            "layer_id": str(layer.id),
            "kind": kind.value,
            "completed": False,
            "version_id": "",
        },
        CommandContext(str(session.principal.id), "server-unmark", revision),
    )
    assert result["payload"]["completed"] is False
    with pytest.raises(Exception, match="Сначала загрузите"):
        server.append_pipeline_event(
            str(project.id),
            {
                "event_type": "LayerStitchingStatusChangedV1",
                "layer_id": str(layer.id),
                "kind": other_kind.value,
                "completed": True,
                "version_id": str(versions[0].id),
            },
            CommandContext(str(session.principal.id), "server-invalid-mark", revision + 1),
        )


def test_stitching_dialog_uploads_and_marks_only_the_matching_kind(tmp_path, monkeypatch):
    from time import monotonic
    from kraken_hub.presentation.qt.stitching_dialog import StitchingDialog

    app = QApplication.instance() or QApplication([])
    _, local, session, project, layer, *_ = _server_layer(tmp_path)
    shell = QMainWindow()
    controller = SimpleNamespace(
        shell=shell, service=local, session=session, _project_id=project.id, _refresh_layer_pipeline=lambda *_: None
    )
    dialog = StitchingDialog(controller, str(layer.id))
    intra = dialog._controls[StitchingKind.INTRA_LAYER]
    inter = dialog._controls[StitchingKind.INTER_LAYER]
    assert not intra[3].isEnabled() and not inter[3].isEnabled()
    path = tmp_path / "external-file.any-extension"
    path.write_bytes(b"stitching")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_: (str(path), ""))

    def wait():
        deadline = monotonic() + 10
        while dialog._worker is not None and monotonic() < deadline:
            QTest.qWait(20)
        assert dialog._worker is None

    dialog._upload(StitchingKind.INTRA_LAYER)
    wait()
    assert intra[1].count() == 1
    assert intra[3].isEnabled()
    assert not inter[3].isEnabled()
    intra[3].click()
    wait()
    assert intra[3].isChecked()
    assert not inter[3].isChecked()
    dialog.reject()
    reopened = StitchingDialog(controller, str(layer.id))
    assert reopened._controls[StitchingKind.INTRA_LAYER][3].isChecked()
    reopened.close()
    shell.close()
    app.processEvents()
