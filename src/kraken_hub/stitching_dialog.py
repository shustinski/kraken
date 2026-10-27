"""Manage arbitrary external stitching files through existing artifact APIs."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from kraken_manager.domain.artifacts import ArtifactScope
from kraken_manager.domain.stitching import StitchingKind, stitching_series_id
from kraken_hub.manager_app_parts.layer_create_thread import _BackgroundCallThread


class StitchingDialog(QDialog):
    def __init__(self, controller, layer_id: str):
        super().__init__(controller.shell)
        self.setWindowTitle("Сшивка слоя")
        self.resize(620, 300)
        self._controller = controller
        self._project_id = str(controller._project_id)
        self._layer_id = str(layer_id)
        self._worker = None
        self._controls = {}
        root = QVBoxLayout(self)
        self._status = QLabel(self)
        root.addWidget(self._status)
        computed = QPushButton("Вычисляемая сшивка…", self)
        computed.setObjectName("computedStitchButton")
        computed.setEnabled(hasattr(controller, "agent_runtime"))
        computed.clicked.connect(self._open_computed)
        root.addWidget(computed)
        for kind in StitchingKind:
            group = QGroupBox(kind.title, self)
            layout = QVBoxLayout(group)
            row = QHBoxLayout()
            variants = QComboBox(group)
            variants.setObjectName(f"stitchingVariants_{kind.value}")
            upload = QPushButton("Загрузить вариант…", group)
            primary = QPushButton("Сделать основным", group)
            completed = QCheckBox("Слой сшит", group)
            completed.setObjectName(f"stitchingCompleted_{kind.value}")
            row.addWidget(variants, 1)
            row.addWidget(upload)
            row.addWidget(primary)
            layout.addLayout(row)
            layout.addWidget(completed)
            root.addWidget(group)
            self._controls[kind] = (group, variants, primary, completed)
            upload.clicked.connect(lambda _checked=False, k=kind: self._upload(k))
            primary.clicked.connect(lambda _checked=False, k=kind: self._make_primary(k))
            completed.clicked.connect(lambda checked, k=kind: self._mark(k, checked))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.refresh()

    def _open_computed(self):
        from kraken_hub.computed_stitch_dialog import ComputedStitchDialog
        self._computed = ComputedStitchDialog(self._controller, self._layer_id, self)
        self._computed.show()

    def refresh(self):
        service = self._controller.service
        available = {str(s.id): s for s in service.list_artifact_series(self._project_id, layer_id=self._layer_id)}
        statuses = {}
        for event in service.history(self._project_id):
            if (
                event.event_type == "LayerStitchingStatusChangedV1"
                and str(event.payload.get("layer_id")) == self._layer_id
            ):
                statuses[str(event.payload.get("kind"))] = bool(event.payload.get("completed"))
        for kind, (_group, variants, primary, completed) in self._controls.items():
            series_id = stitching_series_id(self._layer_id, kind)
            series = available.get(str(series_id))
            versions = (
                () if series is None or series.archived else service.artifact_versions(self._project_id, series_id)
            )
            active = None if not versions else service.active_artifact_version(self._project_id, series_id)
            variants.clear()
            for version in versions:
                mark = " · основной" if active is not None and version.id == active.id else ""
                variants.addItem(f"{version.filename} · {version.created_at:%d.%m.%Y %H:%M}{mark}", str(version.id))
            if active is not None:
                variants.setCurrentIndex(variants.findData(str(active.id)))
            primary.setEnabled(bool(versions))
            completed.setEnabled(bool(versions))
            completed.setChecked(bool(versions) and statuses.get(kind.value, False))
            completed.setToolTip("Сначала загрузите файл этого вида сшивки" if not versions else "")

    def _run(self, function):
        if self._worker is not None:
            return
        self._status.setText("Сохранение…")
        for group, *_ in self._controls.values():
            group.setEnabled(False)
        worker = _BackgroundCallThread(function, parent=self)
        self._worker = worker

        def succeeded(_result):
            self._status.setText("Сохранено")

        def failed(message):
            self._status.setText("Не удалось сохранить")
            QMessageBox.warning(self, "Сшивка слоя", message)

        def finished():
            self._worker = None
            for group, *_ in self._controls.values():
                group.setEnabled(True)
            try:
                self.refresh()
                self._controller._refresh_layer_pipeline(self._project_id, self._layer_id)
            except Exception as exc:
                failed(str(exc))

        worker.succeeded.connect(succeeded)
        worker.failed.connect(failed)
        worker.finished.connect(finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _upload(self, kind: StitchingKind):
        filename, _ = QFileDialog.getOpenFileName(self, "Файл сшивки", "", "Все файлы (*)")
        if not filename:
            return
        service = self._controller.service
        principal = self._controller.session.principal
        series_id = stitching_series_id(self._layer_id, kind)

        def upload():
            series = next(
                (
                    s
                    for s in service.list_artifact_series(self._project_id, layer_id=self._layer_id)
                    if s.id == series_id
                ),
                None,
            )
            if series is None:
                service.create_artifact_series(
                    principal=principal,
                    project_id=self._project_id,
                    layer_id=self._layer_id,
                    scope=ArtifactScope.LAYER_ATTACHMENT,
                    name=kind.title,
                    series_id=series_id,
                    idempotency_key=f"stitching-series:{series_id}",
                )
            return service.add_managed_artifact_version(
                principal=principal,
                project_id=self._project_id,
                series_id=series_id,
                source=Path(filename),
                idempotency_key=str(uuid4()),
            )

        self._run(upload)

    def _make_primary(self, kind: StitchingKind):
        version_id = str(self._controls[kind][1].currentData() or "")
        if not version_id:
            return
        self._run(
            lambda: self._controller.service.activate_artifact_version(
                principal=self._controller.session.principal,
                project_id=self._project_id,
                series_id=stitching_series_id(self._layer_id, kind),
                version_id=version_id,
                idempotency_key=str(uuid4()),
            )
        )

    def _mark(self, kind: StitchingKind, completed: bool):
        version_id = str(self._controls[kind][1].currentData() or "")
        self._run(
            lambda: self._controller.service.set_layer_stitching_status(
                principal=self._controller.session.principal,
                project_id=self._project_id,
                layer_id=self._layer_id,
                kind=kind.value,
                completed=completed,
                version_id=version_id,
            )
        )

    def reject(self):
        if self._worker is None:
            super().reject()

    def closeEvent(self, event):
        if self._worker is not None:
            event.ignore()
        else:
            super().closeEvent(event)
