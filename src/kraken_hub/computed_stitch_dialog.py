"""Asynchronous local computation and bounded block inspection of real mosaics."""
import weakref
from threading import Event
from uuid import uuid4

from PyQt6 import sip
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QVBoxLayout)

from kraken_hub.manager_app_parts.layer_create_thread import _BackgroundCallThread
from kraken_manager.infrastructure.mosaic import MosaicReader, block_preview
from kraken_manager.infrastructure.plugin.stitch_gateway import AgentStitchGateway

_STITCH_CALLS = set()


def _command_result(call, kind, *, document=False):
    from kraken_hub.infrastructure.remote.parts.extract_review_archive import RemoteServerError
    from kraken_manager.application.errors import ApplicationError
    from kraken_manager.domain.common import DomainValidationError
    try:
        return kind, call(), None
    except Exception as exc:
        definitive = (isinstance(exc, (ApplicationError, DomainValidationError))
            or isinstance(exc, RemoteServerError) and exc.status in {400, 401, 403, 404, 409, 422, 428}
            or document and isinstance(exc, (ValueError, TypeError, KeyError)))
        if definitive:
            return "rejected", str(exc), None
        raise


class _AsyncDialog(QDialog):
    def _call(self, function):
        if self._worker is not None:
            return False
        worker = _BackgroundCallThread(function)
        self._worker = worker
        _STITCH_CALLS.add(worker)
        background = self._background
        if background is not None:
            background.add(worker)
        ref = weakref.ref(self)

        def loaded(value):
            owner = ref()
            if owner is not None and not sip.isdeleted(owner):
                try:
                    owner._loaded(value)
                except Exception as exc:
                    owner.status.setText(str(exc))

        def failed(error):
            owner = ref()
            if owner is not None and not sip.isdeleted(owner):
                owner.status.setText(error)

        def finished():
            _STITCH_CALLS.discard(worker)
            if background is not None:
                background.discard(worker)
            owner = ref()
            if owner is not None and not sip.isdeleted(owner):
                owner._worker = None

        worker.succeeded.connect(loaded)
        worker.failed.connect(failed)
        worker.finished.connect(finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        return True


class MosaicBlockDialog(_AsyncDialog):
    def __init__(self, path, parent=None, *, background_workers=None, block_loader=None, vector_exporter=None):
        super().__init__(parent)
        self._path, self._worker, self._background, self._metadata = path, None, background_workers, None
        self._block_loader = block_loader
        self._vector_exporter, self._export_cancel = vector_exporter, None
        self.setWindowTitle("Мозаика · просмотр блоков")
        self.resize(800, 720)
        layout = QVBoxLayout(self)
        self.status = QLabel("Загрузка…")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        row = QHBoxLayout()
        self.column, self.row = QSpinBox(), QSpinBox()
        self.coverage = QCheckBox("Покрытие: чёрный — пропуск, белый — один кадр, голубой — перекрытие")
        for label, widget in (("Столбец блока", self.column), ("Строка блока", self.row)):
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        button = QPushButton("Показать блок")
        button.clicked.connect(self.refresh)
        row.addWidget(button)
        layout.addLayout(row)
        layout.addWidget(self.coverage)
        self.export_button = QPushButton("Экспортировать векторы…")
        self.export_button.setObjectName("exportStitchVectors")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_vectors)
        layout.addWidget(self.export_button)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image, 1)
        self.refresh()

    def refresh(self):
        path, column, row, coverage = self._path, self.column.value(), self.row.value(), self.coverage.isChecked()
        loader = self._block_loader

        def read():
            if loader is not None:
                return loader(column, row, coverage)
            with MosaicReader(path) as reader:
                return block_preview(reader, column, row, coverage=coverage)
        self._call(read)

    def _loaded(self, value):
        if isinstance(value, dict) and value.get("kind") == "vector_export_failed":
            self._export_cancel = None
            self.export_button.setText("Экспортировать векторы…")
            self.status.setText(value["error"])
            return
        if isinstance(value, dict) and value.get("kind") == "vector_export":
            self._export_cancel = None
            self.export_button.setText("Экспортировать векторы…")
            self.status.setText(f"Векторы сохранены: {value['path']} · блоков: {value['blocks']}\n"
                "Геометрия в пикселях и метрах; расхождения сохранены. Электрическая связность не подтверждена.")
            return
        data, self._metadata, x, y, columns, rows = value
        self.column.setMaximum(columns-1)
        self.row.setMaximum(rows-1)
        pixmap = QPixmap()
        pixmap.loadFromData(data, "PNG")
        self.image.setPixmap(pixmap.scaled(740, 560, Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.FastTransformation))
        render = self._metadata["render"]
        self.status.setText(f"{render['width']} × {render['height']} px · блок ({x}, {y}) · "
                           f"{self._metadata['sampling']} / {self._metadata['overlap']}\n"
                           "Физическая точность определяется наблюдениями и их неопределённостью.")
        if "vector_stitch" in self._metadata:
            vectors = self._metadata["vector_stitch"]
            self.status.setText(self.status.text() + "\nКонтуры — зелёные; расхождения кадров — красные. "
                f"Блоков с расхождениями: {vectors['conflict_blocks']}. Электрическая связность не подтверждена.")
        self.export_button.setEnabled(self._vector_exporter is not None and "vector_stitch" in self._metadata)

    def export_vectors(self):
        if self._export_cancel is not None and self._worker is not None:
            self._export_cancel.set()
            self.status.setText("Отмена экспорта…")
            return
        if self._worker is not None or self._vector_exporter is None:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Экспорт векторов слоя", "vectors.kraken.jsonl",
            "Kraken vectors (*.jsonl)", options=QFileDialog.Option.DontConfirmOverwrite)
        if not filename:
            return
        cancel, exporter = Event(), self._vector_exporter
        self._export_cancel = cancel
        self.export_button.setText("Отменить экспорт")
        def export():
            try:
                return {"kind": "vector_export", **exporter(filename, cancelled=cancel.is_set)}
            except Exception as exc:
                return {"kind": "vector_export_failed", "error": str(exc)}
        self._call(export)

    def closeEvent(self, event):
        if self._export_cancel is not None:
            self._export_cancel.set()
        super().closeEvent(event)

    def reject(self):
        if self._export_cancel is not None:
            self._export_cancel.set()
        super().reject()


class ComputedStitchDialog(_AsyncDialog):
    def __init__(self, controller, layer_id, parent=None):
        super().__init__(parent or controller.shell)
        service = controller.service
        self.service = service._backend(controller._project_id) if hasattr(service, "_backend") else service
        self.project_id, self.layer_id = str(controller._project_id), str(layer_id)
        self.principal, self.runtime = controller.session.principal, controller.agent_runtime
        self._worker, self._background = None, getattr(controller, "_background_workers", None)
        self.snapshot, self._pending, self._gateway, self._pending_import = None, None, None, None
        self._notice = ""
        self.setWindowTitle("Вычисляемая сшивка слоя")
        self.resize(840, 420)
        root = QVBoxLayout(self)
        root.addWidget(QLabel("Геометрия строится по контрольным точкам или явно заданным преобразованиям.\n"
                             "Позиции файлов в каталоге не считаются физическим расположением кадров."))
        self.plans = QComboBox()
        self.plans.setObjectName("computedStitchPlans")
        root.addWidget(self.plans)
        row = QHBoxLayout()
        self.mode = QComboBox()
        for label, value in (("RGB", "rgb"), ("Оттенки серого", "gray"), ("Метки классов", "labels")):
            self.mode.addItem(label, value)
        row.addWidget(self.mode)
        self.import_button, self.run_button, self.cancel_button, self.view_button = (
            QPushButton(label) for label in ("Импорт наблюдений JSON…", "Вычислить", "Отменить", "Просмотреть результат"))
        for button in (self.import_button, self.run_button, self.cancel_button, self.view_button):
            row.addWidget(button)
        root.addLayout(row)
        self.status = QLabel("Загрузка…")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.import_button.clicked.connect(self.import_observations)
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(self.cancel)
        self.view_button.clicked.connect(self.view)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def refresh(self):
        if self._pending_import is not None:
            self._retry_import()
            return
        if self._pending is not None:
            self._retry_mutation()
            return
        service, project_id, principal, runtime = self.service, self.project_id, self.principal, self.runtime
        gateway = self._gateway
        remote = bool(getattr(service, "stitch_execution_remote", False))
        if not remote and not hasattr(service, "refresh_stitch"):
            self.status.setText("Серверное выполнение сшивки ещё не подключено.")
            for button in (self.import_button, self.run_button, self.cancel_button, self.view_button):
                button.setEnabled(False)
            self.timer.stop()
            return

        def read():
            snapshot = service.get_multilayer_model(project_id)
            if remote:
                return "snapshot", snapshot, None
            requests = snapshot["model"].get("stitch_renders", [])
            active = [r for r in requests if r["state"] in {"requested", "cancelled", "imported"}]
            current_gateway = gateway
            if active:
                runtime.ensure_started()
                current_gateway = AgentStitchGateway(base_url=runtime.base_url, token=runtime.token,
                    staging_root=runtime.data_dir / "staging")
                for request in active:
                    service.refresh_stitch(project_id=project_id, job_id=request["job_id"], principal=principal, gateway=current_gateway)
                snapshot = service.get_multilayer_model(project_id)
            return "snapshot", snapshot, current_gateway
        self._call(read)

    def _loaded(self, value):
        kind, payload, extra = value
        if kind == "rejected":
            self._pending, self._pending_import = None, None
            self._notice = payload
            self.status.setText(payload)
            return
        if kind in {"view", "remote_view"}:
            options = extra if isinstance(extra, dict) else {"loader": extra}
            dialog = MosaicBlockDialog(payload if kind == "view" else None, self,
                background_workers=self._background, block_loader=options.get("loader") if kind == "remote_view" else None,
                vector_exporter=options.get("exporter"))
            self._viewer = dialog
            dialog.show()
            return
        self.snapshot, self._gateway = payload, extra or self._gateway
        if kind == "mutation":
            self._pending = None
        if kind == "geometry":
            self._pending_import = None
        selected = self.plans.currentData()
        self.plans.clear()
        model = self.snapshot["model"]
        states = {(s["id"], s["revision"]): s for s in self.snapshot["assessment"]["stitches"]}
        for plan in model["stitches"]:
            if plan["layer_id"] != self.layer_id:
                continue
            key = (plan["id"], plan["revision"])
            state = states[key]
            fits = [p["fit"] for p in plan["tiles"] if p["fit"]]
            residuals = [p["max_residual_m"] for p in fits if p["max_residual_m"] is not None]
            self.plans.addItem(f"Ревизия {plan['revision']} · {len(plan['tiles'])} кадров · {state['geometry_status']} · "
                f"{state['render_status']} · {'current' if state['current'] else 'stale'} · max residual {max(residuals, default=None)} m", key)
        index = self.plans.findData(selected)
        self.plans.setCurrentIndex(index if index >= 0 and kind != "geometry" else self.plans.count()-1)
        chosen = self.plans.currentData()
        latest = next((r for r in reversed(model.get("stitch_renders", []))
                       if (r["stitch_id"], r["stitch_revision"]) == chosen), None)
        self.status.setText(self._notice or (latest["error"] if latest and latest.get("error") else
            "Готово" if self.plans.count() else "Настройте физический стек и импортируйте наблюдения сшивки."))

    def _mutate(self, operation, arguments):
        if self.snapshot is None:
            return
        if self._pending_import is not None:
            self.status.setText("Импорт наблюдений ещё ожидает подтверждения.")
            return
        revision = self.snapshot["model"]["revision"]
        if self._pending is not None and self._pending[:2] != (operation, arguments):
            self.status.setText("Предыдущая команда ещё ожидает подтверждения.")
            return
        self._pending = self._pending or (operation, arguments, str(uuid4()), revision)
        self._notice = ""
        self._retry_mutation()

    def _retry_mutation(self):
        service, project_id, principal = self.service, self.project_id, self.principal
        pending = self._pending

        def mutate():
            operation, arguments, key, revision = pending
            return _command_result(lambda: service.mutate_multilayer_model(project_id=project_id, principal=principal,
                expected_revision=revision, operation=operation, arguments=arguments, idempotency_key=key), "mutation")
        self._call(mutate)

    def run(self):
        key = self.plans.currentData()
        if key:
            self._mutate("request_layer_stitch", {"stitch_id": key[0], "stitch_revision": key[1], "mode": self.mode.currentData()})

    def cancel(self):
        key = self.plans.currentData()
        if self.snapshot and key:
            request = next((r for r in reversed(self.snapshot["model"].get("stitch_renders", []))
                            if (r["stitch_id"], r["stitch_revision"]) == key and r["state"] == "requested"), None)
            if request:
                self._mutate("cancel_layer_stitch", {"job_id": request["job_id"]})

    def import_observations(self):
        if self._worker is not None or self._pending is not None:
            return
        if self._pending_import is not None:
            self._retry_import()
            return
        filename, _ = QFileDialog.getOpenFileName(self, "Наблюдения сшивки", "", "JSON (*.json)")
        if not filename:
            return
        self._pending_import = filename, str(uuid4())
        self._notice = ""
        self._retry_import()

    def _retry_import(self):
        service, project_id, layer_id, principal = self.service, self.project_id, self.layer_id, self.principal
        filename, key = self._pending_import
        self._call(lambda: _command_result(lambda: service.import_stitch_observations(project_id=project_id, layer_id=layer_id,
            principal=principal, source_path=filename, idempotency_key=key), "geometry", document=True))

    def view(self):
        if not self.snapshot:
            return
        key = self.plans.currentData()
        request = next((r for r in reversed(self.snapshot["model"].get("stitch_renders", []))
                        if (r["stitch_id"], r["stitch_revision"]) == key and r["state"] == "imported"), None)
        if request:
            service, project_id, version = self.service, self.project_id, request["output"]["version_id"]
            job_id = request["job_id"]
            def exporter(destination, *, cancelled):
                return service.export_stitch_vectors(project_id=project_id, job_id=job_id, destination=destination, cancelled=cancelled)
            if getattr(service, "stitch_execution_remote", False):
                def loader(column, row, coverage):
                    return service.remote_stitch_block(project_id, job_id, column, row, coverage=coverage)
                self._loaded(("remote_view", None, {"loader": loader, "exporter": exporter}))
            else:
                self._call(lambda: ("view", service.managed_artifact_path(project_id, version), {"exporter": exporter}))

    def reject(self):
        self.timer.stop()
        super().reject()

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)
