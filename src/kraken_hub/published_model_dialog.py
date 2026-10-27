"""Named immutable model selection through the existing local/remote catalog."""
from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QVBoxLayout

from kraken_hub.manager_app_parts.layer_create_thread import _BackgroundCallThread
from kraken_manager.domain.artifacts import ArtifactScope, is_published_model_artifact


def published_models(service, project_id, layer_id):
    choices = []
    for series in service.list_artifact_series(project_id, layer_id=layer_id):
        if (series.archived or series.scope is not ArtifactScope.LAYER_ATTACHMENT
                or str(series.project_id) != str(project_id) or str(series.layer_id) != str(layer_id)):
            continue
        versions = service.artifact_versions(project_id, series.id)
        for version in versions:
            if str(version.series_id) == str(series.id) and is_published_model_artifact(version):
                choices.append((series.name, version))
    return tuple(sorted(choices, key=lambda item: (item[1].created_at, str(item[1].id)), reverse=True))


class PublishedModelDialog(QDialog):
    def __init__(self, service, project_id, layer_id, *, source_representation_id=None,
                 preferred_version_id=None, allow_legacy=True, language="Русский", background_workers=None, parent=None):
        super().__init__(parent)
        self.service, self.project_id, self.layer_id = service, str(project_id), str(layer_id)
        self.source_id, self.preferred = source_representation_id, preferred_version_id
        self.language, self._registry, self._worker = language, background_workers, None
        self.setWindowTitle(self._t("Выбор опубликованной модели", "Choose a published model"))
        self.resize(780, 550)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(self._t("Модель и неизменяемая версия", "Model and immutable version")))
        self.models = QComboBox()
        self.models.setObjectName("publishedModelVersions")
        self.models.setEnabled(False)
        layout.addWidget(self.models)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details)
        self.status = QLabel(self._t("Загрузка версий…", "Loading versions…"))
        layout.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        if allow_legacy:
            self.models.addItem(self._t("Внешний файл модели (существующий сценарий)", "External model file (legacy workflow)"), "legacy")
        layout.addWidget(self.buttons)
        self.models.currentIndexChanged.connect(self._show_details)
        self._sources, self._versions = (), {}
        worker = _BackgroundCallThread(self._load, parent=self)
        self._worker = worker
        if self._registry is not None:
            self._registry.add(worker)
        worker.succeeded.connect(self._loaded)
        worker.failed.connect(lambda error: self.status.setText(error))
        def finished():
            if self._registry is not None:
                self._registry.discard(worker)
            self._worker = None
        worker.finished.connect(finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _t(self, ru, en):
        return en if self.language == "English" else ru

    def _load(self):
        models = published_models(self.service, self.project_id, self.layer_id)
        sources = []
        if self.source_id:
            series = self.service.list_artifact_series(self.project_id, layer_id=self.layer_id, representation_id=self.source_id)
            for item in series:
                version = self.service.active_artifact_version(self.project_id, item.id)
                if version is not None:
                    sources.append((version.filename, str(version.id), version.sha256))
        return models, tuple(sources)

    def _loaded(self, result):
        models, self._sources = result
        for name, version in models:
            identifier = str(version.id)
            self._versions[identifier] = version
            self.models.addItem(f"{name} — {version.filename} — {version.created_at.isoformat()} — {identifier[:8]}", identifier)
        preferred = self.models.findData(str(self.preferred)) if self.preferred else -1
        self.models.setCurrentIndex(preferred if preferred >= 0 else (1 if self.models.count() > 1 and self.models.itemData(0) == "legacy" else 0))
        self.models.setEnabled(True)
        self.status.setText(self._t(f"Опубликованных версий: {len(models)}", f"Published versions: {len(models)}"))
        self._show_details()

    def _show_details(self):
        identifier = self.models.currentData()
        version = self._versions.get(identifier)
        lines = []
        if version is not None:
            lines = [self._t("Версия модели: ", "Model version: ") + str(version.id),
                     f"SHA-256: {version.sha256}",
                     self._t("Инструмент: ", "Tool: ") + f"{version.tool_name or '—'} {version.tool_version or '—'}"]
        elif identifier == "legacy":
            lines = [self._t("Файл будет выбран и закреплён по SHA-256. Он не является опубликованной версией Kraken.",
                             "The external file will be pinned by SHA-256. It is not a published Kraken version.")]
        lines += ["", self._t(f"Исходных версий сейчас: {len(self._sources)}", f"Current source versions: {len(self._sources)}")]
        lines += [f"{name} | {identifier} | SHA-256 {digest}" for name, identifier, digest in self._sources[:100]]
        if len(self._sources) > 100:
            lines.append(self._t("Показаны первые 100; полный набор закрепляется при отправке задания.", "First 100 shown; submission pins the complete input set."))
        lines.append(self._t("При отправке Kraken заново проверит доступ и закрепит актуальные версии исходных кадров.",
                             "Submission rechecks authorization and pins the current source frame versions."))
        self.details.setPlainText("\n".join(lines))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(version or identifier == "legacy"))

    def parameters(self):
        identifier = self.models.currentData()
        return {"legacy_file": True} if identifier == "legacy" else {"model_version_id": identifier}

    def done(self, result):
        if self._worker is None:
            super().done(result)

    def closeEvent(self, event):
        if self._worker is not None:
            event.ignore()
        else:
            super().closeEvent(event)
