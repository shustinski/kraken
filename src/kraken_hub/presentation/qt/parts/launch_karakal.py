"""manager_app: launch karakal."""
from __future__ import annotations
from kraken_manager.domain.ordering import natural_name_key
import logging
import shutil
from pathlib import Path
from uuid import uuid4
from kraken_manager.domain.project import RepresentationPurpose


class LaunchKarakalMixin:
    def _launch_karakal(self, layer_id: str) -> None:
        inventory = self.plugin_items.get("karakal")
        if inventory is None:
            self._error("Плагин karakal не установлен")
            return
        try:
            from karakal.core.domain import FolderSpec
            from karakal.app.presenter import FOLDER_CONFIDENCE_ROLE
            from karakal.plugin.plugin import KarakalPlugin
            from PyQt6.QtCore import Qt, QUrl
            from kraken_core.settings import QSettings
            from PyQt6.QtGui import QDesktopServices
            from PyQt6.QtWidgets import QDialog, QVBoxLayout
        except ImportError as exc:
            self._error(f"Не удалось загрузить Karakal: {exc}")
            return

        controller = self
        project_id = str(self._project_id)
        frame_key_map: dict[str, str] = {}
        project = self.service.get_project(project_id)
        if project is not None:
            for representation in self.service.list_representations(project_id, layer_id):
                if not representation.source or not Path(representation.source).is_dir():
                    continue
                paths = sorted(
                    (
                        path
                        for path in Path(representation.source).iterdir()
                        if path.is_file()
                    ),
                    key=lambda path: natural_name_key(path.name),
                )
                for index, path in enumerate(paths[: project.width * project.height]):
                    frame_id = str(
                        project.frame_id_at(
                            index % project.width + 1,
                            index // project.width + 1,
                        )
                    )
                    for key in (path.name, path.stem, str(path), frame_id):
                        frame_key_map[key] = frame_id

        class KrakenHost:
            def settings(self):
                return QSettings("Kraken", "Karakal")

            def logger(self):
                return logging.getLogger("kraken.karakal")

            def task_runner(self):
                return None

            def open_path(self, path: Path) -> None:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

            def publish_quality(self, payload: dict) -> None:
                confidences = {
                    frame_key_map.get(
                        str(item.get("frame_key", "")),
                        str(item.get("frame_key", "")),
                    ): float(item.get("confidence", 0.0))
                    for item in payload.get("frames", ())
                    if str(item.get("frame_key", "")).strip()
                }
                controller.service.publish_karakal_analysis(
                    principal=controller.session.principal,
                    project_id=project_id,
                    layer_id=layer_id,
                    frame_confidence=confidences,
                    report=dict(payload),
                    parameters=dict(payload.get("parameters", {})),
                    plugin_version=inventory.metadata.version,
                    idempotency_key=f"karakal:{uuid4()}",
                )
                workspace = controller._workspace
                if workspace is not None:
                    controller._refresh_matrix(workspace, project_id)
                if controller._layer_dialog is not None:
                    controller._layer_dialog.set_pipeline(
                        controller._pipeline_snapshot(project_id, layer_id)
                    )

        plugin = KarakalPlugin()
        dialog = QDialog(self.shell)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.setModal(False)
        dialog.setWindowTitle(f"Karakal · {inventory.metadata.version}")
        dialog.resize(1380, 860)
        layout = QVBoxLayout(dialog)
        widget = plugin.create_widget(KrakenHost(), dialog)
        layout.addWidget(widget)

        # External directory representations can be transferred immediately.
        # Managed BlobStore representations remain available through Agent
        # staging and are not exposed as arbitrary host filesystem paths.
        representations = self.service.list_representations(project_id, layer_id)
        karakal_stage = (
            self.service.data_dir / "agent-staging" / f"{project_id}-karakal-{uuid4()}"
        ).resolve()
        karakal_stage.mkdir(parents=True, exist_ok=False)
        staged_folders: dict[str, Path | None] = {}

        def representation_folder(representation) -> Path | None:
            identifier = str(representation.id)
            if identifier in staged_folders:
                return staged_folders[identifier]
            if representation.source and Path(representation.source).is_dir():
                result = Path(representation.source).resolve()
                staged_folders[identifier] = result
                return result
            cells = self.service.frame_cells(
                project_id,
                layer_id,
                representation.id,
            )
            if not cells:
                staged_folders[identifier] = None
                return None
            folder = karakal_stage / identifier
            folder.mkdir()
            for cell in cells:
                if not cell.sha256:
                    continue
                destination = folder / f"{cell.x}_{cell.y}.png"
                with destination.open("xb") as stream:
                    stream.write(self.service.read_project_blob(project_id, cell.sha256))
            result = folder if any(folder.iterdir()) else None
            staged_folders[identifier] = result
            return result

        source = next(
            (
                item
                for item in representations
                if item.purpose is RepresentationPurpose.SOURCE
                and representation_folder(item) is not None
            ),
            None,
        )
        if source is not None:
            source_path = representation_folder(source)
            assert source_path is not None
            widget._presenter._original_folder = FolderSpec(
                path=source_path,
                label=source.name,
            )
        for binary in representations:
            binary_path = (
                representation_folder(binary)
                if binary.purpose is RepresentationPurpose.BINARY
                else None
            )
            if binary_path is not None:
                item = widget._presenter._append_folder_item(
                    binary_path,
                    checked=True,
                )
                matching_confidence = [
                    value for value in representations
                    if value.purpose is RepresentationPurpose.CONFIDENCE
                    and value.source_image_representation_id == binary.source_image_representation_id
                ]
                matching_binaries = [
                    value for value in representations
                    if value.purpose is RepresentationPurpose.BINARY
                    and value.source_image_representation_id == binary.source_image_representation_id
                ]
                if len(matching_confidence) == len(matching_binaries) == 1:
                    confidence_path = representation_folder(matching_confidence[0])
                    if confidence_path is not None:
                        item.setData(FOLDER_CONFIDENCE_ROLE, str(confidence_path))
        widget._presenter._update_source_labels()
        widget._presenter._refresh_folder_rows()
        widget._presenter._sync_action_buttons()

        windows = getattr(self, "_plugin_windows", None)
        if windows is None:
            windows = []
            self._plugin_windows = windows
        windows.append((dialog, plugin))

        def dispose() -> None:
            plugin.shutdown()
            if (dialog, plugin) in windows:
                windows.remove((dialog, plugin))
            staging_root = (self.service.data_dir / "agent-staging").resolve()
            try:
                karakal_stage.relative_to(staging_root)
            except ValueError:
                return
            if karakal_stage.is_dir() and not karakal_stage.is_symlink():
                shutil.rmtree(karakal_stage)

        dialog.finished.connect(lambda _result: dispose())
        dialog.show()
