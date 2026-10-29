"""manager_app: open project."""
from __future__ import annotations
import os
from pathlib import Path
from kraken_core.frame_matrix import (
    MatrixSession,
    ThumbnailStoreFactory,
)
from kraken_core.frame_matrix.qt import FrameMatrixWidget
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.project import RepresentationKind
from kraken_manager.presentation.qt import (
    ProjectWorkspacePage,
)
from kraken_manager.presentation.qt.models import ProjectListItem
from kraken_hub.presentation.qt.matrix_source import KrakenMatrixAssetSource, KrakenMatrixDataSource


class OpenProjectMixin:
    def open_project(self, item: ProjectListItem) -> None:
        project = self.service.get_project(item.project_id)
        if project is None:
            self._error("Проект больше не доступен")
            self.refresh_projects()
            return
        is_remote = bool(getattr(self.service, "is_remote_project", lambda _id: False)(project.id))
        workspace_binding = self.service.project_workspace(project.id)
        if (
            workspace_binding is None
            and not is_remote
            and Permission.MANAGE_STRUCTURE in self.service.project_permissions(project.id, self.session.principal)
        ):
            workspace_binding = self._setup_project_workspace(project)
        if workspace_binding is not None and not is_remote:
            unavailable = [
                path
                for path in (
                    workspace_binding.source_project_dir,
                    workspace_binding.derived_project_dir,
                )
                if not Path(path).is_dir()
            ]
            if unavailable:
                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.warning(
                    self.shell,
                    "Хранилище недоступно",
                    "Метаданные и история доступны только для просмотра. "
                    "Файловые операции и плагины заблокированы.\n\n"
                    + "\n".join(unavailable),
                )
        cache_root = self.service.data_dir / "cache" / "frame-thumbnails"
        store_uri = (
            self.thumbnail_store_uri
            or os.environ.get("KRAKEN_THUMBNAIL_STORE_URI")
            or f"sqlite:///{cache_root.as_posix()}"
        )
        try:
            thumbnail_store = ThumbnailStoreFactory().create(store_uri)
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning("Кэш миниатюр недоступен (%s): %s", store_uri, exc)
            self.shell.statusBar().showMessage(
                "Кэш миниатюр недоступен, используется память процесса",
                10000,
            )
            thumbnail_store = ThumbnailStoreFactory().create("memory://")
        data_source = KrakenMatrixDataSource(
            self.service,
            project_id=str(project.id),
            matrix_width=project.width,
        )
        asset_source = KrakenMatrixAssetSource(self.service, project_id=str(project.id))
        matrix_view = FrameMatrixWidget(
            project.width,
            project.height,
            project.orientation.value,
            data_source=data_source,
            asset_source=asset_source,
            thumbnail_store=thumbnail_store,
        )
        workspace = self.shell.open_project_workspace(ProjectWorkspacePage(matrix_view=matrix_view))
        self._workspace = workspace
        if self._project_id is not None:
            unsubscribe = getattr(self.service, "unsubscribe_project", None)
            if callable(unsubscribe):
                unsubscribe(self._project_id)
        self._project_id = str(project.id)
        subscribe = getattr(self.service, "subscribe_project", None)
        if is_remote and callable(subscribe):
            subscribe(project.id)
        if self._layer_dialog is not None:
            self._layer_dialog.hide()
            self._layer_dialog.deleteLater()
            self._layer_dialog = None
        workspace._matrix_data_source = data_source
        workspace._matrix_asset_source = asset_source
        workspace.set_project_title(project.name)
        workspace.matrix_view.set_session(
            MatrixSession(
                namespace=str(project.id),
                width=project.width,
                height=project.height,
                source_revision=str(project.revision),
                orientation=project.orientation.value,
                generation=0,
            ),
            data_source=data_source,
            asset_source=asset_source,
        )
        workspace.matrix_view.set_visual_modes(*self.shell.visual_modes())
        self._load_layers(workspace, project)

        def add_layer() -> None:
            self._add_layer(workspace, project.id)

        workspace.addLayerRequested.connect(add_layer)
        workspace.layerActivated.connect(
            lambda layer_item: self._select_layer(workspace, project.id, layer_item)
        )
        workspace.addImageRepresentationRequested.connect(
            lambda: self._add_representation(workspace, project.id, RepresentationKind.IMAGE)
        )
        workspace.addVectorRepresentationRequested.connect(
            lambda: self._add_representation(workspace, project.id, RepresentationKind.VECTOR)
        )
        workspace.imageRepresentationChanged.connect(
            lambda identifier: self._activate_representation(workspace, project.id, identifier)
        )
        workspace.vectorRepresentationChanged.connect(
            lambda identifier: self._activate_representation(workspace, project.id, identifier)
        )
        workspace.matrix_view.contextMenuRequested.connect(
            self._show_matrix_context_menu
        )
        permissions = self.service.project_permissions(
            project.id,
            self.session.principal if not is_remote else getattr(
                getattr(self.service, "remote", None),
                "auth",
                type("A", (), {"principal": self.session.principal})(),
            ).principal,
        )
        self.shell.review_return_action.setVisible(
            Permission.RETURN_REVIEW in permissions
        )
        layers = self.service.list_layers(project.id)
        if layers:
            first = workspace.layer_model.layer_by_id(str(layers[0].id))
            if first is not None:
                workspace.layer_tabs.setCurrentIndex(0)
                self._select_layer(workspace, project.id, first)
        self._configure_analysis(workspace, project.id)
