"""manager_app: refresh matrix."""
from __future__ import annotations
from kraken_core.frame_matrix import (
    MatrixSession,
)
from kraken_core.frame_matrix.qt import FrameMatrixWidget


class RefreshMatrixMixin:
    def _refresh_matrix(self, workspace, project_id) -> None:
        layer_id = getattr(workspace, "_selected_layer_id", None)
        if layer_id is None:
            workspace.matrix_view.clear_cells()
            return
        representation_ids = tuple(
            str(value)
            for value in (
                workspace.image_representation_combo.currentData(),
                workspace.vector_representation_combo.currentData(),
            )
            if value
        )
        data_source = getattr(workspace, "_matrix_data_source", None)
        if data_source is None or not isinstance(workspace.matrix_view, FrameMatrixWidget):
            return
        data_source.set_context(
            layer_id=str(layer_id),
            representation_ids=representation_ids,
            matrix_width=workspace.matrix_view.matrix_width,
        )
        project = self.service.get_project(project_id)
        if project is None:
            workspace.matrix_view.clear_cells()
            return
        previous = workspace.matrix_view.session()
        generation = 1 if previous is None else previous.generation + 1
        workspace.matrix_view.set_session(
            MatrixSession(
                namespace=str(project.id),
                width=project.width,
                height=project.height,
                source_revision=f"{project.revision}:{layer_id}:{','.join(representation_ids)}",
                orientation=project.orientation.value,
                generation=generation,
            ),
            data_source=data_source,
            asset_source=getattr(workspace, "_matrix_asset_source", None),
        )
