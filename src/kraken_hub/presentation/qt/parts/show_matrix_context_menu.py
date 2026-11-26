"""manager_app: actions of the layer frame matrix."""
from __future__ import annotations
from kraken_manager.domain.identity import Permission


class ShowMatrixContextMenuMixin:
    """Host of ``LayerFrameMatrixWidget`` actions: state checks and handlers."""

    def can_assign_work(self) -> bool:
        return getattr(self, "_project_id", None) is not None and self._has_permission(Permission.ASSIGN_WORK)

    def vector_file_for(self, x: int, y: int):
        workspace = getattr(self, "_workspace", None)
        if workspace is None:
            return None
        vector_representation_id = str(workspace.vector_representation_combo.currentData() or "")
        return next(
            (
                item
                for item in self._active_frame_files(x, y)
                if str(item[1].representation_id or "") == vector_representation_id
            ),
            None,
        )

    def _on_matrix_frame_action(self, action_id: str, context) -> None:
        if action_id == "export_files":
            self._export_selected_frame_files(context)
        elif action_id == "assign_work":
            self.send_selection_for_review()
        elif action_id == "frame_properties":
            workspace = self._workspace
            if workspace is not None:
                workspace.matrix_view.set_selection(type(context.selection).single(context.x, context.y))
            self.show_selected_frame()
        elif action_id == "vector_properties":
            vector_file = self.vector_file_for(context.x, context.y)
            if vector_file is not None:
                self._show_frame_file_properties(*vector_file)
