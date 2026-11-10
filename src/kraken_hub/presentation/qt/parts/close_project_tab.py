"""manager_app: close project tab."""
from __future__ import annotations


class CloseProjectTabMixin:
    def _on_project_tab_close_requested(self, project_id: str) -> None:
        project_id = str(project_id)
        workspace = self._project_workspaces.pop(project_id, None)
        is_remote = getattr(self.service, "is_remote_project", None)
        unsubscribe = getattr(self.service, "unsubscribe_project", None)
        if callable(unsubscribe) and callable(is_remote) and is_remote(project_id):
            unsubscribe(project_id)
        if self._project_id == project_id:
            self._project_id = None
            self._workspace = None
            if self._layer_dialog is not None:
                self._layer_dialog.hide()
                self._layer_dialog.deleteLater()
                self._layer_dialog = None
        if workspace is not None:
            workspace.matrix_view.close()
        self.shell.close_project_tab(project_id)
