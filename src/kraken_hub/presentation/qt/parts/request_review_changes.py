"""manager_app: request review changes."""
from __future__ import annotations
from uuid import uuid4


class RequestReviewChangesMixin:
    def _request_review_changes(self, batch) -> None:
        from PyQt6.QtWidgets import QInputDialog

        reason, accepted = QInputDialog.getMultiLineText(
            self.shell,
            "На доработку",
            "Причина (обязательно):",
        )
        if not accepted:
            return
        if not reason.strip():
            raise ValueError("Укажите причину возврата на доработку.")
        self.service.request_review_changes(
            principal=self.session.principal,
            batch=batch,
            reason=reason,
            idempotency_key=str(uuid4()),
        )

    def _refresh_current_matrix(self) -> None:
        workspace = self._workspace
        project_id = str(self._project_id or "")
        layer_id = str(getattr(workspace, "_selected_layer_id", "") if workspace else "")
        if workspace is None or not project_id or not layer_id:
            return
        self._load_representations(workspace, project_id, layer_id)
