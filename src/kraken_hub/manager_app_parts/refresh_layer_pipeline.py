"""manager_app: refresh layer pipeline."""
from __future__ import annotations


class RefreshLayerPipelineMixin:
    def _refresh_layer_pipeline(self, project_id: str, layer_id: str) -> bool:
        if self._layer_dialog is None:
            return False
        try:
            snapshot = self._pipeline_snapshot(project_id, layer_id)
        except Exception as exc:
            self._error(f"Не удалось загрузить историю обработки слоя: {exc}")
            return False
        self._layer_dialog.set_pipeline(snapshot)
        return True
