"""details_dialog: populate model combos."""
from __future__ import annotations
import numpy as np
from PyQt6.QtCore import QTimer


class PopulateModelCombosMixin:
    def _populate_model_combos(self) -> None:
        current_a = self.model_a_combo.currentData()
        current_b = self.model_b_combo.currentData()
        for combo in (self.model_a_combo, self.model_b_combo):
            combo.blockSignals(True)
            combo.clear()
            for spec in self._build_result.model_specs:
                combo.addItem(spec.display_name, spec.model_id)
            combo.blockSignals(False)
        if self.model_a_combo.count() <= 0:
            return
        index_a = self.model_a_combo.findData(current_a)
        self.model_a_combo.setCurrentIndex(index_a if index_a >= 0 else 0)
        preferred_b = current_b
        if preferred_b is None or preferred_b == self.model_a_combo.currentData():
            preferred_b = None
            for index in range(self.model_b_combo.count()):
                candidate = self.model_b_combo.itemData(index)
                if candidate != self.model_a_combo.currentData():
                    preferred_b = candidate
                    break
            if preferred_b is None:
                preferred_b = self.model_a_combo.currentData()
        index_b = self.model_b_combo.findData(preferred_b)
        self.model_b_combo.setCurrentIndex(index_b if index_b >= 0 else 0)

    def _schedule_reset_view(self) -> None:
        QTimer.singleShot(0, self.overlay_view.reset_view)

    def _has_ground_truth(self) -> bool:
        return self._payload.get("gt_mask") is not None

    def _has_original(self) -> bool:
        original = self._payload.get("original_gray")
        if original is None:
            return False
        return np.asarray(original).size > 0

    def _model_display_name(self, model_id: str | None) -> str:
        if model_id is None:
            return "-"
        payload_names = self._payload.get("model_display_names") or {}
        if model_id in payload_names:
            return str(payload_names[model_id])
        for spec in self._build_result.model_specs:
            if spec.model_id == model_id:
                return spec.display_name
        return str(model_id)

    def _status_label(self, status: str) -> str:
        return self._t(f"status.{status}")
