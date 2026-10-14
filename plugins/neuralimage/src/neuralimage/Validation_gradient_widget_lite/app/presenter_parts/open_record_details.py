"""presenter: open record details."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from ...core.domain import FrameRecord
from ...ui.details_dialog import ExtendFrameDetailsDialog
from ...ui.ui_constants import (
    DEFAULT_MATRIX_METRIC_KEY,
)
from neuralimage.Validation_gradient_widget_lite.app.state import ExtendMatrixTabState


class OpenRecordDetailsMixin:
    def _open_record_details(self, record: FrameRecord, state: ExtendMatrixTabState, tile_selection: object | None = None) -> None:
        session_view_state = dict(self._details_view_payload)
        if tile_selection is not None:
            parent_row = int(getattr(tile_selection, "matrix_row", getattr(tile_selection, "row", 0)))
            parent_column = int(getattr(tile_selection, "matrix_column", getattr(tile_selection, "column", 0)))
            sub_row = int(getattr(tile_selection, "sub_row", getattr(tile_selection, "tile_row", getattr(tile_selection, "row", 0))))
            sub_column = int(getattr(tile_selection, "sub_column", getattr(tile_selection, "tile_column", getattr(tile_selection, "column", 0))))
            spec = getattr(tile_selection, "spec", None)
            session_view_state["subpixel_selection"] = {
                "parent_row": parent_row,
                "parent_column": parent_column,
                "sub_row": sub_row,
                "sub_column": sub_column,
                "parent_value": float(getattr(tile_selection, "parent_value", getattr(record, "score", 0.0))),
                "subpixel_value": float(getattr(tile_selection, "subpixel_value", getattr(record, "score", 0.0))),
                "subpixel_confidence": None if getattr(tile_selection, "subpixel_confidence", None) is None else float(getattr(tile_selection, "subpixel_confidence")),
                "aggregation": str(getattr(tile_selection, "aggregation", "mean")),
                "metric_key": str(getattr(tile_selection, "metric_key", state.metric_key or DEFAULT_MATRIX_METRIC_KEY)),
                "spec_rows": int(getattr(spec, "rows", 0) or 0),
                "spec_columns": int(getattr(spec, "columns", 0) or 0),
            }
            session_view_state["tile_selection"] = {
                "row": sub_row,
                "column": sub_column,
            }
        else:
            session_view_state.pop("tile_selection", None)
            session_view_state.pop("subpixel_selection", None)
        dialog = ExtendFrameDetailsDialog(
            record=record,
            build_result=state.build_result,
            preferred_metric_key=state.metric_key,
            session_view_state=session_view_state,
            on_view_state_changed=self._store_details_view_payload,
            parent=None,
        )
        dialog.setModal(False)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.destroyed.connect(lambda *_args, dialog=dialog: self._forget_details_dialog(dialog))
        self._details_dialogs.append(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _store_details_view_payload(self, payload: dict[str, object]) -> None:
        self._details_view_payload = dict(payload or {})
        self._settings_service.save_details_view_payload(self._details_view_payload)
