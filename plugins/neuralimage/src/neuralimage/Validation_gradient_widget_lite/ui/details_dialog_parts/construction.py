"""details_dialog: construction."""
from __future__ import annotations
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QBrush, QColor, QPen, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QLabel,
)
from ...core.domain import BuildResult, FrameRecord
from ...core.subpixel_grid import (
    SubpixelGridSpec,
    SubpixelSelection,
)
from ...core.tile_grid import TileGridPlan
from ...core.workers import DetailConfidenceWorker, DetailPayloadWorker
from ...ui.i18n import Translator


class ConstructionMixin:
    def __init__(
        self,
        record: FrameRecord,
        build_result: BuildResult,
        preferred_metric_key: str | None = None,
        *,
        session_view_state: dict[str, object] | None = None,
        on_view_state_changed=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._record = record
        self._build_result = build_result
        self._translator = Translator()
        self._t = self._translator.tr
        self._payload: dict[str, object] = {}
        self._overlay_cache: dict[tuple[object, ...], QPixmap] = {}
        self._derived_cache: dict[tuple[object, ...], object] = {}
        self._comparison_score: float | None = None
        self._preferred_metric_key = str(preferred_metric_key or build_result.selected_metric_key or "overall_frame_score")
        self._pending_initial_fit = True
        self._detail_thread: QThread | None = None
        self._detail_worker: DetailPayloadWorker | None = None
        self._confidence_thread: QThread | None = None
        self._confidence_worker: DetailConfidenceWorker | None = None
        self._retired_confidence_workers: list[tuple[DetailConfidenceWorker, QThread]] = []
        self._loading_confidence_model_id: str | None = None
        self._detail_request_generation = 0
        self._confidence_request_generation = 0
        self._payload_loading = False
        self._legacy_base_hold_active = False
        self._hold_preview_mode: str | None = None
        self._hold_preview_pixmap = QPixmap()
        self._session_view_state = session_view_state if session_view_state is not None else {}
        self._on_view_state_changed = on_view_state_changed
        tile_selection_payload = self._session_view_state.get("tile_selection")
        self._tile_selection = None
        self._subpixel_selection: SubpixelSelection | None = None
        subpixel_selection_payload = self._session_view_state.get("subpixel_selection")
        if isinstance(tile_selection_payload, dict):
            row = tile_selection_payload.get("row")
            column = tile_selection_payload.get("column")
            if row is not None and column is not None:
                self._tile_selection = (int(row), int(column))
        elif isinstance(tile_selection_payload, (tuple, list)) and len(tile_selection_payload) >= 2:
            self._tile_selection = (int(tile_selection_payload[0]), int(tile_selection_payload[1]))
        if isinstance(subpixel_selection_payload, dict):
            try:
                spec = None
                spec_rows = subpixel_selection_payload.get("spec_rows")
                spec_columns = subpixel_selection_payload.get("spec_columns")
                if spec_rows is not None and spec_columns is not None:
                    spec = SubpixelGridSpec(
                        rows=int(spec_rows),
                        columns=int(spec_columns),
                        mode=str(subpixel_selection_payload.get("spec_mode") or "grid"),
                        tile_width=int(subpixel_selection_payload.get("spec_tile_width") or 1),
                        tile_height=int(subpixel_selection_payload.get("spec_tile_height") or 1),
                        overlap=int(subpixel_selection_payload.get("spec_overlap") or 0),
                    ).normalized()
                self._subpixel_selection = SubpixelSelection(
                    parent_row=int(subpixel_selection_payload.get("parent_row", 0)),
                    parent_column=int(subpixel_selection_payload.get("parent_column", 0)),
                    sub_row=int(subpixel_selection_payload.get("sub_row", 0)),
                    sub_column=int(subpixel_selection_payload.get("sub_column", 0)),
                    parent_value=float(subpixel_selection_payload.get("parent_value", 0.0)),
                    subpixel_value=float(subpixel_selection_payload.get("subpixel_value", 0.0)),
                    subpixel_confidence=(
                        None if subpixel_selection_payload.get("subpixel_confidence") is None else float(subpixel_selection_payload.get("subpixel_confidence"))
                    ),
                    aggregation=str(subpixel_selection_payload.get("aggregation") or "mean"),
                    metric_key=str(subpixel_selection_payload.get("metric_key") or "overall_frame_score"),
                    spec=spec,
                )
            except Exception:
                self._subpixel_selection = None
        self._restored_result_kind: str | None = None
        self._sticky_result_kind: str | None = None
        self._tile_plan: TileGridPlan | None = None
        self.frame_id_value = QLabel("-", self)
        self.frame_id_value.hide()
        self.resize(1440, 940)
        self.setWindowTitle(self._t("details.window_title", name=record.display_name))
        self._build_ui()
        scene = self.overlay_view.scene()
        assert scene is not None
        self.original_item = scene.addPixmap(QPixmap())
        self.first_source_item = scene.addPixmap(QPixmap())
        self.second_source_item = scene.addPixmap(QPixmap())
        self.result_item = scene.addPixmap(QPixmap())
        self.hold_preview_item = scene.addPixmap(QPixmap())
        self.tile_preview_item = scene.addPixmap(QPixmap())
        self.tile_grid_item = scene.addPixmap(QPixmap())
        self.tile_selection_item = scene.addRect(0.0, 0.0, 0.0, 0.0)
        # Legacy lite aliases retained for backward compatibility.
        self.base_item = self.original_item
        self.first_item = self.first_source_item
        self.second_item = self.second_source_item
        self.base_visible = self.original_visible
        self.first_visible = self.first_source_visible
        self.second_visible = self.second_source_visible
        self.first_color = QComboBox(self)
        self.second_color = QComboBox(self)
        self.first_color.hide()
        self.second_color.hide()
        self.original_item.setZValue(0.0)
        self.first_source_item.setZValue(1.0)
        self.second_source_item.setZValue(2.0)
        self.result_item.setZValue(3.0)
        self.hold_preview_item.setZValue(4.0)
        self.tile_preview_item.setZValue(-1.0)
        self.tile_grid_item.setZValue(3.5)
        self.tile_selection_item.setZValue(4.5)
        selection_pen = QPen(QColor(255, 196, 0, 235), 0.0)
        selection_pen.setCosmetic(True)
        self.tile_selection_item.setPen(selection_pen)
        self.tile_selection_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.tile_selection_item.setVisible(False)
        self.tile_preview_item.setVisible(False)
        self.tile_grid_item.setVisible(False)
        self.overlay_view.viewTransformChanged.connect(self._update_tile_lod)
        self.overlay_view.leftClicked.connect(self._on_view_clicked)
        self._connect_signals()
        self._create_navigation_shortcuts()
        self._restore_view_settings()
        self._populate_model_combos()
        self._load_current_payload(reset_view=True, preserve_selection=False)
        self._legacy_update_frame_id_value()
