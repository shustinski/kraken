"""Bank of normal cells of one layer: where it came from, its sizes and limits, its examples.

The window shows what the defect search compares every cell with. A cell from the frame
window can be put next to the bank: its distance, the limit and the nearest examples, and the
operator can add it to the bank as a normal cell (its size too, when asked). Cells added by
hand are marked in the gallery and can be taken out again.
Building a bank reads dozens of masks, so it runs in a thread while the window waits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from PyQt6.QtCore import QAbstractListModel, QModelIndex, QRect, QRectF, QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from ..core.cell_error_analysis import CellErrorConfig, bank_verdict
from ..core.normal_bank import (
    ORIGIN_OPERATOR,
    ORIGIN_RUN,
    ExemplarView,
    ManualCell,
    NormalBank,
    exemplar_unusualness,
    nearest_exemplars,
)

ORIGIN_MANUAL = "manual"

THUMB_BOX = 76
CELL_COLOR = QColor("#e8eef6")
QUERY_COLOR = QColor("#ffd84a")
MANUAL_COLOR = QColor("#8fe3a0")
SIZE_FRAME_COLOR = QColor(111, 168, 255, 150)
LIMIT_COLOR = QColor("#ff8a65")
BAR_COLOR = QColor("#4f7fc4")


@dataclass(frozen=True, slots=True)
class CellQuery:
    """A cell from a frame, put next to the bank."""

    frame: str
    view: ExemplarView
    descriptor: np.ndarray


@dataclass(frozen=True, slots=True)
class _Entry:
    origin: str
    index: int
    view: ExemplarView
    unusual: float
    note: str = ""
    cell_id: str = ""


class _BuildThread(QThread):
    built = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, build: Callable[[], Any], parent=None) -> None:
        super().__init__(parent)
        self._build = build

    def run(self) -> None:
        try:
            profile = self._build()
            unusual = None
            bank = getattr(profile, "normal_bank", None)
            if bank is not None and bank.ready:
                unusual = exemplar_unusualness(bank)
            self.built.emit((profile, unusual))
        except Exception as error:  # pragma: no cover - shown in the window
            self.failed.emit(str(error))


def _mask_image(mask: np.ndarray, color: QColor) -> QImage:
    rows, cols = mask.shape
    rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
    rgba[mask] = (color.red(), color.green(), color.blue(), 255)
    image = QImage(rgba.data, cols, rows, cols * 4, QImage.Format.Format_RGBA8888)
    return image.copy()


def _paint_cell(painter: QPainter, rect: QRect, view: ExemplarView, scale: float, cell: tuple[float, float], color: QColor) -> None:
    """The cell at a common scale, centred, over a frame of the usual cell size for comparison."""

    center = QRectF(rect).center()
    usual_w, usual_h = cell[0] * scale, cell[1] * scale
    pen = QPen(SIZE_FRAME_COLOR, 1.0, Qt.PenStyle.DashLine)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(QRectF(center.x() - usual_w / 2, center.y() - usual_h / 2, usual_w, usual_h))
    width, height = view.width * scale, view.height * scale
    target = QRectF(center.x() - width / 2, center.y() - height / 2, width, height)
    painter.drawImage(target, _mask_image(view.mask(), color))


class _GalleryModel(QAbstractListModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.entries: list[_Entry] = []
        self.tooltip: Callable[[_Entry], str] = lambda _entry: ""

    def set_entries(self, entries: list[_Entry]) -> None:
        self.beginResetModel()
        self.entries = list(entries)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt name
        return 0 if parent.isValid() else len(self.entries)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.entries):
            return None
        entry = self.entries[index.row()]
        if role == Qt.ItemDataRole.ToolTipRole:
            return self.tooltip(entry)
        if role == Qt.ItemDataRole.UserRole:
            return entry
        return None


class _CellDelegate(QStyledItemDelegate):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.scale = 2.0
        self.cell = (0.0, 0.0)
        self.color = CELL_COLOR

    def sizeHint(self, _option, _index) -> QSize:  # noqa: N802 - Qt name
        return QSize(THUMB_BOX, THUMB_BOX + 14)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        entry = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(entry, _Entry):
            return
        painter.save()
        palette = option.palette
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        painter.fillRect(option.rect.adjusted(1, 1, -1, -1), palette.highlight() if selected else palette.base())
        box = QRect(option.rect.left(), option.rect.top(), option.rect.width(), option.rect.height() - 14)
        # Cells added by hand stand out in green.
        color = MANUAL_COLOR if entry.origin == ORIGIN_MANUAL else self.color
        _paint_cell(painter, box, entry.view, self.scale, self.cell, color)
        painter.setPen(palette.color(palette.ColorRole.PlaceholderText))
        text = entry.note or f"{entry.unusual:.2f}"
        painter.drawText(
            QRect(option.rect.left(), option.rect.bottom() - 14, option.rect.width(), 14),
            Qt.AlignmentFlag.AlignCenter,
            text,
        )
        painter.restore()


class _SizeHistogram(QWidget):
    """Widths or heights of the bank's normal cells, with the limits of the current slider."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(70)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._values = np.zeros(0)
        self._limits = (0.0, 0.0)

    def set_values(self, values: np.ndarray, limits: tuple[float, float]) -> None:
        self._values = np.asarray(values, dtype=np.float64)
        self._limits = (float(limits[0]), float(limits[1]))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt name
        painter = QPainter(self)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.base())
        if self._values.size == 0:
            painter.end()
            return
        low = int(np.floor(min(self._values.min(), self._limits[0]))) - 1
        high = int(np.ceil(max(self._values.max(), self._limits[1]))) + 1
        counts = np.bincount(np.clip(np.round(self._values).astype(int) - low, 0, high - low), minlength=high - low + 1)
        area = self.rect().adjusted(6, 4, -6, -16)
        step = area.width() / float(len(counts))
        peak = max(1, int(counts.max()))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(BAR_COLOR)
        for offset, count in enumerate(counts):
            if count:
                height = max(1.0, area.height() * count / peak)
                painter.drawRect(QRectF(area.left() + offset * step + 1, area.bottom() - height, max(1.0, step - 2), height))

        def x_of(value: float) -> float:
            return area.left() + (value - low + 0.5) * step

        painter.setPen(QPen(LIMIT_COLOR, 1.5, Qt.PenStyle.DashLine))
        for value in self._limits:
            painter.drawLine(int(x_of(value)), area.top(), int(x_of(value)), area.bottom())
        painter.setPen(palette.color(palette.ColorRole.PlaceholderText))
        for value in (low + 1, high - 1):
            painter.drawText(QRectF(x_of(value) - 20, area.bottom() + 2, 40, 14), Qt.AlignmentFlag.AlignCenter, str(value))
        painter.end()


class CellBankDialog(QDialog):
    """Bank of normal cells of one layer (not modal)."""

    # The shown cell is normal: (query, its size is normal too, what the bank said about it).
    addRequested = pyqtSignal(object, bool, str)
    # Take a cell added by hand out of the bank: its id.
    removeRequested = pyqtSignal(str)

    def __init__(
        self,
        translator: Callable[..., str],
        *,
        layer_name: str,
        geometry_sensitivity: int,
        shape_limit: float,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._t = translator
        self._geometry = int(geometry_sensitivity)
        self._shape_limit = float(shape_limit)
        self._profile = None
        self._bank: NormalBank | None = None
        self._unusual: dict[str, np.ndarray] = {}
        self._query: CellQuery | None = None
        self._manual: dict[str, ManualCell] = {}
        self._thread: _BuildThread | None = None
        self.setWindowTitle(self._t("cell_bank.window", name=layer_name))
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.resize(980, 700)

        root = QHBoxLayout(self)
        left = QVBoxLayout()
        root.addLayout(left, stretch=0)
        right = QVBoxLayout()
        root.addLayout(right, stretch=1)

        self.status_label = QLabel("", self)
        self.status_label.setWordWrap(True)
        left.addWidget(self.status_label)
        summary_group = QGroupBox(self._t("cell_bank.summary"), self)
        summary_group.setMinimumWidth(300)
        summary_group.setMaximumWidth(360)
        self.summary_form = QFormLayout(summary_group)
        left.addWidget(summary_group)
        sizes_group = QGroupBox(self._t("cell_bank.sizes"), self)
        sizes_group.setMaximumWidth(360)
        sizes_layout = QVBoxLayout(sizes_group)
        self.width_title = QLabel("", sizes_group)
        self.width_histogram = _SizeHistogram(sizes_group)
        self.height_title = QLabel("", sizes_group)
        self.height_histogram = _SizeHistogram(sizes_group)
        for widget in (self.width_title, self.width_histogram, self.height_title, self.height_histogram):
            sizes_layout.addWidget(widget)
        sizes_hint = QLabel(self._t("cell_bank.sizes_hint"), sizes_group)
        sizes_hint.setWordWrap(True)
        sizes_layout.addWidget(sizes_hint)
        left.addWidget(sizes_group)
        left.addStretch(1)

        self.query_group = QGroupBox(self._t("cell_bank.query"), self)
        query_layout = QHBoxLayout(self.query_group)
        self.query_view = QLabel(self.query_group)
        self.query_view.setFixedSize(THUMB_BOX * 2, THUMB_BOX * 2)
        query_layout.addWidget(self.query_view)
        query_text = QVBoxLayout()
        self.query_label = QLabel("", self.query_group)
        self.query_label.setWordWrap(True)
        query_text.addWidget(self.query_label)
        self.nearest_title = QLabel(self._t("cell_bank.nearest"), self.query_group)
        query_text.addWidget(self.nearest_title)
        self.nearest_model = _GalleryModel(self)
        self.nearest_delegate = _CellDelegate(self)
        self.nearest_view = self._make_list(self.nearest_model, self.nearest_delegate, flow_wrap=False)
        self.nearest_view.setFixedHeight(THUMB_BOX + 34)
        query_text.addWidget(self.nearest_view)
        add_row = QHBoxLayout()
        self.add_button = QPushButton(self._t("cell_bank.add"), self.query_group)
        self.add_button.setToolTip(self._t("cell_bank.add_tooltip"))
        self.add_button.clicked.connect(self._request_add)
        add_row.addWidget(self.add_button)
        self.size_normal_check = QCheckBox(self._t("cell_bank.size_normal"), self.query_group)
        self.size_normal_check.setToolTip(self._t("cell_bank.size_normal_tooltip"))
        add_row.addWidget(self.size_normal_check)
        add_row.addStretch(1)
        query_text.addLayout(add_row)
        query_layout.addLayout(query_text, stretch=1)
        self.query_group.hide()
        right.addWidget(self.query_group)

        gallery_bar = QHBoxLayout()
        self.gallery_title = QLabel(self._t("cell_bank.examples"), self)
        gallery_bar.addWidget(self.gallery_title, stretch=1)
        self.order_combo = QComboBox(self)
        for key in ("unusual", "width", "height", "area", "frame"):
            self.order_combo.addItem(self._t(f"cell_bank.order.{key}"), key)
        self.order_combo.currentIndexChanged.connect(self._fill_gallery)
        self.show_combo = QComboBox(self)
        for key in ("all", "manual"):
            self.show_combo.addItem(self._t(f"cell_bank.show.{key}"), key)
        self.show_combo.currentIndexChanged.connect(self._fill_gallery)
        gallery_bar.addWidget(self.show_combo)
        gallery_bar.addWidget(QLabel(self._t("cell_bank.order"), self))
        gallery_bar.addWidget(self.order_combo)
        self.remove_button = QPushButton(self._t("cell_bank.remove"), self)
        self.remove_button.setToolTip(self._t("cell_bank.remove_tooltip"))
        self.remove_button.setEnabled(False)
        self.remove_button.clicked.connect(self._request_remove)
        gallery_bar.addWidget(self.remove_button)
        right.addLayout(gallery_bar)
        self.gallery_model = _GalleryModel(self)
        self.gallery_model.tooltip = self._entry_tooltip
        self.nearest_model.tooltip = self._entry_tooltip
        self.gallery_delegate = _CellDelegate(self)
        self.gallery_view = self._make_list(self.gallery_model, self.gallery_delegate, flow_wrap=True)
        self.gallery_view.selectionModel().currentChanged.connect(lambda *_args: self._sync_remove_button())
        right.addWidget(self.gallery_view, stretch=1)
        hint = QLabel(self._t("cell_bank.examples_hint"), self)
        hint.setWordWrap(True)
        right.addWidget(hint)

    # --- Building ------------------------------------------------------------------------------

    def build(self, build: Callable[[], Any], on_built: Callable[[Any], None] | None = None) -> None:
        """Build the bank in a thread (a resolved profile is shown at once by set_profile)."""

        self.status_label.setText(self._t("cell_bank.building"))
        thread = _BuildThread(build, self)
        thread.built.connect(lambda result: self._on_built(result, on_built))
        thread.failed.connect(lambda text: self.status_label.setText(self._t("cell_bank.failed", error=text)))
        self._thread = thread
        thread.start()

    def _on_built(self, result, on_built) -> None:
        profile, unusual = result
        if profile is not None and callable(on_built):
            on_built(profile)
        self.set_profile(profile, unusual)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        if self._thread is not None and self._thread.isRunning():
            self._thread.wait(30000)
        super().closeEvent(event)

    # --- Content -------------------------------------------------------------------------------

    def set_profile(self, profile, unusual: dict[str, np.ndarray] | None = None) -> None:
        self._profile = profile
        bank = getattr(profile, "normal_bank", None)
        self._bank = bank
        while self.summary_form.rowCount():
            self.summary_form.removeRow(0)
        if bank is None:
            self.status_label.setText(self._t("cell_bank.none"))
            self.gallery_model.set_entries([])
            return
        if not bank.ready:
            self.status_label.setText(self._t("cell_bank.not_ready", reason=self._reason_text(bank.reason)))
        else:
            self.status_label.setText(self._source_text(profile, bank))
        self._unusual = unusual if unusual is not None else (exemplar_unusualness(bank) if bank.ready else {})
        self._fill_summary(profile, bank)
        if bank.ready:
            (w_lo, w_hi), (h_lo, h_hi) = bank.side_limits(self._geometry)
            sizes = bank.normal_sizes if len(bank.normal_sizes) else np.zeros((0, 2))
            self.width_title.setText(self._side_text("width", bank.cell_width, (w_lo, w_hi), bank.width_variable))
            self.height_title.setText(self._side_text("height", bank.cell_height, (h_lo, h_hi), bank.height_variable))
            self.width_histogram.set_values(sizes[:, 0], (w_lo, w_hi))
            self.height_histogram.set_values(sizes[:, 1], (h_lo, h_hi))
            views = [*bank.exemplar_views, *bank.operator_views]
            largest = max([1, *(max(view.width, view.height) for view in views)])
            cell_side = max(bank.cell_width, bank.cell_height, 1.0)
            # One scale for every example (sizes stay comparable), the usual cell about half the box.
            scale = max(0.5, min(4.0, (THUMB_BOX - 8) / max(float(largest), 1.6 * cell_side)))
            for delegate in (self.gallery_delegate, self.nearest_delegate):
                delegate.scale = scale
                delegate.cell = (float(bank.cell_width), float(bank.cell_height))
        self._fill_gallery()
        self._show_query()

    def set_query(self, query: CellQuery | None) -> None:
        self._query = query
        self._show_query()

    def set_manual_cells(self, cells) -> None:
        """Cells added by hand to this layer (for their tooltips and the summary); set before
        set_profile, which shows them."""

        self._manual = {cell.cell_id: cell for cell in cells or ()}

    def _request_add(self) -> None:
        query, bank = self._query, self._bank
        if query is None or bank is None or not bank.ready:
            return
        verdict = bank_verdict(bank, query.view.width, query.view.height, query.descriptor, self._config())
        self.addRequested.emit(query, bool(self.size_normal_check.isChecked()), verdict)

    def _request_remove(self) -> None:
        entry = self._current_entry()
        if entry is not None and entry.origin == ORIGIN_MANUAL and entry.cell_id:
            self.removeRequested.emit(entry.cell_id)

    def _current_entry(self) -> _Entry | None:
        index = self.gallery_view.currentIndex()
        entry = index.data(Qt.ItemDataRole.UserRole) if index.isValid() else None
        return entry if isinstance(entry, _Entry) else None

    def _sync_remove_button(self) -> None:
        entry = self._current_entry()
        self.remove_button.setEnabled(entry is not None and entry.origin == ORIGIN_MANUAL)

    def _config(self) -> CellErrorConfig:
        return CellErrorConfig(geometry_sensitivity=self._geometry)

    def _fill_summary(self, profile, bank: NormalBank) -> None:
        rows: list[tuple[str, str]] = [
            (self._t("cell_bank.frames"), str(int(bank.frames_used))),
            (self._t("cell_bank.cells"), str(int(bank.candidates))),
            (self._t("cell_bank.exemplars"), self._exemplar_count_text(bank)),
        ]
        if bank.ready:
            rows.append(
                (
                    self._t("cell_bank.usual_cell"),
                    self._t("cell_bank.usual_cell_value", w=f"{bank.cell_width:.0f}", h=f"{bank.cell_height:.0f}", area=f"{bank.cell_area:.0f}"),
                )
            )
            rows.append((self._t("cell_bank.shape_limit"), self._t("cell_bank.shape_limit_value", limit=f"{self._shape_limit:.2f}", slider=self._geometry)))
        report = getattr(profile, "markup_report", None)
        if report:
            if not report.get("used"):
                text = self._t("cell_bank.markup_unused", cells=int(report.get("cells", 0)))
            elif report.get("flagged"):
                text = self._t("cell_bank.markup_flagged", count=len(report["flagged"]))
            else:
                text = self._t("cell_bank.markup_clean")
            rows.append((self._t("cell_bank.markup_check"), text))
        manual_ids = tuple(bank.manual_ids)
        if manual_ids or self._manual:
            overruled = sum(1 for cell_id in manual_ids if self._manual.get(cell_id) and self._manual[cell_id].flagged)
            text = self._t("cell_bank.manual_value", count=len(manual_ids), overruled=overruled)
            if self._manual and not bank.ready:
                text = self._t("cell_bank.manual_waiting", count=len(self._manual))
            rows.append((self._t("cell_bank.manual"), text))
        for title, value in rows:
            label = QLabel(value, self)
            label.setWordWrap(True)
            self.summary_form.addRow(title, label)
        tip = QLabel(self._t("cell_bank.shape_hint"), self)
        tip.setWordWrap(True)
        self.summary_form.addRow(tip)

    def _entries(self) -> list[_Entry]:
        bank = self._bank
        if bank is None or not bank.ready:
            return []
        entries: list[_Entry] = []
        first_manual = len(bank.operator_views) - len(bank.manual_ids)
        for origin, views in ((ORIGIN_RUN, bank.exemplar_views), (ORIGIN_OPERATOR, bank.operator_views)):
            unusual = self._unusual.get(origin, np.zeros(len(views)))
            for index, view in enumerate(views):
                value = float(unusual[index]) if index < len(unusual) else 0.0
                if origin == ORIGIN_OPERATOR and index >= first_manual:
                    cell_id = bank.manual_ids[index - first_manual]
                    entries.append(_Entry(ORIGIN_MANUAL, index, view, value, cell_id=cell_id))
                else:
                    entries.append(_Entry(origin, index, view, value))
        return entries

    def _fill_gallery(self) -> None:
        entries = self._entries()
        if str(self.show_combo.currentData() or "all") == "manual":
            entries = [entry for entry in entries if entry.origin == ORIGIN_MANUAL]
        key = str(self.order_combo.currentData() or "unusual")
        orders: dict[str, Callable[[_Entry], Any]] = {
            "unusual": lambda entry: -entry.unusual,
            "width": lambda entry: (entry.view.width, entry.view.height),
            "height": lambda entry: (entry.view.height, entry.view.width),
            "area": lambda entry: entry.view.area,
            "frame": lambda entry: (entry.view.frame, entry.index),
        }
        entries.sort(key=orders.get(key, orders["unusual"]))
        self.gallery_model.set_entries(entries)
        self.gallery_title.setText(self._t("cell_bank.examples_count", count=len(entries)))
        self._sync_remove_button()

    def _show_query(self) -> None:
        query, bank = self._query, self._bank
        if query is None:
            self.query_group.hide()
            return
        self.query_group.show()
        pixmap_image = QImage(THUMB_BOX * 2, THUMB_BOX * 2, QImage.Format.Format_RGBA8888)
        pixmap_image.fill(self.palette().base().color())
        painter = QPainter(pixmap_image)
        cell = (float(bank.cell_width), float(bank.cell_height)) if bank is not None and bank.ready else (0.0, 0.0)
        _paint_cell(painter, pixmap_image.rect(), query.view, 2.0 * self.gallery_delegate.scale, cell, QUERY_COLOR)
        painter.end()
        from PyQt6.QtGui import QPixmap

        self.query_view.setPixmap(QPixmap.fromImage(pixmap_image))
        if bank is None or not bank.ready:
            self.query_label.setText(self._t("cell_bank.query_no_bank", frame=query.frame))
            self.nearest_model.set_entries([])
            self.add_button.setEnabled(False)
            return
        distance = float(bank.normal_distance(query.descriptor[None, :])[0])
        verdict = bank_verdict(bank, query.view.width, query.view.height, query.descriptor, self._config())
        verdict_key = "cell_bank.query_shape_off" if "shape" in verdict else "cell_bank.query_shape_ok"
        lines = [
            self._t("cell_bank.query_cell", frame=query.frame, w=query.view.width, h=query.view.height, area=query.view.area),
            self._t(verdict_key, distance=f"{distance:.2f}", limit=f"{self._shape_limit:.2f}"),
            self._t("cell_bank.query_size_off" if "size" in verdict else "cell_bank.query_size_ok"),
        ]
        views = {ORIGIN_RUN: bank.exemplar_views, ORIGIN_OPERATOR: bank.operator_views}
        first_manual = len(bank.operator_views) - len(bank.manual_ids)
        nearest = []
        already = False
        for origin, index, value in nearest_exemplars(bank, query.descriptor, count=8):
            pool = views.get(origin) or ()
            if index >= len(pool):
                continue
            if origin == ORIGIN_OPERATOR and index >= first_manual:
                # The very same cell is in the bank already (added by hand).
                # (the distance is not exactly 0 after float16 storage and rounding; the pixels decide)
                already = already or (value < 0.01 and pool[index].mask_bits == query.view.mask_bits)
                nearest.append(_Entry(ORIGIN_MANUAL, index, pool[index], value, note=f"{value:.2f}"))
            else:
                nearest.append(_Entry(origin, index, pool[index], value, note=f"{value:.2f}"))
        if already:
            lines.append(self._t("cell_bank.query_in_bank"))
        self.query_label.setText("\n".join(lines))
        self.nearest_model.set_entries(nearest)
        self.add_button.setEnabled(not already)

    # --- Texts ---------------------------------------------------------------------------------

    def _make_list(self, model: _GalleryModel, delegate: _CellDelegate, *, flow_wrap: bool) -> QListView:
        view = QListView(self)
        view.setModel(model)
        view.setItemDelegate(delegate)
        view.setViewMode(QListView.ViewMode.IconMode)
        view.setFlow(QListView.Flow.LeftToRight)
        view.setWrapping(flow_wrap)
        view.setResizeMode(QListView.ResizeMode.Adjust)
        view.setUniformItemSizes(True)
        view.setMovement(QListView.Movement.Static)
        view.setSpacing(2)
        view.setSelectionMode(QListView.SelectionMode.SingleSelection)
        return view

    def _entry_tooltip(self, entry: _Entry) -> str:
        text = self._t(
            "cell_bank.example_tooltip",
            origin=self._t(f"cell_bank.origin.{entry.origin}"),
            frame=entry.view.frame or "—",
            w=entry.view.width,
            h=entry.view.height,
            area=entry.view.area,
            unusual=f"{entry.unusual:.2f}",
        )
        cell = self._manual.get(entry.cell_id) if entry.cell_id else None
        if cell is not None:
            text += "\n" + self._t("cell_bank.manual_size_normal" if cell.size_normal else "cell_bank.manual_shape_only")
            if cell.flagged:
                text += "\n" + self._t("cell_bank.manual_overruled")
            if cell.added:
                text += "\n" + self._t("cell_bank.manual_added", date=cell.added)
        return text

    def _source_text(self, profile, bank: NormalBank) -> str:
        frame_id = str(getattr(profile, "frame_id", "") or "")
        kind = frame_id.split(":", 1)[0] if ":" in frame_id else "run"
        if kind not in ("markup", "reference", "run", "details"):
            kind = "run"
        return self._t(f"cell_bank.source.{kind}", frames=int(bank.frames_used))

    def _exemplar_count_text(self, bank: NormalBank) -> str:
        parts = []
        if len(bank.exemplars):
            parts.append(self._t("cell_bank.exemplars_run", count=len(bank.exemplars)))
        drawn = len(bank.operator_exemplars) - len(bank.manual_ids)
        if drawn:
            parts.append(self._t("cell_bank.exemplars_operator", count=drawn))
        if bank.manual_ids:
            parts.append(self._t("cell_bank.exemplars_manual", count=len(bank.manual_ids)))
        return ", ".join(parts) or "0"

    def _side_text(self, side: str, usual: float, limits: tuple[float, float], variable: bool) -> str:
        key = "cell_bank.side_variable" if variable else "cell_bank.side_fixed"
        return self._t(
            key,
            side=self._t(f"cell_bank.{side}"),
            usual=f"{usual:.0f}",
            low=f"{limits[0]:.0f}",
            high=f"{limits[1]:.0f}",
        )

    def _reason_text(self, reason: str) -> str:
        key = f"cell_bank.reason.{reason}"
        text = self._t(key)
        return reason if text == key else text
