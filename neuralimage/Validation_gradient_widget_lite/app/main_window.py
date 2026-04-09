"""Main window for the extended validation gradient widget."""
from __future__ import annotations

from PyQt6.QtCore import QSettings, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMenu,
    QMenuBar,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QStyle,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from ..infra.services import ValidationGradientLiteSettingsService
from ..core.analysis_modes import ANALYSIS_MODE_OPTIONS, default_confidence_model_id
from ..ui.i18n import Translator, set_current_language
from ..ui.matrix_view import GradientPresetSelectorWidget, GradientRangeSelectorWidget, MatrixListWidget, MatrixMiniMapWidget
from ..ui.ui_constants import (
    BOUNDARY_RADIUS_RANGE,
    CONFIDENCE_UNCERTAINTY_DELTA_RANGE,
    CONTROL_PANEL_SPLITTER_SIZES,
    DEFAULT_BOUNDARY_RADIUS,
    DEFAULT_CELL_SIZE,
    DEFAULT_CONFIDENCE_UNCERTAINTY_DELTA,
    DEFAULT_ANALYSIS_MODE,
    DEFAULT_ERROR_WINDOW,
    DEFAULT_EXPORT_NEIGHBOR_RADIUS,
    DEFAULT_EXPORT_PERCENT,
    DEFAULT_EXPORT_PERCENTILE,
    DEFAULT_EXPORT_SELECTION_MODE,
    DEFAULT_FILTER_TO_EXPORT_CANDIDATES,
    DEFAULT_FRAMES_PER_ROW,
    DEFAULT_GEOMETRY_MODE,
    DEFAULT_GRADIENT_NAME,
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_POINT_CONFIDENCE_RADIUS,
    DEFAULT_POINT_EXTRACTION_MODE,
    DEFAULT_POINT_MATCH_RADIUS,
    DEFAULT_POLYGON_CONFIDENCE_SUMMARY,
    DEFAULT_MATRIX_COLUMNS,
    DEFAULT_MATRIX_LAYOUT_MODE,
    DEFAULT_MATRIX_METRIC_KEY,
    DEFAULT_METRIC_SCOPE,
    DEFAULT_MATRIX_ROWS,
    DEFAULT_TOP_K_EXPORT,
    DEFAULT_TOTAL_FRAMES,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    EXPORT_NEIGHBOR_RANGE,
    EXPORT_PERCENTILE_RANGE,
    EXPORT_PERCENT_RANGE,
    EXPORT_SELECTION_MODE_OPTIONS,
    EXPORT_TOP_K_RANGE,
    EXTEND_LANGUAGE_BUTTON_OBJECT_NAME,
    EXTEND_ROOT_OBJECT_NAME,
    EXTEND_WIDGET_STYLESHEET,
    FRAMES_PER_ROW_RANGE,
    GEOMETRY_MODE_OPTIONS,
    MASK_THRESHOLD_RANGE,
    MATRIX_COLUMNS_RANGE,
    METRIC_SETTINGS_COMBO_MIN_CONTENTS_LENGTH,
    METRIC_SETTINGS_LABEL_MIN_WIDTH,
    METRIC_SETTINGS_WIDGET_MIN_WIDTH,
    MATRIX_METRIC_GROUP_OPTIONS,
    MATRIX_METRIC_OPTIONS,
    MATRIX_ROWS_RANGE,
    OVERVIEW_PANEL_MAX_WIDTH,
    POINT_CONFIDENCE_RADIUS_RANGE,
    POINT_MATCH_RADIUS_RANGE,
    POLYGON_CONFIDENCE_SUMMARY_OPTIONS,
    SETTINGS_APP,
    SETTINGS_LABEL_MIN_WIDTH,
    SETTINGS_ORG,
    THUMBNAIL_SIZE_RANGE,
    TOTAL_FRAMES_RANGE,
)
from .presenter import ValidationGradientExtendPresenter
from .state import ExtendMatrixTabState, ExtendPreviewPanel




class _ExpandableScoreCard(QWidget):
    """Show one score as a clickable card with expandable details."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.title_label = QLabel(title, self)
        self.title_label.setWordWrap(True)
        value_row = QWidget(self)
        value_layout = QHBoxLayout(value_row)
        value_layout.setContentsMargins(0, 0, 0, 0)
        value_layout.setSpacing(6)
        self.value_button = QPushButton('-', value_row)
        self.value_button.setCheckable(True)
        self.value_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.value_button.setStyleSheet('padding: 6px 10px; border-radius: 8px; background-color: #2f3844; color: #edf3fb; font-weight: 700; border: none; text-align: center;')
        self.percentile_label = QLabel('-', value_row)
        self.percentile_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.percentile_label.setMinimumWidth(72)
        self.percentile_label.setStyleSheet('padding: 6px 10px; border-radius: 8px; background-color: #2f3844; color: #edf3fb; font-weight: 700;')
        self.details_label = QLabel('', self)
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet('padding: 4px 6px; color: #c9d3df; background-color: #11161d; border-radius: 6px;')
        self.details_label.hide()
        self.value_button.toggled.connect(self._on_toggled)
        value_layout.addWidget(self.value_button, stretch=1)
        value_layout.addWidget(self.percentile_label)
        layout.addWidget(self.title_label)
        layout.addWidget(value_row)
        layout.addWidget(self.details_label)

    def _on_toggled(self, checked: bool) -> None:
        self.details_label.setVisible(bool(checked) and bool(self.details_label.text().strip()))

    def set_payload(self, text: str, style: str, details: str, *, visible: bool, percentile_text: str = '-', percentile_style: str | None = None, tooltip: str = '') -> None:
        self.setVisible(visible)
        self.value_button.setText(text)
        self.value_button.setStyleSheet(style + '; border: none; text-align: center;')
        self.percentile_label.setText(percentile_text)
        self.percentile_label.setStyleSheet((percentile_style or style) + ';')
        self.percentile_label.setVisible(visible and bool(percentile_text.strip()))
        self.details_label.setText(details)
        self.details_label.setVisible(bool(self.value_button.isChecked()) and bool(details.strip()) and visible)
        self.setToolTip(tooltip)
        self.title_label.setToolTip(tooltip)
        self.value_button.setToolTip(tooltip)
        self.percentile_label.setToolTip(tooltip)
        self.details_label.setToolTip(tooltip)


class _PercentileHistogramWidget(QWidget):
    """Draw a compact histogram over fixed percentile bins."""

    binClicked = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._counts = [0, 0, 0, 0, 0]
        self._total = 0
        self._active_bin: int | None = None
        self.setMinimumHeight(120)

    def set_payload(self, counts: list[int], total: int, *, active_bin: int | None = None) -> None:
        self._counts = [int(value) for value in counts[:5]] + [0] * max(0, 5 - len(counts))
        self._counts = self._counts[:5]
        self._total = int(total)
        self._active_bin = None if active_bin is None else int(active_bin)
        self.update()

    def _chart_rect(self) -> QRectF:
        return QRectF(self.rect().adjusted(6, 8, -6, -20))

    def _bar_rects(self, rect: QRectF) -> list[QRectF]:
        if rect.width() <= 0 or rect.height() <= 0:
            return []
        max_count = max(1, max(self._counts, default=1))
        bar_width = max(12.0, rect.width() / max(5.0, len(self._counts) * 1.35))
        gap = max(6.0, (rect.width() - bar_width * len(self._counts)) / max(1.0, len(self._counts) - 1.0))
        rects: list[QRectF] = []
        for index, count in enumerate(self._counts):
            height_ratio = float(count) / float(max_count)
            bar_height = max(2.0, (rect.height() - 24.0) * height_ratio) if count > 0 else 2.0
            left = rect.left() + index * (bar_width + gap)
            top = rect.bottom() - 18.0 - bar_height
            rects.append(QRectF(left, top, bar_width, bar_height))
        return rects

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self._chart_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            painter.end()
            return
        painter.fillRect(rect, QColor('#11161d'))
        labels = ('0-20', '20-40', '40-60', '60-80', '80-100')
        for index, bar_rect in enumerate(self._bar_rects(rect)):
            count = self._counts[index]
            if index <= 1:
                color = QColor('#1f5f3b')
            elif index == 2:
                color = QColor('#8a6a12')
            else:
                color = QColor('#8c2f39')
            painter.fillRect(bar_rect, color)
            if self._active_bin == index:
                painter.setPen(QPen(QColor('#f5f8fc'), 2.0))
                painter.drawRect(bar_rect.adjusted(0.5, 0.5, -0.5, -0.5))
            painter.setPen(QPen(QColor('#dce7f3')))
            painter.drawText(QRectF(bar_rect.left() - 6.0, bar_rect.top() - 16.0, bar_rect.width() + 12.0, 14.0), Qt.AlignmentFlag.AlignCenter, str(count))
            painter.drawText(QRectF(bar_rect.left() - 8.0, rect.bottom() - 16.0, bar_rect.width() + 16.0, 14.0), Qt.AlignmentFlag.AlignCenter, labels[index])
        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        point = event.position()
        for index, bar_rect in enumerate(self._bar_rects(self._chart_rect())):
            if bar_rect.contains(point):
                self.binClicked.emit(index)
                event.accept()
                return
        super().mousePressEvent(event)


class _PercentileHistogramCard(QWidget):
    """Show one metric percentile distribution as a compact chart card."""

    binClicked = pyqtSignal(str, int)

    def __init__(self, title: str, metric_key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._metric_key = str(metric_key)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.setStyleSheet('background-color: #1a2028; border: 1px solid #304050; border-radius: 10px;')
        self.title_label = QLabel(title, self)
        self.title_label.setWordWrap(True)
        self.summary_label = QLabel('-', self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet('color: #c9d3df; background: transparent; border: none;')
        self.chart = _PercentileHistogramWidget(self)
        self.chart.binClicked.connect(lambda index: self.binClicked.emit(self._metric_key, index))
        layout.addWidget(self.title_label)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.chart)

    def set_payload(self, title: str, counts: list[int], total: int, *, visible: bool, active_bin: int | None = None, tooltip: str = '') -> None:
        self.setVisible(visible)
        self.title_label.setText(title)
        self.chart.set_payload(counts, total, active_bin=active_bin)
        labels = ('P0-20', 'P20-40', 'P40-60', 'P60-80', 'P80-100')
        translator = Translator()
        summary = ' | '.join(f'{label}: {int(count)}' for label, count in zip(labels, counts))
        if active_bin is not None and 0 <= int(active_bin) < len(labels):
            summary = f'{translator.tr("hist.filter_active")}: {labels[int(active_bin)]} | ' + summary
        self.summary_label.setText(summary)
        self.setToolTip(tooltip)
        self.title_label.setToolTip(tooltip)
        self.summary_label.setToolTip(tooltip)
        self.chart.setToolTip(tooltip)


class _CorrelationColumnWidget(QFrame):
    """Show one colored correlation column as one clickable filter block."""

    columnClicked = pyqtSignal(str)

    def __init__(self, title: str, band: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.band = str(band)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.header_button = QPushButton(title, self)
        self.header_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header_button.clicked.connect(lambda: self.columnClicked.emit(self.band))
        self.summary_label = QLabel('-', self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.summary_label.setStyleSheet('color: #c9d3df; background: transparent; border: none;')
        layout.addWidget(self.header_button)
        layout.addWidget(self.summary_label)
        layout.addStretch(1)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self._refresh_style(False)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.columnClicked.emit(self.band)
            event.accept()
            return
        super().mousePressEvent(event)

    def _refresh_style(self, active: bool) -> None:
        palette = ('#8c2f39', '#ffe9ec', '#b84a58') if self.band == 'bad' else ('#1f5f3b', '#e9fff1', '#2d7a50')
        border_width = '2px' if active else '1px'
        self.setStyleSheet(
            'QFrame {'
            f'background-color: {palette[0]}; color: {palette[1]}; border: {border_width} solid {palette[2]}; '
            'border-radius: 10px;'
            '}'
            'QLabel { color: #f4f7fb; background: transparent; border: none; }'
        )
        self.header_button.setStyleSheet(
            'QPushButton {'
            f'background-color: rgba(0, 0, 0, 0.10); color: {palette[1]}; border: 0px; '
            'border-radius: 8px; padding: 8px 10px; font-weight: 700; text-align: left;'
            '}'
        )

    def set_payload(self, frame_count: int, mean_hits: float, max_hits: int, *, active: bool) -> None:
        self._refresh_style(active)
        translator = Translator()
        if frame_count <= 0:
            self.summary_label.setText(translator.tr("hist.no_repeated"))
            return
        self.summary_label.setText(
            f'{translator.tr("hist.frames_in_band")}: {int(frame_count)}\n'
            f'{translator.tr("hist.mean_hits")}: {float(mean_hits):.2f}\n'
            f'{translator.tr("hist.max_hits")}: {int(max_hits)}'
        )



class ValidationGradientExtendWidget(QWidget):

    """Embeddable widget for multi-model segmentation quality evaluation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(EXTEND_ROOT_OBJECT_NAME)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(EXTEND_WIDGET_STYLESHEET)
        self._settings_service = ValidationGradientLiteSettingsService(QSettings(SETTINGS_ORG, SETTINGS_APP))
        language = self._settings_service.load_language()
        set_current_language(language)
        self._i18n = Translator(language)
        self._t = self._i18n.tr

        self._build_ui()
        self._setup_menu_bar()

        self._presenter = ValidationGradientExtendPresenter(self, self._settings_service)
        self._connect_signals()
        self._presenter._restore_persisted_state()
        self._presenter._refresh_folder_rows()
        self._presenter._sync_action_buttons()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._menu_bar = QMenuBar(self)
        root_layout.addWidget(self._menu_bar)

        content = QWidget(self)
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.addWidget(content, stretch=1)

        splitter = QSplitter(Qt.Orientation.Horizontal, content)
        content_layout.addWidget(splitter)

        control_scroll = QScrollArea(splitter)
        control_scroll.setWidgetResizable(True)
        control_host = QWidget(control_scroll)
        control_layout = QVBoxLayout(control_host)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_scroll.setWidget(control_host)

        folders_group = QGroupBox(self._t("folders.group"), control_host)
        self.folders_group = folders_group
        folders_layout = QVBoxLayout(folders_group)
        folders_info = QLabel(self._t("folders.info"), folders_group)
        self.folders_info_label = folders_info
        folders_info.setWordWrap(True)
        folders_info.hide()
        folders_layout.addWidget(folders_info)

        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(4)
        style = self.style()

        self.btn_add_folder = QToolButton(folders_group)
        self.btn_add_folder.setAutoRaise(True)
        self.btn_add_folder.setProperty("toolbarButton", True)
        self.btn_add_folder.setProperty("liteToolbarButton", True)
        self.btn_add_folder.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder))
        self.btn_add_folder.setToolTip(self._t("folders.add_model"))
        toolbar_layout.addWidget(self.btn_add_folder)

        self.btn_clear_folders = QToolButton(folders_group)
        self.btn_clear_folders.setAutoRaise(True)
        self.btn_clear_folders.setProperty("toolbarButton", True)
        self.btn_clear_folders.setText("x")
        self.btn_clear_folders.setToolTip(self._t("folders.clear_models"))
        toolbar_layout.addWidget(self.btn_clear_folders)

        self.btn_build = QToolButton(folders_group)
        self.btn_build.setAutoRaise(True)
        self.btn_build.setProperty("toolbarButton", True)
        self.btn_build.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.btn_build.setToolTip(self._t("folders.build"))
        toolbar_layout.addWidget(self.btn_build)

        self.btn_compute = QToolButton(folders_group)
        self.btn_compute.setAutoRaise(True)
        self.btn_compute.setProperty("toolbarButton", True)
        self.btn_compute.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.btn_compute.setToolTip(self._t("folders.compute_mismatch"))
        toolbar_layout.addWidget(self.btn_compute)

        # Legacy export button disabled in current UI.
        # self.btn_export = QToolButton(folders_group)
        # self.btn_export.setAutoRaise(True)
        # self.btn_export.setProperty("toolbarButton", True)
        # self.btn_export.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        # self.btn_export.setToolTip("Export ranked frames")
        # toolbar_layout.addWidget(self.btn_export)

        self.btn_cancel = QToolButton(folders_group)
        self.btn_cancel.setAutoRaise(True)
        self.btn_cancel.setProperty("toolbarButton", True)
        self.btn_cancel.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_BrowserStop))
        self.btn_cancel.setToolTip(self._t("folders.cancel"))
        toolbar_layout.addWidget(self.btn_cancel)
        toolbar_layout.addStretch(1)
        folders_layout.addLayout(toolbar_layout)

        self.build_progress = QProgressBar(folders_group)
        self.build_progress.setTextVisible(True)
        self.build_progress.setRange(0, 1)
        self.build_progress.setValue(0)
        self.build_progress.hide()
        folders_layout.addWidget(self.build_progress)

        self.folder_list = QListWidget(folders_group)
        self.folder_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.folder_list.setSpacing(2)
        folders_layout.addWidget(self.folder_list, stretch=1)
        control_layout.addWidget(folders_group)

        source_group = QGroupBox(self._t("sources.group"), control_host)
        self.source_group = source_group
        source_layout = QFormLayout(source_group)
        source_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        original_row = QWidget(source_group)
        original_row_layout = QHBoxLayout(original_row)
        original_row_layout.setContentsMargins(0, 0, 0, 0)
        original_row_layout.setSpacing(6)
        self.original_folder_value = QLabel(self._t("sources.not_set"), original_row)
        self.original_folder_value.setWordWrap(True)
        self.original_folder_value.setMinimumWidth(0)
        self.original_folder_value.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.btn_set_original = QToolButton(original_row)
        self.btn_set_original.setText(self._t("common.set"))
        self.btn_clear_original = QToolButton(original_row)
        self.btn_clear_original.setText(self._t("common.clear"))
        original_row_layout.addWidget(self.original_folder_value, stretch=1)
        original_row_layout.addWidget(self.btn_set_original)
        original_row_layout.addWidget(self.btn_clear_original)
        self.original_source_label = QLabel(self._t("sources.original"), source_group)
        source_layout.addRow(self.original_source_label, original_row)

        gt_row = QWidget(source_group)
        gt_row_layout = QHBoxLayout(gt_row)
        gt_row_layout.setContentsMargins(0, 0, 0, 0)
        gt_row_layout.setSpacing(6)
        self.gt_folder_value = QLabel(self._t("sources.not_set"), gt_row)
        self.gt_folder_value.setWordWrap(True)
        self.gt_folder_value.setMinimumWidth(0)
        self.gt_folder_value.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.btn_set_gt = QToolButton(gt_row)
        self.btn_set_gt.setText(self._t("common.set"))
        self.btn_clear_gt = QToolButton(gt_row)
        self.btn_clear_gt.setText(self._t("common.clear"))
        gt_row_layout.addWidget(self.gt_folder_value, stretch=1)
        gt_row_layout.addWidget(self.btn_set_gt)
        gt_row_layout.addWidget(self.btn_clear_gt)
        self.gt_source_label = QLabel(self._t("sources.ground_truth"), source_group)
        source_layout.addRow(self.gt_source_label, gt_row)
        control_layout.addWidget(source_group)

        control_layout.addStretch(1)

        self.matrix_tabs = QTabWidget(splitter)
        self.matrix_tabs.setTabsClosable(True)
        self.matrix_tabs.setMovable(True)
        self.matrix_tabs.setDocumentMode(True)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes(list(CONTROL_PANEL_SPLITTER_SIZES))

        # Legacy export controls disabled in current UI.
        # self.export_group = QGroupBox("Export", control_host)
        # export_layout = QFormLayout(self.export_group)
        # export_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        # self.export_selection_mode_combo = QComboBox(self.export_group)
        # for label, key in EXPORT_SELECTION_MODE_OPTIONS:
        #     self.export_selection_mode_combo.addItem(label, key)
        # self.export_selection_mode_combo.setCurrentIndex(self.export_selection_mode_combo.findData(DEFAULT_EXPORT_SELECTION_MODE))
        # self.export_top_k_spin = QSpinBox(self.export_group)
        # self.export_top_k_spin.setRange(*EXPORT_TOP_K_RANGE)
        # self.export_top_k_spin.setValue(DEFAULT_TOP_K_EXPORT)
        # self.export_percent_spin = QSpinBox(self.export_group)
        # self.export_percent_spin.setRange(*EXPORT_PERCENT_RANGE)
        # self.export_percent_spin.setValue(DEFAULT_EXPORT_PERCENT)
        # self.export_percentile_spin = QSpinBox(self.export_group)
        # self.export_percentile_spin.setRange(*EXPORT_PERCENTILE_RANGE)
        # self.export_percentile_spin.setValue(DEFAULT_EXPORT_PERCENTILE)
        # self.export_neighbor_radius_spin = QSpinBox(self.export_group)
        # self.export_neighbor_radius_spin.setRange(*EXPORT_NEIGHBOR_RANGE)
        # self.export_neighbor_radius_spin.setValue(DEFAULT_EXPORT_NEIGHBOR_RADIUS)
        # self.filter_to_candidates_checkbox = QCheckBox("Show only selected candidates in matrix", self.export_group)
        # self.filter_to_candidates_checkbox.setChecked(DEFAULT_FILTER_TO_EXPORT_CANDIDATES)
        # self.export_selection_preview = QLabel("Candidates: -", self.export_group)
        # self.export_selection_preview.setWordWrap(True)
        # export_layout.addRow("Selection mode", self.export_selection_mode_combo)
        # export_layout.addRow("Worst frame count", self.export_top_k_spin)
        # export_layout.addRow("Worst percent", self.export_percent_spin)
        # export_layout.addRow("Percentile >=", self.export_percentile_spin)
        # export_layout.addRow("Neighbor radius", self.export_neighbor_radius_spin)
        # export_layout.addRow("Matrix filter", self.filter_to_candidates_checkbox)
        # export_layout.addRow("Selection preview", self.export_selection_preview)
        # control_layout.addWidget(self.export_group)

        self.thumbnail_size_spin = QSpinBox(self)
        self.thumbnail_size_spin.setRange(DEFAULT_CELL_SIZE, DEFAULT_CELL_SIZE)
        self.thumbnail_size_spin.setValue(DEFAULT_CELL_SIZE)
        self.thumbnail_size_spin.setEnabled(False)

        self.layout_mode_combo = QComboBox(self)
        self._populate_layout_mode_combo(DEFAULT_MATRIX_LAYOUT_MODE)

        self.total_frames_spin = QSpinBox(self)
        self.total_frames_spin.setRange(*TOTAL_FRAMES_RANGE)
        self.total_frames_spin.setValue(DEFAULT_TOTAL_FRAMES)
        self.frames_per_row_spin = QSpinBox(self)
        self.frames_per_row_spin.setRange(*FRAMES_PER_ROW_RANGE)
        self.frames_per_row_spin.setValue(DEFAULT_FRAMES_PER_ROW)
        self.matrix_rows_spin = QSpinBox(self)
        self.matrix_rows_spin.setRange(*MATRIX_ROWS_RANGE)
        self.matrix_rows_spin.setValue(DEFAULT_MATRIX_ROWS)
        self.matrix_columns_spin = QSpinBox(self)
        self.matrix_columns_spin.setRange(*MATRIX_COLUMNS_RANGE)
        self.matrix_columns_spin.setValue(DEFAULT_MATRIX_COLUMNS)
        self.analysis_mode_combo = QComboBox(self)
        self._populate_analysis_mode_combo(DEFAULT_ANALYSIS_MODE)
        self.geometry_mode_combo = QComboBox(self)
        self._populate_geometry_mode_combo(DEFAULT_GEOMETRY_MODE)
        self.mask_threshold_spin = QDoubleSpinBox(self)
        self.mask_threshold_spin.setRange(*MASK_THRESHOLD_RANGE)
        self.mask_threshold_spin.setSingleStep(0.05)
        self.mask_threshold_spin.setDecimals(2)
        self.mask_threshold_spin.setValue(DEFAULT_MASK_THRESHOLD)
        self.boundary_radius_spin = QSpinBox(self)
        self.boundary_radius_spin.setRange(*BOUNDARY_RADIUS_RANGE)
        self.boundary_radius_spin.setValue(DEFAULT_BOUNDARY_RADIUS)
        self.confidence_uncertainty_delta_spin = QDoubleSpinBox(self)
        self.confidence_uncertainty_delta_spin.setRange(*CONFIDENCE_UNCERTAINTY_DELTA_RANGE)
        self.confidence_uncertainty_delta_spin.setSingleStep(0.01)
        self.confidence_uncertainty_delta_spin.setDecimals(2)
        self.confidence_uncertainty_delta_spin.setValue(DEFAULT_CONFIDENCE_UNCERTAINTY_DELTA)
        self.polygon_confidence_summary_combo = QComboBox(self)
        self._populate_polygon_confidence_summary_combo(DEFAULT_POLYGON_CONFIDENCE_SUMMARY)
        self.point_match_radius_spin = QDoubleSpinBox(self)
        self.point_match_radius_spin.setRange(*POINT_MATCH_RADIUS_RANGE)
        self.point_match_radius_spin.setSingleStep(0.5)
        self.point_match_radius_spin.setDecimals(1)
        self.point_match_radius_spin.setValue(DEFAULT_POINT_MATCH_RADIUS)
        self.point_confidence_radius_spin = QSpinBox(self)
        self.point_confidence_radius_spin.setRange(*POINT_CONFIDENCE_RADIUS_RANGE)
        self.point_confidence_radius_spin.setValue(DEFAULT_POINT_CONFIDENCE_RADIUS)
        self.point_extraction_mode_combo = QComboBox(self)
        self._populate_point_extraction_mode_combo(DEFAULT_POINT_EXTRACTION_MODE)

        self.metric_group_combo = QComboBox(self)
        for label, key in MATRIX_METRIC_GROUP_OPTIONS:
            self.metric_group_combo.addItem(self._t(label), key)
        self.metric_group_combo.hide()
        self.metric_scope_combo = QComboBox(self)
        self._populate_metric_scope_combo(None, DEFAULT_METRIC_SCOPE)
        self.metric_combo = QComboBox(self)
        self._populate_metric_combo(DEFAULT_MATRIX_METRIC_KEY)
        self.frame_type_filter_combo = QComboBox(self)
        self._populate_frame_type_filter_combo('all')

        self.gradient_selector = GradientPresetSelectorWidget(self)
        self.gradient_range_selector = GradientRangeSelectorWidget(self)
        self.gradient_selector.set_selected_gradient(DEFAULT_GRADIENT_NAME, emit_signal=False)
        self.gradient_range_selector.set_gradient_name(DEFAULT_GRADIENT_NAME)
        self.gradient_range_selector.set_error_window(*DEFAULT_ERROR_WINDOW)

        self.language_toggle_button = QToolButton(self._menu_bar)
        self.language_toggle_button.setAutoRaise(True)
        self.language_toggle_button.setObjectName(EXTEND_LANGUAGE_BUTTON_OBJECT_NAME)
        self.language_toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_language_toggle_button()

    def _setup_menu_bar(self) -> None:
        self._menu_bar.clear()
        matrix_menu = self._menu_bar.addMenu(self._t("menu.matrix"))
        self._add_menu_widget(matrix_menu, self._build_matrix_settings_widget())
        metric_menu = self._menu_bar.addMenu(self._t("menu.metric"))
        self._add_menu_widget(metric_menu, self._build_metric_settings_widget())
        view_menu = self._menu_bar.addMenu(self._t("menu.error_view"))
        self._add_menu_widget(view_menu, self._build_error_view_settings_widget())
        self._menu_bar.setCornerWidget(self.language_toggle_button, Qt.Corner.TopRightCorner)

    def _populate_layout_mode_combo(self, selected_mode: str | None) -> None:
        current = str(selected_mode or DEFAULT_MATRIX_LAYOUT_MODE)
        self.layout_mode_combo.blockSignals(True)
        self.layout_mode_combo.clear()
        self.layout_mode_combo.addItem(self._t("matrix.layout.indexed"), "indexed_grid")
        self.layout_mode_combo.addItem(self._t("matrix.layout.manual"), "manual_grid")
        index = self.layout_mode_combo.findData(current)
        self.layout_mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.layout_mode_combo.blockSignals(False)

    def _populate_analysis_mode_combo(self, selected_mode: str | None) -> None:
        current = str(selected_mode or DEFAULT_ANALYSIS_MODE)
        self.analysis_mode_combo.blockSignals(True)
        self.analysis_mode_combo.clear()
        for label_key, key in ANALYSIS_MODE_OPTIONS:
            self.analysis_mode_combo.addItem(self._t(label_key), key)
        index = self.analysis_mode_combo.findData(current)
        self.analysis_mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.analysis_mode_combo.blockSignals(False)

    def _populate_geometry_mode_combo(self, selected_mode: str | None) -> None:
        current = str(selected_mode or DEFAULT_GEOMETRY_MODE)
        self.geometry_mode_combo.blockSignals(True)
        self.geometry_mode_combo.clear()
        labels = {
            "mask": self._t("geometry.mask"),
            "point": self._t("geometry.point"),
        }
        for label, key in GEOMETRY_MODE_OPTIONS:
            if str(key) == "auto":
                continue
            self.geometry_mode_combo.addItem(labels.get(str(key), str(label)), key)
        index = self.geometry_mode_combo.findData(current)
        self.geometry_mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.geometry_mode_combo.blockSignals(False)

    def _populate_polygon_confidence_summary_combo(self, selected_value: str | None) -> None:
        current = str(selected_value or DEFAULT_POLYGON_CONFIDENCE_SUMMARY)
        self.polygon_confidence_summary_combo.blockSignals(True)
        self.polygon_confidence_summary_combo.clear()
        for label_key, key in POLYGON_CONFIDENCE_SUMMARY_OPTIONS:
            self.polygon_confidence_summary_combo.addItem(self._t(label_key), key)
        index = self.polygon_confidence_summary_combo.findData(current)
        self.polygon_confidence_summary_combo.setCurrentIndex(index if index >= 0 else 0)
        self.polygon_confidence_summary_combo.blockSignals(False)

    def _populate_metric_scope_combo(self, build_result: BuildResult | None, selected_scope: str | None) -> None:
        current = str(selected_scope or "")
        self.metric_scope_combo.blockSignals(True)
        self.metric_scope_combo.clear()
        if build_result is not None:
            for spec in build_result.model_specs:
                self.metric_scope_combo.addItem(str(spec.display_name), str(spec.model_id))
        if self.metric_scope_combo.count() <= 0 and current:
            self.metric_scope_combo.addItem(current, current)
        index = self.metric_scope_combo.findData(current)
        if index < 0 and build_result is not None:
            fallback = default_confidence_model_id(build_result)
            index = self.metric_scope_combo.findData(fallback)
        self.metric_scope_combo.setCurrentIndex(index if index >= 0 else 0)
        self.metric_scope_combo.setToolTip(self._t('analysis.confidence_model'))
        self.metric_scope_combo.blockSignals(False)

    def _metric_text_for_key(self, metric_key: str, build_result: BuildResult | None = None) -> str:
        metric_key_text = str(metric_key)
        for label_key, key, _group in MATRIX_METRIC_OPTIONS:
            if str(key) == metric_key_text:
                return self._t(label_key)
        translated = self._t(f"metric.{metric_key_text}")
        if translated != f"metric.{metric_key_text}":
            return translated
        if '::' in metric_key_text:
            family, model_id = metric_key_text.split('::', 1)
            model_name = model_id
            if build_result is not None:
                for spec in build_result.model_specs:
                    if spec.model_id == model_id:
                        model_name = spec.display_name
                        break
            if family == 'model_confidence':
                return f"{self._t('metric.model_confidence')} [{model_name}]"
            if family == 'model_uncertain_fraction':
                return f"{self._t('metric.model_uncertain_fraction')} [{model_name}]"
            if family == 'model_point_contrast':
                return f"{self._t('metric.model_point_contrast')} [{model_name}]"
        return metric_key_text

    def _populate_metric_combo(self, selected_metric_key: str | None) -> None:
        current = str(selected_metric_key or DEFAULT_MATRIX_METRIC_KEY)
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        for label_key, key, _group in MATRIX_METRIC_OPTIONS:
            self.metric_combo.addItem(self._t(label_key), key)
        index = self.metric_combo.findData(current)
        self.metric_combo.setCurrentIndex(index if index >= 0 else 0)
        self.metric_combo.blockSignals(False)

    def _populate_frame_type_filter_combo(self, selected_value: str | None) -> None:
        current = str(selected_value or 'all')
        self.frame_type_filter_combo.blockSignals(True)
        self.frame_type_filter_combo.clear()
        self.frame_type_filter_combo.addItem(self._t('frame_type.all'), 'all')
        self.frame_type_filter_combo.addItem(self._t('frame_type.polygon'), 'polygon')
        self.frame_type_filter_combo.addItem(self._t('frame_type.point'), 'point')
        index = self.frame_type_filter_combo.findData(current)
        self.frame_type_filter_combo.setCurrentIndex(index if index >= 0 else 0)
        self.frame_type_filter_combo.blockSignals(False)

    def _populate_point_extraction_mode_combo(self, selected_value: str | None) -> None:
        current = str(selected_value or DEFAULT_POINT_EXTRACTION_MODE)
        self.point_extraction_mode_combo.blockSignals(True)
        self.point_extraction_mode_combo.clear()
        self.point_extraction_mode_combo.addItem(self._t("point_extraction.component_centroids"), "component_centroids")
        self.point_extraction_mode_combo.addItem(self._t("point_extraction.local_maxima_legacy"), "local_maxima_legacy")
        index = self.point_extraction_mode_combo.findData(current)
        self.point_extraction_mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.point_extraction_mode_combo.blockSignals(False)

    def _build_matrix_settings_widget(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        self._matrix_pixel_size_row = self._build_setting_row(self._t("matrix.pixel_size"), self.thumbnail_size_spin)
        self._matrix_analysis_mode_row = self._build_setting_row(self._t("analysis.mode"), self.analysis_mode_combo)
        self._matrix_geometry_row = self._build_setting_row(self._t("analysis.object_type"), self.geometry_mode_combo)
        self._matrix_threshold_row = self._build_setting_row(self._t('matrix.threshold'), self.mask_threshold_spin)
        self._matrix_boundary_row = self._build_setting_row(self._t('matrix.boundary_radius'), self.boundary_radius_spin)
        self._matrix_confidence_delta_row = self._build_setting_row(self._t('matrix.confidence_delta'), self.confidence_uncertainty_delta_spin)
        self._matrix_polygon_confidence_summary_row = self._build_setting_row(self._t('matrix.polygon_confidence_summary'), self.polygon_confidence_summary_combo)
        self._matrix_point_radius_row = self._build_setting_row(self._t('matrix.point_match_radius'), self.point_match_radius_spin)
        self._matrix_point_confidence_radius_row = self._build_setting_row(self._t('matrix.point_confidence_radius'), self.point_confidence_radius_spin)
        self._matrix_point_mode_row = self._build_setting_row(self._t('matrix.point_extraction_mode'), self.point_extraction_mode_combo)
        self._matrix_layout_row = self._build_setting_row(self._t("matrix.layout"), self.layout_mode_combo)
        self._matrix_frame_type_filter_row = self._build_setting_row(self._t('matrix.frame_type_filter'), self.frame_type_filter_combo)
        self._matrix_total_frames_row = self._build_setting_row(self._t("matrix.total_frames"), self.total_frames_spin)
        self._matrix_frames_per_row_row = self._build_setting_row(self._t("matrix.frames_per_row"), self.frames_per_row_spin)
        self._matrix_rows_row = self._build_setting_row(self._t("matrix.rows"), self.matrix_rows_spin)
        self._matrix_columns_row = self._build_setting_row(self._t("matrix.columns"), self.matrix_columns_spin)
        for row in (
            self._matrix_pixel_size_row,
            self._matrix_analysis_mode_row,
            self._matrix_geometry_row,
            self._matrix_threshold_row,
            self._matrix_boundary_row,
            self._matrix_confidence_delta_row,
            self._matrix_polygon_confidence_summary_row,
            self._matrix_point_radius_row,
            self._matrix_point_confidence_radius_row,
            self._matrix_point_mode_row,
            self._matrix_layout_row,
            self._matrix_frame_type_filter_row,
            self._matrix_total_frames_row,
            self._matrix_frames_per_row_row,
            self._matrix_rows_row,
            self._matrix_columns_row,
        ):
            layout.addWidget(row)
        self._matrix_pixel_size_row.setVisible(False)
        is_indexed_layout = str(self.layout_mode_combo.currentData() or DEFAULT_MATRIX_LAYOUT_MODE) == "indexed_grid"
        self._matrix_total_frames_row.setVisible(is_indexed_layout)
        self._matrix_frames_per_row_row.setVisible(is_indexed_layout)
        self._matrix_rows_row.setVisible(not is_indexed_layout)
        self._matrix_columns_row.setVisible(not is_indexed_layout)
        layout.addStretch(1)
        return widget

    def _build_metric_settings_widget(self) -> QWidget:
        widget = QWidget(self)
        widget.setMinimumWidth(METRIC_SETTINGS_WIDGET_MIN_WIDTH)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        self.metric_scope_combo.setMinimumContentsLength(METRIC_SETTINGS_COMBO_MIN_CONTENTS_LENGTH)
        self.metric_scope_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.metric_combo.setMinimumContentsLength(METRIC_SETTINGS_COMBO_MIN_CONTENTS_LENGTH)
        self.metric_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._metric_scope_row = self._build_setting_row(
            self._t("analysis.confidence_model"),
            self.metric_scope_combo,
            label_min_width=METRIC_SETTINGS_LABEL_MIN_WIDTH,
        )
        layout.addWidget(self._metric_scope_row)
        self._metric_select_row = self._build_setting_row(
            self._t("menu.metric.select"),
            self.metric_combo,
            label_min_width=METRIC_SETTINGS_LABEL_MIN_WIDTH,
        )
        layout.addWidget(self._metric_select_row)
        layout.addStretch(1)
        return widget

    def _build_error_view_settings_widget(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addWidget(self.gradient_selector)
        layout.addWidget(self.gradient_range_selector)
        layout.addStretch(1)
        return widget

    def _update_language_toggle_button(self) -> None:
        current_language = str(self._i18n.language or "en").lower()
        next_language = "RU" if current_language == "en" else "EN"
        self.language_toggle_button.setText(next_language)
        self.language_toggle_button.setToolTip(self._t("language.toggle_tooltip"))

    def retranslate_ui(self) -> None:
        self._t = self._i18n.tr
        self._setup_menu_bar()
        self._update_language_toggle_button()
        self.folders_group.setTitle(self._t("folders.group"))
        self.folders_info_label.setText(self._t("folders.info"))
        self.btn_add_folder.setToolTip(self._t("folders.add_model"))
        self.btn_clear_folders.setToolTip(self._t("folders.clear_models"))
        self.btn_build.setToolTip(self._t("folders.build"))
        self.btn_compute.setToolTip(self._t("folders.compute_mismatch"))
        self.btn_cancel.setToolTip(self._t("folders.cancel"))
        self.source_group.setTitle(self._t("sources.group"))
        self.original_source_label.setText(self._t("sources.original"))
        self.gt_source_label.setText(self._t("sources.ground_truth"))
        self.btn_set_original.setText(self._t("common.set"))
        self.btn_clear_original.setText(self._t("common.clear"))
        self.btn_set_gt.setText(self._t("common.set"))
        self.btn_clear_gt.setText(self._t("common.clear"))
        current_layout = str(self.layout_mode_combo.currentData() or DEFAULT_MATRIX_LAYOUT_MODE)
        current_analysis_mode = str(self.analysis_mode_combo.currentData() or DEFAULT_ANALYSIS_MODE)
        current_geometry = str(self.geometry_mode_combo.currentData() or DEFAULT_GEOMETRY_MODE)
        current_metric_scope = str(self.metric_scope_combo.currentData() or DEFAULT_METRIC_SCOPE)
        current_metric = str(self.metric_combo.currentData() or DEFAULT_MATRIX_METRIC_KEY)
        current_frame_type_filter = str(self.frame_type_filter_combo.currentData() or 'all')
        current_point_extraction_mode = str(self.point_extraction_mode_combo.currentData() or DEFAULT_POINT_EXTRACTION_MODE)
        current_polygon_confidence_summary = str(self.polygon_confidence_summary_combo.currentData() or DEFAULT_POLYGON_CONFIDENCE_SUMMARY)
        self._populate_layout_mode_combo(current_layout)
        self._populate_analysis_mode_combo(current_analysis_mode)
        self._populate_geometry_mode_combo(current_geometry)
        self._populate_point_extraction_mode_combo(current_point_extraction_mode)
        self._populate_polygon_confidence_summary_combo(current_polygon_confidence_summary)
        current_state = self._presenter._current_tab_state() if hasattr(self, "_presenter") else None
        if current_state is not None:
            self._presenter._sync_metric_controls(
                current_state.build_result,
                preferred_metric_key=current_metric,
                preferred_scope_key=current_metric_scope,
                context_state=current_state,
            )
        else:
            self._populate_metric_scope_combo(None, current_metric_scope)
            self._populate_metric_combo(current_metric)
        self._populate_frame_type_filter_combo(current_frame_type_filter)
        for row, key in (
            (getattr(self, "_matrix_pixel_size_row", None), "matrix.pixel_size"),
            (getattr(self, "_matrix_analysis_mode_row", None), "analysis.mode"),
            (getattr(self, "_matrix_geometry_row", None), "analysis.object_type"),
            (getattr(self, "_matrix_threshold_row", None), "matrix.threshold"),
            (getattr(self, "_matrix_boundary_row", None), "matrix.boundary_radius"),
            (getattr(self, "_matrix_confidence_delta_row", None), "matrix.confidence_delta"),
            (getattr(self, "_matrix_polygon_confidence_summary_row", None), "matrix.polygon_confidence_summary"),
            (getattr(self, "_matrix_point_radius_row", None), "matrix.point_match_radius"),
            (getattr(self, "_matrix_point_confidence_radius_row", None), "matrix.point_confidence_radius"),
            (getattr(self, "_matrix_point_mode_row", None), "matrix.point_extraction_mode"),
            (getattr(self, "_matrix_layout_row", None), "matrix.layout"),
            (getattr(self, "_matrix_frame_type_filter_row", None), "matrix.frame_type_filter"),
            (getattr(self, "_matrix_total_frames_row", None), "matrix.total_frames"),
            (getattr(self, "_matrix_frames_per_row_row", None), "matrix.frames_per_row"),
            (getattr(self, "_matrix_rows_row", None), "matrix.rows"),
            (getattr(self, "_matrix_columns_row", None), "matrix.columns"),
            (getattr(self, "_metric_scope_row", None), "analysis.confidence_model"),
            (getattr(self, "_metric_select_row", None), "menu.metric.select"),
        ):
            label = getattr(row, "_title_label", None)
            if label is not None:
                label.setText(self._t(key))
        for state in getattr(self, "_presenter", None)._tab_states.values() if hasattr(getattr(self, "_presenter", None), "_tab_states") else ():
            state.preview.group.setTitle(self._t("matrix.preview.group"))
            state.preview.frame_title.setText(self._t("matrix.preview.frame"))
            for metric_key, card in state.preview.score_cards.items():
                if hasattr(card, "title_label"):
                    card.title_label.setText(self._metric_text_for_key(metric_key, state.build_result))
            for metric_key, card in state.preview.histogram_cards.items():
                if hasattr(card, "title_label"):
                    card.title_label.setText(self._metric_text_for_key(metric_key, state.build_result))
            if state.content_tabs is not None:
                state.content_tabs.setTabText(0, self._t("tab.matrix"))
                state.content_tabs.setTabText(1, self._t("tab.percentiles"))
            if state.repeated_bad_column is not None:
                state.repeated_bad_column.header_button.setText(self._t("correlation.bad"))
            if state.repeated_good_column is not None:
                state.repeated_good_column.header_button.setText(self._t("correlation.good"))
        window = self.window()
        if isinstance(window, QMainWindow):
            window.setWindowTitle(self._t("window.title"))

    def _toggle_language(self) -> None:
        current_language = str(self._i18n.language or "en").lower()
        language = "ru" if current_language == "en" else "en"
        self._i18n.set_language(language)
        self._settings_service.save_language(language)
        self.retranslate_ui()

    def _build_setting_row(self, title: str, control: QWidget, *, label_min_width: int = SETTINGS_LABEL_MIN_WIDTH) -> QWidget:
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label = QLabel(title, row)
        label.setMinimumWidth(int(label_min_width))
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addWidget(control, stretch=1)
        row._title_label = label  # type: ignore[attr-defined]
        return row

    def _build_preview_panel(self, parent: QWidget, metric_keys: tuple[str, ...], build_result: BuildResult) -> ExtendPreviewPanel:
        group = QGroupBox(self._t("matrix.preview.group"), parent)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        frame_row = QWidget(group)
        frame_layout = QHBoxLayout(frame_row)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(8)
        frame_title = QLabel(self._t("matrix.preview.frame"), frame_row)
        frame_value = QLabel("-", frame_row)
        frame_value.setWordWrap(True)
        frame_value.setMinimumWidth(0)
        frame_layout.addWidget(frame_title)
        frame_layout.addWidget(frame_value, stretch=1)
        layout.addWidget(frame_row)

        scores_host = QWidget(group)
        scores_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scores_layout = QVBoxLayout(scores_host)
        scores_layout.setContentsMargins(0, 0, 0, 0)
        scores_layout.setSpacing(8)
        score_cards: dict[str, QWidget] = {}
        seen_keys: set[str] = set()
        for metric_key in metric_keys:
            if metric_key in seen_keys:
                continue
            seen_keys.add(metric_key)
            card = _ExpandableScoreCard(self._metric_text_for_key(metric_key, build_result), scores_host)
            card.hide()
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            scores_layout.addWidget(card)
            score_cards[str(metric_key)] = card
        scores_layout.addStretch(1)
        layout.addWidget(scores_host, stretch=1)

        overall_title = QLabel(self._t("metric.overall_frame_score"), group)
        overall_value = QLabel("-", group)
        metric_title = QLabel(self._t("matrix.metric"), group)
        metric_value = QLabel("-", group)
        labeled_title = QLabel(self._t("metric.labeled_best_quality"), group)
        labeled_value = QLabel("-", group)
        acquisition_title = QLabel(self._t("metric.acquisition_score"), group)
        acquisition_value = QLabel("-", group)
        overall_title.hide()
        overall_value.hide()
        metric_title.hide()
        metric_value.hide()
        labeled_title.hide()
        labeled_value.hide()
        acquisition_title.hide()
        acquisition_value.hide()
        return ExtendPreviewPanel(group, frame_title, frame_value, overall_title, overall_value, metric_title, metric_value, labeled_title, labeled_value, acquisition_title, acquisition_value, score_cards, {})

    def _build_histograms_panel(self, parent: QWidget, metric_keys: tuple[str, ...], build_result: BuildResult) -> tuple[QWidget, dict[str, QWidget], QWidget, QWidget]:
        panel = QWidget(parent)
        panel_layout = QHBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(10)

        charts_scroll = QScrollArea(panel)
        charts_scroll.setWidgetResizable(True)
        charts_host = QWidget(charts_scroll)
        charts_layout = QVBoxLayout(charts_host)
        charts_layout.setContentsMargins(8, 8, 8, 8)
        charts_layout.setSpacing(8)
        histogram_cards: dict[str, QWidget] = {}
        seen_keys: set[str] = set()
        for metric_key in metric_keys:
            if metric_key in seen_keys:
                continue
            seen_keys.add(metric_key)
            card = _PercentileHistogramCard(self._metric_text_for_key(metric_key, build_result), metric_key, charts_host)
            card.hide()
            charts_layout.addWidget(card)
            histogram_cards[str(metric_key)] = card
        charts_layout.addStretch(1)
        charts_scroll.setWidget(charts_host)
        panel_layout.addWidget(charts_scroll, stretch=3)

        columns_host = QWidget(panel)
        columns_layout = QHBoxLayout(columns_host)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.setSpacing(8)
        repeated_bad_column = _CorrelationColumnWidget(self._t("correlation.bad"), 'bad', columns_host)
        repeated_good_column = _CorrelationColumnWidget(self._t("correlation.good"), 'good', columns_host)
        columns_layout.addWidget(repeated_bad_column, 0, Qt.AlignmentFlag.AlignTop)
        columns_layout.addWidget(repeated_good_column, 0, Qt.AlignmentFlag.AlignTop)
        columns_layout.addStretch(1)
        panel_layout.addWidget(columns_host, stretch=2)

        return panel, histogram_cards, repeated_bad_column, repeated_good_column

    def _add_menu_widget(self, menu: QMenu, widget: QWidget) -> None:
        action = QWidgetAction(menu)
        action.setDefaultWidget(widget)
        menu.addAction(action)

    def _connect_signals(self) -> None:
        self.btn_add_folder.clicked.connect(self._presenter._add_folder)
        self.btn_clear_folders.clicked.connect(self._presenter._clear_folders)
        self.btn_set_original.clicked.connect(self._presenter._set_original_folder)
        self.btn_clear_original.clicked.connect(self._presenter._clear_original_folder)
        self.btn_set_gt.clicked.connect(self._presenter._set_gt_folder)
        self.btn_clear_gt.clicked.connect(self._presenter._clear_gt_folder)
        self.btn_build.clicked.connect(self._presenter._start_build)
        self.btn_compute.clicked.connect(self._presenter._start_compute_analytics)
        # Legacy export signal bindings disabled with export UI.
        # self.btn_export.clicked.connect(self._presenter._export_ranked)
        self.btn_cancel.clicked.connect(self._presenter._request_cancel_build)
        # self.export_selection_mode_combo.currentIndexChanged.connect(self._presenter._on_export_selection_changed)
        # self.export_top_k_spin.valueChanged.connect(self._presenter._on_export_selection_changed)
        # self.export_percent_spin.valueChanged.connect(self._presenter._on_export_selection_changed)
        # self.export_percentile_spin.valueChanged.connect(self._presenter._on_export_selection_changed)
        # self.export_neighbor_radius_spin.valueChanged.connect(self._presenter._on_export_selection_changed)
        # self.filter_to_candidates_checkbox.toggled.connect(self._presenter._on_export_selection_changed)

        self.layout_mode_combo.currentIndexChanged.connect(self._presenter._on_matrix_visual_parameter_changed)
        self.thumbnail_size_spin.valueChanged.connect(self._presenter._on_matrix_visual_parameter_changed)
        self.analysis_mode_combo.currentIndexChanged.connect(self._presenter._on_analysis_mode_changed)
        self.geometry_mode_combo.currentIndexChanged.connect(self._presenter._on_object_type_changed)
        self.mask_threshold_spin.valueChanged.connect(self._presenter._sync_action_buttons)
        self.boundary_radius_spin.valueChanged.connect(self._presenter._sync_action_buttons)
        self.confidence_uncertainty_delta_spin.valueChanged.connect(self._presenter._sync_action_buttons)
        self.polygon_confidence_summary_combo.currentIndexChanged.connect(self._presenter._sync_action_buttons)
        self.point_match_radius_spin.valueChanged.connect(self._presenter._sync_action_buttons)
        self.point_confidence_radius_spin.valueChanged.connect(self._presenter._sync_action_buttons)
        self.point_extraction_mode_combo.currentIndexChanged.connect(self._presenter._sync_action_buttons)
        self.total_frames_spin.valueChanged.connect(self._presenter._on_matrix_visual_parameter_changed)
        self.frames_per_row_spin.valueChanged.connect(self._presenter._on_matrix_visual_parameter_changed)
        self.matrix_rows_spin.valueChanged.connect(self._presenter._on_matrix_visual_parameter_changed)
        self.matrix_columns_spin.valueChanged.connect(self._presenter._on_matrix_visual_parameter_changed)
        self.metric_scope_combo.currentIndexChanged.connect(self._presenter._on_metric_scope_changed)
        self.metric_combo.currentIndexChanged.connect(self._presenter._on_metric_changed)
        self.frame_type_filter_combo.currentIndexChanged.connect(self._presenter._on_frame_type_filter_changed)
        self.language_toggle_button.clicked.connect(self._toggle_language)
        self.gradient_selector.gradientChanged.connect(self._presenter._on_gradient_preset_changed)
        self.gradient_range_selector.rangeChanged.connect(self._presenter._on_error_window_changed)

        self.matrix_tabs.currentChanged.connect(self._presenter._on_current_tab_changed)
        self.matrix_tabs.tabCloseRequested.connect(self._presenter._close_matrix_tab)

    def _create_matrix_tab(self, build_result: BuildResult, snapshot: dict[str, object]) -> ExtendMatrixTabState:
        host = QWidget(self.matrix_tabs)
        matrix_layout = QHBoxLayout(host)
        matrix_layout.setContentsMargins(0, 0, 0, 0)

        content_tabs = QTabWidget(host)
        matrix_page = QWidget(content_tabs)
        matrix_page_layout = QVBoxLayout(matrix_page)
        matrix_page_layout.setContentsMargins(0, 0, 0, 0)
        matrix_view = MatrixListWidget(matrix_page)
        matrix_page_layout.addWidget(matrix_view, stretch=1)
        charts_page, histogram_cards, repeated_bad_column, repeated_good_column = self._build_histograms_panel(content_tabs, tuple(build_result.available_metric_keys), build_result)
        content_tabs.addTab(matrix_page, self._t("tab.matrix"))
        content_tabs.addTab(charts_page, self._t("tab.percentiles"))
        matrix_layout.addWidget(content_tabs, stretch=1)

        overview_host = QWidget(host)
        overview_layout = QVBoxLayout(overview_host)
        overview_layout.setContentsMargins(0, 0, 0, 0)
        mini_map = MatrixMiniMapWidget(overview_host)
        overview_layout.addWidget(mini_map)
        preview = self._build_preview_panel(overview_host, tuple(build_result.available_metric_keys), build_result)
        overview_layout.addWidget(preview.group)
        overview_layout.addStretch(1)
        overview_host.setMaximumWidth(OVERVIEW_PANEL_MAX_WIDTH)
        matrix_layout.addWidget(overview_host)

        preview.histogram_cards = histogram_cards
        state = ExtendMatrixTabState(
            widget=host,
            matrix_view=matrix_view,
            mini_map=mini_map,
            build_result=build_result,
            content_tabs=content_tabs,
            cell_size=int(snapshot["cell_size"]),
            layout_config=snapshot["layout_config"],
            gradient_name=str(snapshot["gradient_name"]),
            error_window=tuple(snapshot["error_window"]),
            metric_key=str(snapshot.get("metric_key") or DEFAULT_MATRIX_METRIC_KEY),
            metric_scope=str(snapshot.get("metric_scope") or ""),
            analysis_mode=str(snapshot.get("analysis_mode") or DEFAULT_ANALYSIS_MODE),
            object_type=str(snapshot.get("object_type") or "polygon"),
            confidence_model_id=str(snapshot.get("confidence_model_id") or snapshot.get("metric_scope") or "") or None,
            frame_type_filter=str(snapshot.get("frame_type_filter") or snapshot.get("object_type") or "all"),
            preview=preview,
            repeated_bad_column=repeated_bad_column,
            repeated_good_column=repeated_good_column,
        )
        matrix_view.recordSelected.connect(lambda record, s=state: self._presenter._on_record_selected(s, record))
        matrix_view.recordActivated.connect(lambda record, s=state: self._presenter._open_record_details(record, s))
        matrix_view.overviewChanged.connect(lambda image, visible_rect, selected_position, selected_blink_on, processing_positions, reference_position, s=state: self._presenter._on_matrix_overview_changed(s, image, visible_rect, selected_position, selected_blink_on, processing_positions, reference_position))
        return state

    def shutdown(self) -> None:
        self._presenter.shutdown()

    def closeEvent(self, event) -> None:
        self._presenter.shutdown()
        super().closeEvent(event)


class ValidationGradientExtendMainWindow(QMainWindow):
    """Standalone host window for the extended widget."""

    def __init__(self) -> None:
        super().__init__()
        self._widget = ValidationGradientExtendWidget(self)
        self.setWindowTitle(self._widget._t("window.title"))
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        self.setCentralWidget(self._widget)

    def plugin_widget(self) -> ValidationGradientExtendWidget:
        return self._widget

    def closeEvent(self, event) -> None:
        self._widget.shutdown()
        super().closeEvent(event)


# Backward-compatible aliases for legacy lite imports.
ValidationGradientLiteWidget = ValidationGradientExtendWidget
ValidationGradientLiteMainWindow = ValidationGradientExtendMainWindow

