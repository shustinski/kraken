"""Main window for the extended validation gradient widget."""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QSettings, QRectF, QSignalBlocker, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMenuBar,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QSpinBox,
    QStyle,
    QTabWidget,
    QTableWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)
from kraken_core.analysis_protocol import AnalysisProfileKind

from ..infra.services import KarakalSettingsService
from ..updater import (
    QtUpdateController,
    create_karakal_update_controller,
    load_karakal_update_channel,
    load_karakal_update_client_config,
    save_karakal_update_channel,
)
from ..core.analysis_modes import ANALYSIS_MODE_OPTIONS, default_confidence_model_id
from ..core.analysis_profiles import AnalysisPreflightReport, DEFAULT_ANALYSIS_PROFILE
from ..core.domain import BuildResult
from ..core.performance import PerformanceConfig
from ..ui.app_icon import apply_karakal_icon
from ..version import __version__
from ..ui.i18n import Translator, set_current_language
from ..ui.analysis_setup import AnalysisSetupPanel
from ..ui.matrix_view import MatrixLegendWidget, MatrixListWidget, MatrixMiniMapWidget
from ..ui.profiling_dialog import ProfilingDialog
from ..ui.history_dialog import StandaloneHistoryDialog
from ..ui.ui_constants import (
    BOUNDARY_RADIUS_RANGE,
    COMPARISON_TARGET_OPTIONS,
    CONFIDENCE_UNCERTAINTY_PROFILE_OPTIONS,
    CONTROL_PANEL_SPLITTER_SIZES,
    DEFAULT_BOUNDARY_RADIUS,
    DEFAULT_CELL_SIZE,
    DEFAULT_COMPARISON_TARGET,
    DEFAULT_CONFIDENCE_UNCERTAINTY_PROFILE,
    DEFAULT_ANALYSIS_MODE,
    DEFAULT_FRAMES_PER_ROW,
    DEFAULT_GEOMETRY_MODE,
    DEFAULT_GRADIENT_NAME,
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_POLYGON_COMPARE_PROFILE,
    DEFAULT_POINT_CONFIDENCE_RADIUS,
    DEFAULT_POINT_EXTRACTION_MODE,
    DEFAULT_POINT_MATCH_RADIUS,
    DEFAULT_POLYGON_CONFIDENCE_SUMMARY,
    DEFAULT_MATRIX_COLUMNS,
    DEFAULT_MATRIX_LAYOUT_MODE,
    DEFAULT_MATRIX_SCORE_VIEW_MODE,
    DEFAULT_MATRIX_METRIC_KEY,
    DEFAULT_METRIC_SCOPE,
    DEFAULT_MATRIX_ROWS,
    GRID_INSPECTION_DEFAULT_ERROR_TYPES,
    GRID_INSPECTION_DAMAGE_METRIC_KEY,
    GRID_INSPECTION_ERROR_TYPE_OPTIONS,
    grid_inspection_error_type_icon,
    DEFAULT_TOTAL_FRAMES,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    EXTEND_LANGUAGE_BUTTON_OBJECT_NAME,
    EXTEND_ROOT_OBJECT_NAME,
    EXTEND_WIDGET_STYLESHEET,
    FRAMES_PER_ROW_RANGE,
    GEOMETRY_MODE_OPTIONS,
    MASK_THRESHOLD_RANGE,
    MATRIX_COLUMNS_RANGE,
    METRIC_SETTINGS_COMBO_MIN_CONTENTS_LENGTH,
    METRIC_SETTINGS_LABEL_MIN_WIDTH,
    MATRIX_METRIC_GROUP_OPTIONS,
    MATRIX_METRIC_OPTIONS,
    MATRIX_SCORE_VIEW_OPTIONS,
    GRADIENT_LABELS,
    MATRIX_ROWS_RANGE,
    OVERVIEW_PANEL_MAX_WIDTH,
    PERCENTILE_BAND_BOUNDS,
    PERCENTILE_BAND_COLORS,
    PERCENTILE_BAND_LABELS,
    PERCENTILE_BAND_TITLES,
    POINT_CONFIDENCE_RADIUS_RANGE,
    POINT_MATCH_RADIUS_RANGE,
    POLYGON_CONFIDENCE_SUMMARY_OPTIONS,
    POLYGON_COMPARE_PROFILE_OPTIONS,
    SETTINGS_APP,
    SETTINGS_LABEL_MIN_WIDTH,
    SETTINGS_ORG,
    TOTAL_FRAMES_RANGE,
)
from .presenter import KarakalPresenter
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


class _NoWheelSpinBox(QSpinBox):
    """Ignore wheel scrolling to avoid accidental value changes inside the control panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._clamp_to_max_on_edit = False
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAccelerated(False)
        self.lineEdit().textEdited.connect(self._clamp_edited_text_to_maximum)

    def set_clamp_to_max_on_edit(self, enabled: bool) -> None:
        self._clamp_to_max_on_edit = bool(enabled)

    def _clamp_edited_text_to_maximum(self, text: str) -> None:
        if not self._clamp_to_max_on_edit:
            return
        stripped = str(text).strip()
        if not stripped.isdigit():
            return
        value = int(stripped)
        maximum = int(self.maximum())
        if value > maximum:
            self.setValue(maximum)
            self.lineEdit().selectAll()

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()


class _NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Ignore wheel scrolling to avoid accidental value changes inside the control panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAccelerated(False)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()


class _NoWheelComboBox(QComboBox):
    """Ignore wheel scrolling unless the popup list is explicitly open."""

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        view = self.view()
        if view is not None and view.isVisible():
            super().wheelEvent(event)
            return
        event.ignore()


class _PercentileHistogramWidget(QWidget):
    """Draw a compact histogram over fixed percentile bins."""

    binClicked = pyqtSignal(int)
    binDoubleClicked = pyqtSignal(int)
    binContextMenuRequested = pyqtSignal(int, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._counts = [0] * len(PERCENTILE_BAND_BOUNDS)
        self._total = 0
        self._active_bin: int | None = None
        self.setMinimumHeight(120)

    def set_payload(self, counts: list[int], total: int, *, active_bin: int | None = None) -> None:
        expected = len(PERCENTILE_BAND_BOUNDS)
        self._counts = [int(value) for value in counts[:expected]] + [0] * max(0, expected - len(counts))
        self._counts = self._counts[:expected]
        self._total = int(total)
        self._active_bin = None if active_bin is None else int(active_bin)
        self.update()

    def _chart_rect(self) -> QRectF:
        return QRectF(self.rect().adjusted(6, 8, -6, -20))

    def _bar_rects(self, rect: QRectF) -> list[QRectF]:
        if rect.width() <= 0 or rect.height() <= 0:
            return []
        max_count = max(1, max(self._counts, default=1))
        band_count = max(1, len(self._counts))
        bar_width = max(12.0, rect.width() / max(4.0, band_count * 1.25))
        gap = max(6.0, (rect.width() - bar_width * band_count) / max(1.0, band_count - 1.0))
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
        labels = PERCENTILE_BAND_LABELS
        for index, bar_rect in enumerate(self._bar_rects(rect)):
            count = self._counts[index]
            color = QColor(PERCENTILE_BAND_COLORS[index])
            painter.fillRect(bar_rect, color)
            if self._active_bin == index:
                painter.setPen(QPen(QColor('#f5f8fc'), 2.0))
                painter.drawRect(bar_rect.adjusted(0.5, 0.5, -0.5, -0.5))
            painter.setPen(QPen(QColor('#dce7f3')))
            painter.drawText(QRectF(bar_rect.left() - 6.0, bar_rect.top() - 16.0, bar_rect.width() + 12.0, 14.0), Qt.AlignmentFlag.AlignCenter, str(count))
            painter.drawText(QRectF(bar_rect.left() - 8.0, rect.bottom() - 16.0, bar_rect.width() + 16.0, 14.0), Qt.AlignmentFlag.AlignCenter, labels[index])
        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() not in {Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton}:
            super().mousePressEvent(event)
            return
        point = event.position()
        for index, bar_rect in enumerate(self._bar_rects(self._chart_rect())):
            if bar_rect.contains(point):
                if event.button() == Qt.MouseButton.RightButton:
                    self.binContextMenuRequested.emit(index, event.globalPosition().toPoint())
                else:
                    self.binClicked.emit(index)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseDoubleClickEvent(event)
            return
        point = event.position()
        for index, bar_rect in enumerate(self._bar_rects(self._chart_rect())):
            if bar_rect.contains(point):
                self.binDoubleClicked.emit(index)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)


class _PercentileHistogramCard(QWidget):
    """Show one metric percentile distribution as a compact chart card."""

    binClicked = pyqtSignal(str, int)
    binDoubleClicked = pyqtSignal(str, int)
    binContextMenuRequested = pyqtSignal(str, int, object)

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
        self.chart.binDoubleClicked.connect(lambda index: self.binDoubleClicked.emit(self._metric_key, index))
        self.chart.binContextMenuRequested.connect(lambda index, global_pos: self.binContextMenuRequested.emit(self._metric_key, index, global_pos))
        layout.addWidget(self.title_label)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.chart)

    def set_payload(self, title: str, counts: list[int], total: int, *, visible: bool, active_bin: int | None = None, tooltip: str = '') -> None:
        self.setVisible(visible)
        self.title_label.setText(title)
        self.chart.set_payload(counts, total, active_bin=active_bin)
        labels = PERCENTILE_BAND_TITLES
        translator = Translator()
        summary = ' | '.join(f'{label}: {int(count)}' for label, count in zip(labels, counts))
        if active_bin is not None and 0 <= int(active_bin) < len(labels):
            summary = f'{translator.tr("hist.selected")}: {labels[int(active_bin)]} | ' + summary
        self.summary_label.setText(summary)
        self.setToolTip(tooltip)
        self.title_label.setToolTip(tooltip)
        self.summary_label.setToolTip(tooltip)
        self.chart.setToolTip(tooltip)


class _CorrelationColumnWidget(QFrame):
    """Show one colored correlation column as one clickable filter block."""

    columnClicked = pyqtSignal(str)
    columnDoubleClicked = pyqtSignal(str)

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
        self.header_button.installEventFilter(self)
        self.summary_label = QLabel('-', self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.summary_label.setStyleSheet('color: #c9d3df; background: transparent; border: none;')
        self.summary_label.hide()
        layout.addWidget(self.header_button)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self._refresh_style(False)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.header_button and event.type() == QEvent.Type.MouseButtonDblClick:
            if event.button() == Qt.MouseButton.LeftButton:
                self.columnDoubleClicked.emit(self.band)
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.columnClicked.emit(self.band)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.columnDoubleClicked.emit(self.band)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

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
        self.summary_label.clear()
        self.summary_label.hide()


class KarakalWidget(QWidget):

    """Embeddable widget for multi-model segmentation quality evaluation."""

    standaloneAnalysisRepeatRequested = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None, *, settings: QSettings | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(EXTEND_ROOT_OBJECT_NAME)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(EXTEND_WIDGET_STYLESHEET)
        apply_karakal_icon(self)
        self._settings_service = KarakalSettingsService(settings or QSettings(SETTINGS_ORG, SETTINGS_APP))
        self._performance_config = self._settings_service.load_performance_config()
        language = self._settings_service.load_language()
        set_current_language(language)
        self._i18n = Translator(language)
        self._t = self._i18n.tr
        self._update_controller: QtUpdateController | None = None

        self._build_ui()
        self.profiling_dialog = ProfilingDialog(self._performance_config, self)
        self.profiling_dialog.configurationChanged.connect(self._on_performance_config_changed)
        self._setup_menu_bar()

        self._presenter = KarakalPresenter(self, self._settings_service)
        self._connect_signals()
        self._presenter._restore_persisted_state()
        self._presenter._refresh_folder_rows()
        self._presenter._sync_action_buttons()
        if self._update_controller is not None:
            QTimer.singleShot(1500, lambda: self._update_controller.check_for_updates(manual=False))

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._menu_bar = QMenuBar(self)
        root_layout.addWidget(self._menu_bar)

        content = QWidget(self)
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(8, 6, 8, 8)
        root_layout.addWidget(content, stretch=1)

        splitter = QSplitter(Qt.Orientation.Horizontal, content)
        content_layout.addWidget(splitter)

        control_scroll = QScrollArea(splitter)
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        control_scroll.setMinimumWidth(420)
        control_host = QWidget(control_scroll)
        control_layout = QVBoxLayout(control_host)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(10)
        control_scroll.setWidget(control_host)

        style = self.style()

        self.thumbnail_size_spin = _NoWheelSpinBox(self)
        self.thumbnail_size_spin.setRange(DEFAULT_CELL_SIZE, DEFAULT_CELL_SIZE)
        self.thumbnail_size_spin.setValue(DEFAULT_CELL_SIZE)
        self.thumbnail_size_spin.setEnabled(False)

        self.layout_mode_combo = _NoWheelComboBox(self)
        self._populate_layout_mode_combo(DEFAULT_MATRIX_LAYOUT_MODE)
        self.layout_mode_combo.hide()
        self.matrix_score_view_combo = _NoWheelComboBox(self)
        self._populate_matrix_score_view_combo(DEFAULT_MATRIX_SCORE_VIEW_MODE)
        self.matrix_gradient_combo = _NoWheelComboBox(self)
        self._populate_gradient_combo(DEFAULT_GRADIENT_NAME)

        self.total_frames_spin = _NoWheelSpinBox(self)
        self.total_frames_spin.setRange(*TOTAL_FRAMES_RANGE)
        self.total_frames_spin.setValue(DEFAULT_TOTAL_FRAMES)
        self.total_frames_spin.hide()
        self.frames_per_row_spin = _NoWheelSpinBox(self)
        self.frames_per_row_spin.setRange(*FRAMES_PER_ROW_RANGE)
        self.frames_per_row_spin.setValue(DEFAULT_FRAMES_PER_ROW)
        self.matrix_rows_spin = _NoWheelSpinBox(self)
        self.matrix_rows_spin.setRange(*MATRIX_ROWS_RANGE)
        self.matrix_rows_spin.setValue(DEFAULT_MATRIX_ROWS)
        self.matrix_rows_spin.hide()
        self.matrix_columns_spin = _NoWheelSpinBox(self)
        self.matrix_columns_spin.setRange(*MATRIX_COLUMNS_RANGE)
        self.matrix_columns_spin.setValue(DEFAULT_MATRIX_COLUMNS)
        self.matrix_columns_spin.hide()
        self.analysis_mode_combo = _NoWheelComboBox(self)
        self._populate_analysis_mode_combo(DEFAULT_ANALYSIS_MODE)
        self.comparison_target_combo = _NoWheelComboBox(self)
        self._populate_comparison_target_combo(DEFAULT_COMPARISON_TARGET)
        self.geometry_mode_combo = _NoWheelComboBox(self)
        self._populate_geometry_mode_combo(DEFAULT_GEOMETRY_MODE)
        self.mask_threshold_spin = _NoWheelDoubleSpinBox(self)
        self.mask_threshold_spin.setRange(*MASK_THRESHOLD_RANGE)
        self.mask_threshold_spin.setSingleStep(0.05)
        self.mask_threshold_spin.setDecimals(2)
        self.mask_threshold_spin.setValue(DEFAULT_MASK_THRESHOLD)
        self.mask_threshold_spin.setEnabled(False)
        self.mask_threshold_spin.hide()
        self.boundary_radius_spin = _NoWheelSpinBox(self)
        self.boundary_radius_spin.setRange(*BOUNDARY_RADIUS_RANGE)
        self.boundary_radius_spin.setValue(DEFAULT_BOUNDARY_RADIUS)
        self.boundary_radius_spin.setEnabled(False)
        self.boundary_radius_spin.hide()
        self.polygon_compare_profile_combo = _NoWheelComboBox(self)
        self._populate_polygon_compare_profile_combo(DEFAULT_POLYGON_COMPARE_PROFILE)
        self.confidence_uncertainty_profile_combo = _NoWheelComboBox(self)
        self._populate_confidence_uncertainty_profile_combo(DEFAULT_CONFIDENCE_UNCERTAINTY_PROFILE)
        self.polygon_confidence_summary_combo = _NoWheelComboBox(self)
        self._populate_polygon_confidence_summary_combo(DEFAULT_POLYGON_CONFIDENCE_SUMMARY)
        self.point_match_radius_spin = _NoWheelDoubleSpinBox(self)
        self.point_match_radius_spin.setRange(*POINT_MATCH_RADIUS_RANGE)
        self.point_match_radius_spin.setSingleStep(0.5)
        self.point_match_radius_spin.setDecimals(1)
        self.point_match_radius_spin.setValue(DEFAULT_POINT_MATCH_RADIUS)
        self.point_confidence_radius_spin = _NoWheelSpinBox(self)
        self.point_confidence_radius_spin.setRange(*POINT_CONFIDENCE_RADIUS_RANGE)
        self.point_confidence_radius_spin.setValue(DEFAULT_POINT_CONFIDENCE_RADIUS)
        self.point_extraction_mode_combo = _NoWheelComboBox(self)
        self._populate_point_extraction_mode_combo(DEFAULT_POINT_EXTRACTION_MODE)
        self.grid_reference_frame_label = QLabel(self._t("grid_reference.none"), self)
        self.grid_reference_frame_label.setWordWrap(True)
        self.grid_reference_frame_select_button = QPushButton(self._t("grid_reference.select_current"), self)
        self.grid_reference_frame_clear_button = QPushButton(self._t("grid_reference.clear"), self)
        self.grid_error_type_checks: dict[str, QCheckBox] = {}
        for label_key, error_type in GRID_INSPECTION_ERROR_TYPE_OPTIONS:
            checkbox = QCheckBox(self._t(label_key), self)
            checkbox.setIcon(grid_inspection_error_type_icon(error_type))
            checkbox.setChecked(str(error_type) in GRID_INSPECTION_DEFAULT_ERROR_TYPES)
            self.grid_error_type_checks[str(error_type)] = checkbox
        self.metric_group_combo = _NoWheelComboBox(self)
        for label, key in MATRIX_METRIC_GROUP_OPTIONS:
            self.metric_group_combo.addItem(self._t(label), key)
        self.metric_group_combo.hide()
        self.metric_scope_combo = _NoWheelComboBox(self)
        self._populate_metric_scope_combo(None, DEFAULT_METRIC_SCOPE)
        self.metric_combo = _NoWheelComboBox(self)
        self._populate_metric_combo(DEFAULT_MATRIX_METRIC_KEY)
        self.frame_type_filter_combo = _NoWheelComboBox(self)
        self._populate_frame_type_filter_combo('all')
        self.app_mode_combo = _NoWheelComboBox(self)
        self._populate_app_mode_combo("validation")

        self.mode_group = QGroupBox(self._t("app_mode.group"), control_host)
        mode_layout = QVBoxLayout(self.mode_group)
        mode_layout.setContentsMargins(6, 6, 6, 6)
        self._mode_row = self._build_setting_row(self._t("app_mode.current"), self.app_mode_combo)
        mode_layout.addWidget(self._mode_row)
        self.mode_group.setVisible(False)

        self.analysis_setup_panel = AnalysisSetupPanel(self._t, control_host)
        self.analysis_setup_panel.set_profile(DEFAULT_ANALYSIS_PROFILE)
        control_layout.addWidget(self.analysis_setup_panel)

        self.persistent_history_button = QPushButton("История анализов в БД", control_host)
        self.persistent_history_button.setObjectName("persistentAnalysisHistoryButton")
        control_layout.addWidget(self.persistent_history_button)

        self.run_history_group = QGroupBox(self._t("run_history.group"), control_host)
        run_history_layout = QVBoxLayout(self.run_history_group)
        run_history_layout.setContentsMargins(6, 6, 6, 6)
        self.run_history_list = QListWidget(self.run_history_group)
        self.run_history_list.setMaximumHeight(120)
        run_history_layout.addWidget(self.run_history_list)
        self.run_history_group.hide()
        control_layout.addWidget(self.run_history_group)

        folders_group = QGroupBox(self._t("folders.group"), control_host)
        self.folders_group = folders_group
        folders_layout = QVBoxLayout(folders_group)
        folders_info = QLabel(self._t("folders.info"), folders_group)
        self.folders_info_label = folders_info
        folders_info.setWordWrap(True)
        folders_info.show()
        folders_layout.addWidget(folders_info)

        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(4)

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

        self.btn_export_layer = QToolButton(folders_group)
        self.btn_export_layer.setAutoRaise(True)
        self.btn_export_layer.setProperty("toolbarButton", True)
        self.btn_export_layer.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.btn_export_layer.setToolTip(self._t("context.export_result_layer_jpgs_auto"))
        toolbar_layout.addWidget(self.btn_export_layer)

        self.btn_export_grid_checks = QToolButton(folders_group)
        self.btn_export_grid_checks.setAutoRaise(True)
        self.btn_export_grid_checks.setProperty("toolbarButton", True)
        self.btn_export_grid_checks.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DriveFDIcon))
        self.btn_export_grid_checks.setToolTip(self._t("context.export_grid_check_bmps_auto"))
        self.btn_export_grid_checks.setVisible(False)
        toolbar_layout.addWidget(self.btn_export_grid_checks)

        self.btn_cancel = QToolButton(folders_group)
        self.btn_cancel.setAutoRaise(True)
        self.btn_cancel.setProperty("toolbarButton", True)
        self.btn_cancel.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_BrowserStop))
        self.btn_cancel.setToolTip(self._t("folders.cancel"))
        toolbar_layout.addWidget(self.btn_cancel)

        self.frame_search_input = QLineEdit(folders_group)
        self.frame_search_input.setPlaceholderText(self._t("frame_search.placeholder"))
        self.frame_search_input.setToolTip(self._t("frame_search.tooltip"))
        self.frame_search_input.setClearButtonEnabled(True)
        self.frame_search_input.setMaximumWidth(130)
        toolbar_layout.addWidget(self.frame_search_input)

        self.btn_frame_search = QToolButton(folders_group)
        self.btn_frame_search.setAutoRaise(True)
        self.btn_frame_search.setProperty("toolbarButton", True)
        self.btn_frame_search.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView))
        self.btn_frame_search.setToolTip(self._t("frame_search.button"))
        toolbar_layout.addWidget(self.btn_frame_search)
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

        pair_group = QGroupBox("Pair matrix", control_host)
        self.pair_matrix_group = pair_group
        pair_group.setCheckable(True)
        pair_group.setChecked(False)
        pair_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        pair_layout = QVBoxLayout(pair_group)
        pair_layout.setContentsMargins(6, 6, 6, 6)
        pair_layout.setSpacing(0)
        self.pair_matrix_body = QWidget(pair_group)
        pair_body_layout = QVBoxLayout(self.pair_matrix_body)
        pair_body_layout.setContentsMargins(0, 0, 0, 0)
        pair_body_layout.setSpacing(6)
        self.pair_matrix_table = QTableWidget(self.pair_matrix_body)
        self.pair_matrix_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.pair_matrix_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.pair_matrix_table.setAlternatingRowColors(False)
        self.pair_matrix_table.verticalHeader().setVisible(True)
        self.pair_matrix_table.horizontalHeader().setVisible(True)
        self.pair_matrix_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.pair_matrix_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.pair_matrix_table.horizontalHeader().setDefaultSectionSize(92)
        self.pair_matrix_table.verticalHeader().setDefaultSectionSize(28)
        self.pair_matrix_table.verticalHeader().setMinimumSectionSize(28)
        self.pair_matrix_table.horizontalHeader().setMinimumSectionSize(72)
        self.pair_matrix_table.setWordWrap(False)
        self.pair_matrix_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.pair_matrix_table.setMinimumHeight(120)
        pair_body_layout.addWidget(self.pair_matrix_table)
        self.active_pair_list = QListWidget(self.pair_matrix_body)
        self.active_pair_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.active_pair_list.setSpacing(2)
        self.active_pair_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.active_pair_list.setMinimumHeight(72)
        self.active_pair_list.setMaximumHeight(160)
        pair_body_layout.addWidget(self.active_pair_list)
        self.pair_matrix_body.hide()
        pair_layout.addWidget(self.pair_matrix_body)
        control_layout.addWidget(pair_group)

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

        export_row = QWidget(source_group)
        export_row_layout = QHBoxLayout(export_row)
        export_row_layout.setContentsMargins(0, 0, 0, 0)
        export_row_layout.setSpacing(6)
        self.export_folder_value = QLabel(self._t("sources.not_set"), export_row)
        self.export_folder_value.setWordWrap(True)
        self.export_folder_value.setMinimumWidth(0)
        self.export_folder_value.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.btn_set_export = QToolButton(export_row)
        self.btn_set_export.setText(self._t("common.set"))
        self.btn_clear_export = QToolButton(export_row)
        self.btn_clear_export.setText(self._t("common.clear"))
        export_row_layout.addWidget(self.export_folder_value, stretch=1)
        export_row_layout.addWidget(self.btn_set_export)
        export_row_layout.addWidget(self.btn_clear_export)
        self.export_source_label = QLabel(self._t("sources.export"), source_group)
        source_layout.addRow(self.export_source_label, export_row)
        control_layout.addWidget(source_group)

        self.analysis_settings_group = QGroupBox(self._t("ui.analysis_setup"), control_host)
        self.analysis_settings_group.setCheckable(True)
        self.analysis_settings_group.setChecked(False)
        analysis_settings_layout = QVBoxLayout(self.analysis_settings_group)
        analysis_settings_layout.setContentsMargins(6, 6, 6, 6)
        analysis_settings_layout.setSpacing(0)
        self.analysis_settings_body = QWidget(self.analysis_settings_group)
        analysis_settings_body_layout = QVBoxLayout(self.analysis_settings_body)
        analysis_settings_body_layout.setContentsMargins(0, 0, 0, 0)
        analysis_settings_body_layout.setSpacing(0)
        analysis_settings_body_layout.addWidget(self._build_matrix_settings_widget())
        self.analysis_settings_body.hide()
        analysis_settings_layout.addWidget(self.analysis_settings_body)
        control_layout.addWidget(self.analysis_settings_group)

        self.left_mode_stack = QStackedWidget(control_host)
        validation_controls_page = QWidget(self.left_mode_stack)
        validation_controls_layout = QVBoxLayout(validation_controls_page)
        validation_controls_layout.setContentsMargins(0, 0, 0, 0)
        validation_controls_layout.setSpacing(10)

        self.metric_settings_group = QGroupBox(self._t("ui.metric_focus"), validation_controls_page)
        metric_settings_layout = QVBoxLayout(self.metric_settings_group)
        metric_settings_layout.setContentsMargins(4, 4, 4, 4)
        metric_settings_layout.setSpacing(0)
        metric_settings_layout.addWidget(self._build_metric_settings_widget())
        validation_controls_layout.addWidget(self.metric_settings_group)
        validation_controls_layout.addStretch(1)
        self.left_mode_stack.addWidget(validation_controls_page)

        self.left_mode_stack.setCurrentIndex(0)
        control_layout.addWidget(self.left_mode_stack, stretch=1)

        self.main_mode_stack = QStackedWidget(splitter)
        validation_page = QWidget(self.main_mode_stack)
        validation_layout = QVBoxLayout(validation_page)
        validation_layout.setContentsMargins(0, 0, 0, 0)
        validation_layout.setSpacing(6)

        self.validation_matrix_title = QLabel(self._t("validation.matrix.title"), validation_page)
        validation_layout.addWidget(self.validation_matrix_title)

        self.matrix_tabs = QTabWidget(validation_page)
        self.matrix_tabs.setTabsClosable(True)
        self.matrix_tabs.setMovable(True)
        self.matrix_tabs.setDocumentMode(True)
        self.empty_matrix_page = QWidget(self.matrix_tabs)
        empty_layout = QVBoxLayout(self.empty_matrix_page)
        empty_layout.setContentsMargins(48, 48, 48, 48)
        empty_layout.addStretch(1)
        self.empty_matrix_title = QLabel(self._t("empty_matrix.title"), self.empty_matrix_page)
        self.empty_matrix_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_matrix_title.setStyleSheet("font-size: 20px; font-weight: 700; color: #d8e3ef;")
        self.empty_matrix_text = QLabel(self._t("empty_matrix.text"), self.empty_matrix_page)
        self.empty_matrix_text.setWordWrap(True)
        self.empty_matrix_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_matrix_text.setStyleSheet("font-size: 13px; color: #94a6ba;")
        empty_layout.addWidget(self.empty_matrix_title)
        empty_layout.addWidget(self.empty_matrix_text)
        empty_layout.addStretch(1)
        self.show_empty_matrix_state()
        validation_layout.addWidget(self.matrix_tabs)
        self.main_mode_stack.addWidget(validation_page)

        self.grid_inspection_page = self._build_grid_inspection_mode_panel(self.main_mode_stack)
        self.main_mode_stack.addWidget(self.grid_inspection_page)
        self.main_mode_stack.setCurrentIndex(0)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes(list(CONTROL_PANEL_SPLITTER_SIZES))

        self.mode_toggle_button = QToolButton(self._menu_bar)
        self.mode_toggle_button.setAutoRaise(True)
        self.mode_toggle_button.setObjectName("modeToggleButton")
        self.mode_toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mode_toggle_button.setText("⋯")
        self.mode_toggle_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.mode_toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._mode_menu = QMenu(self.mode_toggle_button)
        self.mode_toggle_button.setMenu(self._mode_menu)
        self._mode_menu.triggered.connect(self._on_mode_menu_triggered)

        self.update_tool_button = QToolButton(self._menu_bar)
        self.update_tool_button.setAutoRaise(True)
        self.update_tool_button.setObjectName("updateToolButton")
        self.update_tool_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update_tool_button.setText("Update")
        self.update_tool_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.update_tool_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

        self.language_toggle_button = QToolButton(self._menu_bar)
        self.language_toggle_button.setAutoRaise(True)
        self.language_toggle_button.setObjectName(EXTEND_LANGUAGE_BUTTON_OBJECT_NAME)
        self.language_toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._top_corner_widget = QWidget(self._menu_bar)
        top_corner_layout = QHBoxLayout(self._top_corner_widget)
        top_corner_layout.setContentsMargins(0, 0, 0, 0)
        top_corner_layout.setSpacing(4)
        top_corner_layout.addWidget(self.mode_toggle_button)
        top_corner_layout.addWidget(self.update_tool_button)
        top_corner_layout.addWidget(self.language_toggle_button)
        self._update_language_toggle_button()
        self._rebuild_mode_menu()
        self._update_mode_toggle_button()

    def _build_grid_inspection_mode_panel(self, parent: QWidget) -> QWidget:
        host = QWidget(parent)
        root_layout = QVBoxLayout(host)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(6)

        header = QWidget(host)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(2)
        self.grid_inspection_title = QLabel(self._t("grid_inspection.matrix.title"), header)
        header_layout.addWidget(self.grid_inspection_title)
        self.grid_inspection_legend = MatrixLegendWidget(header)
        header_layout.addWidget(self.grid_inspection_legend)
        root_layout.addWidget(header)

        self.grid_inspection_content_tabs = QTabWidget(host)
        self.grid_inspection_content_tabs.setDocumentMode(True)
        grid_matrix_page = QWidget(self.grid_inspection_content_tabs)
        grid_matrix_layout = QVBoxLayout(grid_matrix_page)
        grid_matrix_layout.setContentsMargins(0, 0, 0, 0)

        row = QWidget(grid_matrix_page)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        self.grid_inspection_layer_tabs = QTabWidget(row)
        self.grid_inspection_layer_tabs.setDocumentMode(True)
        self.grid_inspection_matrix_views: dict[str, MatrixListWidget] = {}
        for layer_key, label_key in (
            ("confidence", "grid_layer.confidence"),
            ("binary", "grid_layer.binary"),
            ("comparison", "grid_layer.comparison"),
        ):
            layer_page = QWidget(self.grid_inspection_layer_tabs)
            layer_layout = QVBoxLayout(layer_page)
            layer_layout.setContentsMargins(0, 0, 0, 0)
            layer_view = MatrixListWidget(layer_page)
            layer_view.set_grid_inspection_visual_mode(True)
            layer_layout.addWidget(layer_view)
            self.grid_inspection_matrix_views[layer_key] = layer_view
            self.grid_inspection_layer_tabs.addTab(layer_page, self._t(label_key))
        self.grid_inspection_matrix_view = self.grid_inspection_matrix_views["confidence"]
        self.grid_inspection_legend.set_scale_info(self.grid_inspection_matrix_view.color_scale_info())
        row_layout.addWidget(self.grid_inspection_layer_tabs, stretch=1)

        overview = QWidget(row)
        overview_layout = QVBoxLayout(overview)
        overview_layout.setContentsMargins(0, 0, 0, 0)
        overview_layout.setSpacing(6)
        self.grid_inspection_matrix_minimap = MatrixMiniMapWidget(overview)
        self.grid_inspection_matrix_minimap.setMinimumHeight(150)
        self.grid_inspection_matrix_minimap.setMaximumHeight(220)
        overview_layout.addWidget(self.grid_inspection_matrix_minimap)

        self.grid_inspection_errors_group = QGroupBox(self._t("grid_errors.group"), overview)
        errors_layout = QVBoxLayout(self.grid_inspection_errors_group)
        errors_layout.setContentsMargins(8, 8, 8, 8)
        errors_layout.setSpacing(6)
        self.grid_inspection_error_filter = QComboBox(self.grid_inspection_errors_group)
        self._populate_grid_inspection_error_filter("all")
        self.grid_inspection_error_counter = QLabel("0 / 0", self.grid_inspection_errors_group)
        self.grid_inspection_error_counter.setWordWrap(True)
        self.grid_inspection_error_list = QListWidget(self.grid_inspection_errors_group)
        self.grid_inspection_error_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.grid_inspection_error_list.setAlternatingRowColors(True)
        self.grid_inspection_error_list.setMinimumHeight(180)
        errors_layout.addWidget(self.grid_inspection_error_filter)
        errors_layout.addWidget(self.grid_inspection_error_counter)
        errors_layout.addWidget(self.grid_inspection_error_list, stretch=1)
        overview_layout.addWidget(self.grid_inspection_errors_group, stretch=1)

        overview.setMinimumWidth(240)
        overview.setMaximumWidth(320)
        overview.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        row_layout.addWidget(overview)
        grid_matrix_layout.addWidget(row, stretch=1)

        (
            grid_percentiles_page,
            self.grid_inspection_histogram_cards,
            self.grid_inspection_repeated_bad_column,
            self.grid_inspection_repeated_good_column,
            self.grid_inspection_percentile_full_matrix_check,
            self.grid_inspection_correlation_limit_spin,
        ) = self._build_histograms_panel(
            self.grid_inspection_content_tabs,
            (GRID_INSPECTION_DAMAGE_METRIC_KEY,),
            None,
        )
        self.grid_inspection_content_tabs.addTab(grid_matrix_page, self._t("tab.matrix"))
        self.grid_inspection_content_tabs.addTab(grid_percentiles_page, self._t("tab.percentiles"))
        root_layout.addWidget(self.grid_inspection_content_tabs, stretch=1)

        for layer_view in self.grid_inspection_matrix_views.values():
            layer_view.colorScaleChanged.connect(
                lambda info, view=layer_view: self.grid_inspection_legend.set_scale_info(info)
                if self.grid_inspection_matrix_view is view
                else None
            )
            layer_view.overviewChanged.connect(
                lambda image, visible_rect, selected_position, selected_blink_on, processing_positions, reference_position, view=layer_view: (
                    self.grid_inspection_matrix_minimap.set_overview(
                        image,
                        visible_rect,
                        selected_position,
                        selected_blink_on,
                        processing_positions,
                        reference_position,
                    )
                    if self.grid_inspection_matrix_view is view
                    else None
                )
            )
        return host

    def _activate_grid_inspection_layer_tab(self, index: int) -> None:
        keys = ("confidence", "binary", "comparison")
        layer_key = keys[max(0, min(int(index), len(keys) - 1))]
        self.grid_inspection_matrix_view = self.grid_inspection_matrix_views[layer_key]
        self.grid_inspection_legend.set_scale_info(self.grid_inspection_matrix_view.color_scale_info())
        self._presenter._on_grid_inspection_layer_changed(layer_key)

    def _set_app_mode(self, mode: str) -> None:
        normalized = str(mode or "validation").strip().lower()
        is_grid_inspection = normalized == "grid_inspection"
        if is_grid_inspection:
            self.main_mode_stack.setCurrentIndex(1)
            self.left_mode_stack.setCurrentIndex(0)
        else:
            self.main_mode_stack.setCurrentIndex(0)
            self.left_mode_stack.setCurrentIndex(0)
        if hasattr(self, "_analysis_task_group"):
            self._analysis_task_group.setVisible(not is_grid_inspection)
        if hasattr(self, "_mode_menu"):
            self._update_mode_toggle_button()

    def _setup_menu_bar(self) -> None:
        self._menu_bar.clear()
        diagnostics_menu = self._menu_bar.addMenu("Diagnostics" if self._i18n.language == "en" else "Диагностика")
        profiling_action = QAction(
            "Validation profiling…" if self._i18n.language == "en" else "Профилирование Validation…",
            diagnostics_menu,
        )
        profiling_action.triggered.connect(self._show_profiling_dialog)
        diagnostics_menu.addAction(profiling_action)
        help_menu = self._menu_bar.addMenu("Help" if self._i18n.language == "en" else "Справка")
        self._update_controller = QtUpdateController(
            self,
            app_id="karakal",
            app_name="Karakal",
            current_version=__version__,
        )
        self._update_controller.add_menu_action(
            help_menu,
            "Check for updates" if self._i18n.language == "en" else "Проверить обновления",
            submenu_title="Update" if self._i18n.language == "en" else "Обновление",
        )
        self._menu_bar.setCornerWidget(self._top_corner_widget, Qt.Corner.TopRightCorner)
        self._setup_update_menu()

    @property
    def performance_config(self) -> PerformanceConfig:
        return self._performance_config

    def _show_profiling_dialog(self) -> None:
        self.profiling_dialog.show()
        self.profiling_dialog.raise_()
        self.profiling_dialog.activateWindow()

    def _on_performance_config_changed(self, config: object) -> None:
        if not isinstance(config, PerformanceConfig):
            return
        self._performance_config = config
        self._settings_service.save_performance_config(config)
        self._settings_service.sync()

    def _setup_update_menu(self) -> None:
        if self._update_controller is None:
            self._update_controller = create_karakal_update_controller(self)
        config = load_karakal_update_client_config()
        selected_channel = load_karakal_update_channel(config)
        self._update_menu = QMenu(self.update_tool_button)
        self._update_channel_menu = QMenu("Channel", self._update_menu)
        self._update_channel_action_group = QActionGroup(self._update_channel_menu)
        self._update_channel_action_group.setExclusive(True)
        channel_labels = {
            "stable": "Stable",
            "beta": "Beta",
        }
        for channel in config.available_channels:
            normalized_channel = str(channel or "").strip().lower()
            action = self._update_channel_menu.addAction(
                channel_labels.get(normalized_channel, normalized_channel or "stable")
            )
            action.setCheckable(True)
            action.setData(normalized_channel)
            action.setChecked(normalized_channel == selected_channel)
            self._update_channel_action_group.addAction(action)
        self._update_channel_action_group.triggered.connect(self._on_update_channel_triggered)
        self._update_menu.addMenu(self._update_channel_menu)
        self._check_updates_action = self._update_menu.addAction("Check for updates")
        self._check_updates_action.triggered.connect(
            lambda _checked=False: self._update_controller.check_for_updates(manual=True)
        )
        self.update_tool_button.setMenu(self._update_menu)

    def _on_update_channel_triggered(self, action) -> None:
        channel = str(action.data() or "").strip().lower()
        if channel:
            save_karakal_update_channel(channel)

    def _rebuild_mode_menu(self) -> None:
        self._mode_menu.clear()
        self._mode_action_group = QActionGroup(self._mode_menu)
        self._mode_action_group.setExclusive(True)
        for label_key, mode_key in (
            ("app_mode.validation", "validation"),
            ("grid_inspection.mode", "grid_inspection"),
        ):
            action = self._mode_menu.addAction(self._t(label_key))
            action.setData(mode_key)
            action.setCheckable(True)
            self._mode_action_group.addAction(action)
            action.setChecked(str(self.app_mode_combo.currentData() or "validation") == mode_key)

    def _on_mode_menu_triggered(self, action) -> None:
        mode = str(action.data() or "validation")
        index = self.app_mode_combo.findData(mode)
        if index >= 0:
            self.app_mode_combo.setCurrentIndex(index)

    def _update_mode_toggle_button(self) -> None:
        current_mode = str(self.app_mode_combo.currentData() or "validation")
        current_label = {
            "validation": self._t("app_mode.validation"),
            "grid_inspection": self._t("grid_inspection.mode"),
        }.get(current_mode, self._t("app_mode.validation"))
        self.mode_toggle_button.setToolTip(f"{self._t('app_mode.group')}: {current_label}")
        for action in self._mode_menu.actions():
            action.setChecked(str(action.data() or "") == current_mode)

    def _populate_app_mode_combo(self, selected_mode: str | None) -> None:
        current = str(selected_mode or "validation")
        self.app_mode_combo.blockSignals(True)
        self.app_mode_combo.clear()
        self.app_mode_combo.addItem(self._t("app_mode.validation"), "validation")
        self.app_mode_combo.addItem(self._t("grid_inspection.mode"), "grid_inspection")
        index = self.app_mode_combo.findData(current)
        self.app_mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.app_mode_combo.blockSignals(False)
        if hasattr(self, "_mode_menu"):
            self._rebuild_mode_menu()
            self._update_mode_toggle_button()

    def _populate_layout_mode_combo(self, selected_mode: str | None) -> None:
        current = str(selected_mode or DEFAULT_MATRIX_LAYOUT_MODE)
        self.layout_mode_combo.blockSignals(True)
        self.layout_mode_combo.clear()
        self.layout_mode_combo.addItem(self._t("matrix.layout.indexed"), "indexed_grid")
        index = self.layout_mode_combo.findData(current)
        self.layout_mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.layout_mode_combo.blockSignals(False)

    def _populate_matrix_score_view_combo(self, selected_mode: str | None) -> None:
        current = str(selected_mode or DEFAULT_MATRIX_SCORE_VIEW_MODE)
        self.matrix_score_view_combo.blockSignals(True)
        self.matrix_score_view_combo.clear()
        for label_key, key in MATRIX_SCORE_VIEW_OPTIONS:
            self.matrix_score_view_combo.addItem(self._t(label_key), key)
        index = self.matrix_score_view_combo.findData(current)
        self.matrix_score_view_combo.setCurrentIndex(index if index >= 0 else 0)
        self.matrix_score_view_combo.blockSignals(False)

    def _populate_gradient_combo(self, selected_name: str | None) -> None:
        current = str(selected_name or DEFAULT_GRADIENT_NAME)
        self.matrix_gradient_combo.blockSignals(True)
        self.matrix_gradient_combo.clear()
        for name, label_key in GRADIENT_LABELS.items():
            self.matrix_gradient_combo.addItem(self._t(label_key), name)
        index = self.matrix_gradient_combo.findData(current)
        self.matrix_gradient_combo.setCurrentIndex(index if index >= 0 else 0)
        self.matrix_gradient_combo.blockSignals(False)

    def _populate_analysis_mode_combo(self, selected_mode: str | None) -> None:
        current = str(selected_mode or DEFAULT_ANALYSIS_MODE)
        self.analysis_mode_combo.blockSignals(True)
        self.analysis_mode_combo.clear()
        for label_key, key in ANALYSIS_MODE_OPTIONS:
            self.analysis_mode_combo.addItem(self._t(label_key), key)
        index = self.analysis_mode_combo.findData(current)
        self.analysis_mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.analysis_mode_combo.blockSignals(False)

    def _populate_comparison_target_combo(self, selected_target: str | None) -> None:
        current = str(selected_target or DEFAULT_COMPARISON_TARGET)
        self.comparison_target_combo.blockSignals(True)
        self.comparison_target_combo.clear()
        for label_key, key in COMPARISON_TARGET_OPTIONS:
            self.comparison_target_combo.addItem(self._t(label_key), key)
        index = self.comparison_target_combo.findData(current)
        self.comparison_target_combo.setCurrentIndex(index if index >= 0 else 0)
        self.comparison_target_combo.blockSignals(False)

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

    def _populate_grid_inspection_error_filter(self, selected_error_type: str | None = None) -> None:
        combo = getattr(self, "grid_inspection_error_filter", None)
        if combo is None:
            return
        current = str(selected_error_type or combo.currentData() or "all")
        blocker = QSignalBlocker(combo)
        combo.clear()
        combo.addItem(self._t("grid_error.all"), "all")
        for label_key, error_type in GRID_INSPECTION_ERROR_TYPE_OPTIONS:
            combo.addItem(grid_inspection_error_type_icon(error_type), self._t(label_key), error_type)
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else 0)
        del blocker

    def _populate_polygon_compare_profile_combo(self, selected_value: str | None) -> None:
        current = str(selected_value or DEFAULT_POLYGON_COMPARE_PROFILE)
        self.polygon_compare_profile_combo.blockSignals(True)
        self.polygon_compare_profile_combo.clear()
        for label_key, key in POLYGON_COMPARE_PROFILE_OPTIONS:
            self.polygon_compare_profile_combo.addItem(self._t(label_key), key)
        index = self.polygon_compare_profile_combo.findData(current)
        if index < 0:
            index = self.polygon_compare_profile_combo.findData(DEFAULT_POLYGON_COMPARE_PROFILE)
        self.polygon_compare_profile_combo.setCurrentIndex(index if index >= 0 else 0)
        self.polygon_compare_profile_combo.blockSignals(False)

    def _populate_confidence_uncertainty_profile_combo(self, selected_value: str | None) -> None:
        current = str(selected_value or DEFAULT_CONFIDENCE_UNCERTAINTY_PROFILE)
        self.confidence_uncertainty_profile_combo.blockSignals(True)
        self.confidence_uncertainty_profile_combo.clear()
        for label_key, key in CONFIDENCE_UNCERTAINTY_PROFILE_OPTIONS:
            self.confidence_uncertainty_profile_combo.addItem(self._t(label_key), key)
        index = self.confidence_uncertainty_profile_combo.findData(current)
        self.confidence_uncertainty_profile_combo.setCurrentIndex(index if index >= 0 else 0)
        self.confidence_uncertainty_profile_combo.blockSignals(False)

    def _metric_scope_label(self, spec, *, output_only: bool = False) -> str:
        base_name = str(getattr(spec, "display_name", "") or getattr(getattr(spec, "mask_folder", None), "name", "") or getattr(spec, "model_id", ""))
        confidence_folder = getattr(spec, "prob_folder", None)
        confidence_name = str(getattr(confidence_folder, "name", "") or "")
        if output_only:
            if confidence_name and confidence_name != base_name:
                return f"{confidence_name} ({base_name})"
            return confidence_name or base_name
        if confidence_name and confidence_name != base_name:
            return f"{base_name} ({confidence_name})"
        return base_name

    def _populate_metric_scope_combo(self, build_result: BuildResult | None, selected_scope: str | None, *, output_only: bool = False) -> None:
        current = str(selected_scope or "")
        self.metric_scope_combo.blockSignals(True)
        self.metric_scope_combo.clear()
        if build_result is not None:
            for spec in build_result.model_specs:
                self.metric_scope_combo.addItem(self._metric_scope_label(spec, output_only=output_only), str(spec.model_id))
        if build_result is None and self.metric_scope_combo.count() <= 0 and current:
            self.metric_scope_combo.addItem(current, current)
        index = self.metric_scope_combo.findData(current)
        if index < 0 and build_result is not None:
            fallback = default_confidence_model_id(build_result, output_only=output_only)
            index = self.metric_scope_combo.findData(fallback)
        self.metric_scope_combo.setCurrentIndex(index if index >= 0 else 0)
        self.metric_scope_combo.setToolTip(self._t('analysis.confidence_model'))
        self.metric_scope_combo.blockSignals(False)

    def _metric_text_for_key(self, metric_key: str, build_result: BuildResult | None = None) -> str:
        metric_key_text = str(metric_key)
        if metric_key_text == GRID_INSPECTION_DAMAGE_METRIC_KEY:
            return self._t("metric.grid_inspection_damage_score")
        for label_key, key, _group in MATRIX_METRIC_OPTIONS:
            if str(key) == metric_key_text:
                return self._t(label_key)
        translated = self._t(f"metric.{metric_key_text}")
        if translated != f"metric.{metric_key_text}":
            return translated
        pair_parts = metric_key_text.split("::")
        if len(pair_parts) == 4 and pair_parts[0] == "pair":
            model_a, model_b, operation = pair_parts[1], pair_parts[2], pair_parts[3]
            names = {
                str(spec.model_id): str(spec.display_name or spec.model_id)
                for spec in tuple(getattr(build_result, "model_specs", ()) or ())
            }
            operation_label = {"xor": "XOR", "iou": "IoU", "dice": "Dice"}.get(operation, operation)
            return f"{names.get(model_a, model_a)} -> {names.get(model_b, model_b)} [{operation_label}]"
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
            if family == 'model_output_confidence':
                return f"{self._t('metric.model_output_confidence')} [{model_name}]"
            if family == 'model_uncertain_fraction':
                return f"{self._t('metric.model_uncertain_fraction')} [{model_name}]"
            if family == 'model_point_contrast':
                return f"{self._t('metric.model_point_contrast')} [{model_name}]"
        return metric_key_text

    @staticmethod
    def _is_overall_score_metric(metric_key: str) -> bool:
        return str(metric_key).startswith("overall_")

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
        self._matrix_comparison_target_row = self._build_setting_row(self._t("comparison_target.label"), self.comparison_target_combo)
        self._matrix_geometry_row = self._build_setting_row(self._t("analysis.object_type"), self.geometry_mode_combo)
        self._matrix_polygon_compare_profile_row = self._build_setting_row(self._t("matrix.polygon_compare_profile"), self.polygon_compare_profile_combo)
        self._matrix_confidence_delta_row = self._build_setting_row(self._t('matrix.confidence_delta'), self.confidence_uncertainty_profile_combo)
        self._matrix_polygon_confidence_summary_row = self._build_setting_row(self._t('matrix.polygon_confidence_summary'), self.polygon_confidence_summary_combo)
        self._matrix_point_radius_row = self._build_setting_row(self._t('matrix.point_match_radius'), self.point_match_radius_spin)
        self._matrix_point_confidence_radius_row = self._build_setting_row(self._t('matrix.point_confidence_radius'), self.point_confidence_radius_spin)
        self._matrix_point_mode_row = self._build_setting_row(self._t('matrix.point_extraction_mode'), self.point_extraction_mode_combo)
        self._matrix_layout_row = None
        self._matrix_score_view_row = self._build_setting_row(self._t("matrix.score_view"), self.matrix_score_view_combo)
        self._matrix_gradient_row = self._build_setting_row(self._t("matrix.gradient"), self.matrix_gradient_combo)
        self._matrix_frame_type_filter_row = self._build_setting_row(self._t('matrix.frame_type_filter'), self.frame_type_filter_combo)
        self._matrix_total_frames_row = None
        self._matrix_frames_per_row_row = self._build_setting_row(self._t("matrix.frames_per_row"), self.frames_per_row_spin)
        self._matrix_rows_row = None
        self._matrix_columns_row = None
        grid_reference_control = QWidget(widget)
        grid_reference_layout = QVBoxLayout(grid_reference_control)
        grid_reference_layout.setContentsMargins(0, 0, 0, 0)
        grid_reference_layout.setSpacing(4)
        grid_reference_layout.addWidget(self.grid_reference_frame_label)
        grid_reference_buttons = QWidget(grid_reference_control)
        grid_reference_buttons_layout = QHBoxLayout(grid_reference_buttons)
        grid_reference_buttons_layout.setContentsMargins(0, 0, 0, 0)
        grid_reference_buttons_layout.setSpacing(6)
        grid_reference_buttons_layout.addWidget(self.grid_reference_frame_select_button)
        grid_reference_buttons_layout.addWidget(self.grid_reference_frame_clear_button)
        grid_reference_layout.addWidget(grid_reference_buttons)
        self._grid_reference_frame_row = self._build_setting_row(self._t("grid_reference.label"), grid_reference_control)
        self._analysis_task_group = QGroupBox(self._t("ui.analysis_task"), widget)
        analysis_layout = QVBoxLayout(self._analysis_task_group)
        analysis_layout.setContentsMargins(8, 8, 8, 8)
        analysis_layout.setSpacing(8)
        for row in (
            self._matrix_pixel_size_row,
            self._matrix_analysis_mode_row,
            self._matrix_comparison_target_row,
            self._matrix_geometry_row,
            self._matrix_polygon_compare_profile_row,
            self._matrix_confidence_delta_row,
            self._matrix_polygon_confidence_summary_row,
            self._matrix_point_radius_row,
            self._matrix_point_confidence_radius_row,
            self._matrix_point_mode_row,
        ):
            analysis_layout.addWidget(row)

        self._matrix_view_group = QGroupBox(self._t("ui.matrix_view"), widget)
        matrix_layout = QVBoxLayout(self._matrix_view_group)
        matrix_layout.setContentsMargins(8, 8, 8, 8)
        matrix_layout.setSpacing(8)
        for row in (
            self._matrix_score_view_row,
            self._matrix_gradient_row,
            self._matrix_frame_type_filter_row,
            self._matrix_frames_per_row_row,
        ):
            if row is not None:
                matrix_layout.addWidget(row)
        self._grid_inspection_tuning_group = QGroupBox(self._t("grid_tuning.group"), widget)
        grid_tuning_layout = QVBoxLayout(self._grid_inspection_tuning_group)
        grid_tuning_layout.setContentsMargins(8, 8, 8, 8)
        grid_tuning_layout.setSpacing(8)
        grid_tuning_layout.addWidget(self._grid_reference_frame_row)
        self._grid_error_type_checks_title = QLabel(self._t("grid_tuning.enabled_errors"), self._grid_inspection_tuning_group)
        self._grid_error_type_checks_title.setWordWrap(True)
        grid_tuning_layout.addWidget(self._grid_error_type_checks_title)
        for _label_key, error_type in GRID_INSPECTION_ERROR_TYPE_OPTIONS:
            checkbox = self.grid_error_type_checks.get(str(error_type))
            if checkbox is not None:
                checkbox.setParent(self._grid_inspection_tuning_group)
                grid_tuning_layout.addWidget(checkbox)
        layout.addWidget(self._matrix_view_group)
        layout.addWidget(self._grid_inspection_tuning_group)
        layout.addWidget(self._analysis_task_group)
        self._matrix_pixel_size_row.setVisible(False)
        self._matrix_frames_per_row_row.setVisible(True)
        layout.addStretch(1)
        return widget

    def _build_metric_settings_widget(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)
        for combo in (self.metric_scope_combo, self.metric_combo):
            combo.setMinimumContentsLength(max(12, min(14, METRIC_SETTINGS_COMBO_MIN_CONTENTS_LENGTH + 2)))
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._metric_scope_row = self._build_setting_row(
            self._t("analysis.confidence_model"),
            self.metric_scope_combo,
            label_min_width=METRIC_SETTINGS_LABEL_MIN_WIDTH,
            compact=True,
        )
        layout.addWidget(self._metric_scope_row)
        self._metric_select_row = self._build_setting_row(
            self._t("menu.metric.select"),
            self.metric_combo,
            label_min_width=METRIC_SETTINGS_LABEL_MIN_WIDTH,
            compact=True,
        )
        layout.addWidget(self._metric_select_row)
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
        self._update_mode_toggle_button()
        self.mode_group.setTitle(self._t("app_mode.group"))
        mode_row_label = getattr(getattr(self, "_mode_row", None), "_title_label", None)
        if mode_row_label is not None:
            mode_row_label.setText(self._t("app_mode.current"))
        self._populate_app_mode_combo(self.app_mode_combo.currentData())
        self.analysis_settings_group.setTitle(self._t("ui.analysis_setup"))
        self.metric_settings_group.setTitle(self._t("ui.metric_focus"))
        self._analysis_task_group.setTitle(self._t("ui.analysis_task"))
        self._matrix_view_group.setTitle(self._t("ui.matrix_view"))
        if hasattr(self, "_grid_inspection_tuning_group"):
            self._grid_inspection_tuning_group.setTitle(self._t("grid_tuning.group"))
            if hasattr(self, "_grid_error_type_checks_title"):
                self._grid_error_type_checks_title.setText(self._t("grid_tuning.enabled_errors"))
            for label_key, error_type in GRID_INSPECTION_ERROR_TYPE_OPTIONS:
                checkbox = getattr(self, "grid_error_type_checks", {}).get(str(error_type))
                if checkbox is not None:
                    checkbox.setText(self._t(label_key))
            grid_tuning_labels = {
                "_grid_reference_frame_row": "grid_reference.label",
            }
            for row_name, label_key in grid_tuning_labels.items():
                label = getattr(getattr(self, row_name, None), "_title_label", None)
                if label is not None:
                    label.setText(self._t(label_key))
            if hasattr(self, "grid_reference_frame_select_button"):
                self.grid_reference_frame_select_button.setText(self._t("grid_reference.select_current"))
            if hasattr(self, "grid_reference_frame_clear_button"):
                self.grid_reference_frame_clear_button.setText(self._t("grid_reference.clear"))
        self._populate_matrix_score_view_combo(self.matrix_score_view_combo.currentData() or DEFAULT_MATRIX_SCORE_VIEW_MODE)
        self._populate_gradient_combo(self.matrix_gradient_combo.currentData() or DEFAULT_GRADIENT_NAME)
        self.analysis_setup_panel.retranslate(self._t)
        self.run_history_group.setTitle(self._t("run_history.group"))
        if hasattr(self, "grid_inspection_legend"):
            self.grid_inspection_legend.retranslate()
        if hasattr(self, "_presenter"):
            for state in self._presenter._tab_states.values():
                if state.legend is not None:
                    state.legend.retranslate()
        self._populate_confidence_uncertainty_profile_combo(self.confidence_uncertainty_profile_combo.currentData() or DEFAULT_CONFIDENCE_UNCERTAINTY_PROFILE)
        self.folders_group.setTitle(self._t("folders.group"))
        if hasattr(self, "pair_matrix_group"):
            title = (
                self._presenter._pair_matrix_title()
                if hasattr(self, "_presenter")
                else self._t("pairs.group")
            )
            self.pair_matrix_group.setTitle(title)
        self.folders_info_label.setText(self._t("folders.info"))
        self.btn_add_folder.setToolTip(self._t("folders.add_model"))
        self.btn_clear_folders.setToolTip(self._t("folders.clear_models"))
        self.btn_build.setToolTip(self._t("folders.build"))
        self.btn_compute.setToolTip(self._t("folders.compute_mismatch"))
        self.btn_export_layer.setToolTip(self._t("context.export_result_layer_jpgs_auto"))
        self.btn_export_grid_checks.setToolTip(self._t("context.export_grid_check_bmps_auto"))
        self.btn_cancel.setToolTip(self._t("folders.cancel"))
        if hasattr(self, "frame_search_input"):
            self.frame_search_input.setPlaceholderText(self._t("frame_search.placeholder"))
            self.frame_search_input.setToolTip(self._t("frame_search.tooltip"))
        if hasattr(self, "btn_frame_search"):
            self.btn_frame_search.setToolTip(self._t("frame_search.button"))
        self.source_group.setTitle(self._t("sources.group"))
        self.original_source_label.setText(self._t("sources.original"))
        self.export_source_label.setText(self._t("sources.export"))
        self.btn_set_original.setText(self._t("common.set"))
        self.btn_clear_original.setText(self._t("common.clear"))
        self.btn_set_export.setText(self._t("common.set"))
        self.btn_clear_export.setText(self._t("common.clear"))
        if hasattr(self, "validation_matrix_title"):
            self.validation_matrix_title.setText(self._t("validation.matrix.title"))
            self.empty_matrix_title.setText(self._t("empty_matrix.title"))
            self.empty_matrix_text.setText(self._t("empty_matrix.text"))
        if hasattr(self, "grid_inspection_title"):
            self.grid_inspection_title.setText(self._t("grid_inspection.matrix.title"))
        if hasattr(self, "grid_inspection_errors_group"):
            self.grid_inspection_errors_group.setTitle(self._t("grid_errors.group"))
            self._populate_grid_inspection_error_filter(self.grid_inspection_error_filter.currentData())
        if hasattr(self, "_presenter"):
            self._presenter._update_source_labels()
        current_layout = str(self.layout_mode_combo.currentData() or DEFAULT_MATRIX_LAYOUT_MODE)
        current_analysis_mode = str(self.analysis_mode_combo.currentData() or DEFAULT_ANALYSIS_MODE)
        current_comparison_target = str(self.comparison_target_combo.currentData() or DEFAULT_COMPARISON_TARGET)
        current_geometry = str(self.geometry_mode_combo.currentData() or DEFAULT_GEOMETRY_MODE)
        current_metric_scope = str(self.metric_scope_combo.currentData() or DEFAULT_METRIC_SCOPE)
        current_metric = str(self.metric_combo.currentData() or DEFAULT_MATRIX_METRIC_KEY)
        current_frame_type_filter = str(self.frame_type_filter_combo.currentData() or 'all')
        current_point_extraction_mode = str(self.point_extraction_mode_combo.currentData() or DEFAULT_POINT_EXTRACTION_MODE)
        current_polygon_confidence_summary = str(self.polygon_confidence_summary_combo.currentData() or DEFAULT_POLYGON_CONFIDENCE_SUMMARY)
        self._populate_layout_mode_combo(current_layout)
        self._populate_analysis_mode_combo(current_analysis_mode)
        self._populate_comparison_target_combo(current_comparison_target)
        self._populate_geometry_mode_combo(current_geometry)
        self._populate_polygon_compare_profile_combo(self.polygon_compare_profile_combo.currentData() or DEFAULT_POLYGON_COMPARE_PROFILE)
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
            (getattr(self, "_matrix_comparison_target_row", None), "comparison_target.label"),
            (getattr(self, "_matrix_geometry_row", None), "analysis.object_type"),
            (getattr(self, "_matrix_polygon_compare_profile_row", None), "matrix.polygon_compare_profile"),
            (getattr(self, "_matrix_confidence_delta_row", None), "matrix.confidence_delta"),
            (getattr(self, "_matrix_polygon_confidence_summary_row", None), "matrix.polygon_confidence_summary"),
            (getattr(self, "_matrix_point_radius_row", None), "matrix.point_match_radius"),
            (getattr(self, "_matrix_point_confidence_radius_row", None), "matrix.point_confidence_radius"),
            (getattr(self, "_matrix_point_mode_row", None), "matrix.point_extraction_mode"),
            (getattr(self, "_matrix_layout_row", None), "matrix.layout"),
            (getattr(self, "_matrix_score_view_row", None), "matrix.score_view"),
            (getattr(self, "_matrix_gradient_row", None), "matrix.gradient"),
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
        if hasattr(self, "_presenter"):
            current_state = self._presenter._current_tab_state()
            self._presenter._sync_mode_controls(current_state, None if current_state is None else current_state.build_result)
        for state in getattr(self, "_presenter", None)._tab_states.values() if hasattr(getattr(self, "_presenter", None), "_tab_states") else ():
            state.preview.group.setTitle(self._t("matrix.preview.group"))
            state.preview.frame_title.setText(self._t("matrix.preview.frame"))
            if state.preview.overall_group is not None:
                state.preview.overall_group.setTitle(self._t("matrix.preview.score_overall"))
            if state.preview.component_group is not None:
                state.preview.component_group.setTitle(self._t("matrix.preview.score_components"))
            for metric_key, card in state.preview.score_cards.items():
                if hasattr(card, "title_label"):
                    card.title_label.setText(self._metric_text_for_key(metric_key, state.build_result))
            for metric_key, card in state.preview.histogram_cards.items():
                if hasattr(card, "title_label"):
                    card.title_label.setText(self._metric_text_for_key(metric_key, state.build_result))
            if state.percentile_full_matrix_check is not None:
                state.percentile_full_matrix_check.setText(self._t("hist.full_matrix_toggle"))
                state.percentile_full_matrix_check.setToolTip(self._t("hist.full_matrix_toggle_hint"))
            if state.correlation_limit_spin is not None:
                label = getattr(state.correlation_limit_spin, "_label", None)
                if label is not None:
                    label.setText(self._t("hist.correlation_limit"))
                state.correlation_limit_spin.setToolTip(self._t("hist.correlation_limit_hint", count=state.correlation_limit_spin.maximum()))
            if state.content_tabs is not None:
                state.content_tabs.setTabText(0, self._t("tab.matrix"))
                state.content_tabs.setTabText(1, self._t("tab.percentiles"))
            if state.repeated_bad_column is not None:
                state.repeated_bad_column.header_button.setText(self._t("correlation.bad"))
            if state.repeated_good_column is not None:
                state.repeated_good_column.header_button.setText(self._t("correlation.good"))
        if hasattr(self, "grid_inspection_content_tabs"):
            self.grid_inspection_content_tabs.setTabText(0, self._t("tab.matrix"))
            self.grid_inspection_content_tabs.setTabText(1, self._t("tab.percentiles"))
        if hasattr(self, "grid_inspection_layer_tabs"):
            self.grid_inspection_layer_tabs.setTabText(0, self._t("grid_layer.confidence"))
            self.grid_inspection_layer_tabs.setTabText(1, self._t("grid_layer.binary"))
            self.grid_inspection_layer_tabs.setTabText(2, self._t("grid_layer.comparison"))
        for metric_key, card in getattr(self, "grid_inspection_histogram_cards", {}).items():
            if hasattr(card, "title_label"):
                card.title_label.setText(self._metric_text_for_key(metric_key, None))
        if hasattr(self, "grid_inspection_percentile_full_matrix_check"):
            self.grid_inspection_percentile_full_matrix_check.setText(self._t("hist.full_matrix_toggle"))
            self.grid_inspection_percentile_full_matrix_check.setToolTip(self._t("hist.full_matrix_toggle_hint"))
        if hasattr(self, "grid_inspection_correlation_limit_spin"):
            label = getattr(self.grid_inspection_correlation_limit_spin, "_label", None)
            if label is not None:
                label.setText(self._t("hist.correlation_limit"))
            self.grid_inspection_correlation_limit_spin.setToolTip(
                self._t("hist.correlation_limit_hint", count=self.grid_inspection_correlation_limit_spin.maximum())
            )
        if hasattr(self, "grid_inspection_repeated_bad_column"):
            self.grid_inspection_repeated_bad_column.header_button.setText(self._t("correlation.bad"))
        if hasattr(self, "grid_inspection_repeated_good_column"):
            self.grid_inspection_repeated_good_column.header_button.setText(self._t("correlation.good"))
        window = self.window()
        if isinstance(window, QMainWindow):
            window.setWindowTitle(self._t("window.title"))

    def _toggle_language(self) -> None:
        current_language = str(self._i18n.language or "en").lower()
        language = "ru" if current_language == "en" else "en"
        self._i18n.set_language(language)
        self._settings_service.save_language(language)
        self.retranslate_ui()

    def _build_setting_row(
        self,
        title: str,
        control: QWidget,
        *,
        label_min_width: int = SETTINGS_LABEL_MIN_WIDTH,
        compact: bool = False,
    ) -> QWidget:
        row = QWidget(self)
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(2 if compact else 4)
        label = QLabel(title, row)
        label.setMinimumWidth(0)
        label.setMaximumWidth(int(label_min_width * 3))
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        control.setMinimumWidth(0)
        control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(label)
        layout.addWidget(control)
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
        frame_value.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        frame_layout.addWidget(frame_title)
        frame_layout.addWidget(frame_value, stretch=1)
        layout.addWidget(frame_row)

        score_cards: dict[str, QWidget] = {}
        seen_keys: set[str] = set()
        overall_card_count = 0
        component_card_count = 0
        overall_group = QGroupBox(self._t("matrix.preview.score_overall"), group)
        overall_layout = QVBoxLayout(overall_group)
        overall_layout.setContentsMargins(0, 0, 0, 0)
        overall_layout.setSpacing(8)
        overall_host = QWidget(overall_group)
        overall_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        overall_host_layout = QVBoxLayout(overall_host)
        overall_host_layout.setContentsMargins(0, 0, 0, 0)
        overall_host_layout.setSpacing(8)
        overall_layout.addWidget(overall_host)

        component_group = QGroupBox(self._t("matrix.preview.score_components"), group)
        component_layout = QVBoxLayout(component_group)
        component_layout.setContentsMargins(0, 0, 0, 0)
        component_layout.setSpacing(8)
        component_host = QWidget(component_group)
        component_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        component_host_layout = QVBoxLayout(component_host)
        component_host_layout.setContentsMargins(0, 0, 0, 0)
        component_host_layout.setSpacing(8)
        component_layout.addWidget(component_host)

        for metric_key in metric_keys:
            if metric_key in seen_keys:
                continue
            seen_keys.add(metric_key)
            target_parent = overall_host if self._is_overall_score_metric(metric_key) else component_host
            card = _ExpandableScoreCard(self._metric_text_for_key(metric_key, build_result), target_parent)
            card.hide()
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            target_layout = overall_host_layout if self._is_overall_score_metric(metric_key) else component_host_layout
            target_layout.addWidget(card)
            score_cards[str(metric_key)] = card
            if self._is_overall_score_metric(metric_key):
                overall_card_count += 1
            else:
                component_card_count += 1
        overall_host_layout.addStretch(1)
        component_host_layout.addStretch(1)
        overall_group.setVisible(overall_card_count > 0)
        component_group.setVisible(component_card_count > 0)
        layout.addWidget(overall_group)
        layout.addWidget(component_group)
        layout.addStretch(1)
        return ExtendPreviewPanel(
            group=group,
            frame_title=frame_title,
            frame_value=frame_value,
            overall_group=overall_group,
            component_group=component_group,
            score_cards=score_cards,
            histogram_cards={},
        )

    def _build_histograms_panel(self, parent: QWidget, metric_keys: tuple[str, ...], build_result: BuildResult | None) -> tuple[QWidget, dict[str, QWidget], QWidget, QWidget, QCheckBox, QSpinBox]:
        panel = QWidget(parent)
        panel_layout = QHBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(10)

        charts_column = QWidget(panel)
        charts_column_layout = QVBoxLayout(charts_column)
        charts_column_layout.setContentsMargins(0, 0, 0, 0)
        charts_column_layout.setSpacing(6)
        full_matrix_check = QCheckBox(self._t("hist.full_matrix_toggle"), charts_column)
        full_matrix_check.setChecked(True)
        full_matrix_check.setToolTip(self._t("hist.full_matrix_toggle_hint"))
        charts_column_layout.addWidget(full_matrix_check)

        charts_scroll = QScrollArea(charts_column)
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
        charts_column_layout.addWidget(charts_scroll, stretch=1)
        panel_layout.addWidget(charts_column, stretch=3)

        columns_host = QWidget(panel)
        columns_host.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        columns_host_layout = QVBoxLayout(columns_host)
        columns_host_layout.setContentsMargins(0, 0, 0, 0)
        columns_host_layout.setSpacing(8)
        columns_host_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        limit_row = QWidget(columns_host)
        limit_layout = QHBoxLayout(limit_row)
        limit_layout.setContentsMargins(0, 0, 0, 0)
        limit_layout.setSpacing(6)
        limit_label = QLabel(self._t("hist.correlation_limit"), limit_row)
        correlation_limit_spin = _NoWheelSpinBox(limit_row)
        correlation_limit_spin.setRange(1, 1000)
        correlation_limit_spin.setValue(25)
        correlation_limit_spin.set_clamp_to_max_on_edit(True)
        correlation_limit_spin.setMaximumWidth(96)
        correlation_limit_spin.setToolTip(self._t("hist.correlation_limit_hint", count=correlation_limit_spin.maximum()))
        correlation_limit_spin._label = limit_label
        limit_layout.addWidget(limit_label)
        limit_layout.addWidget(correlation_limit_spin)
        limit_layout.addStretch(1)
        columns_host_layout.addWidget(limit_row, 0, Qt.AlignmentFlag.AlignTop)
        columns_layout = QHBoxLayout()
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.setSpacing(8)
        repeated_bad_column = _CorrelationColumnWidget(self._t("correlation.bad"), 'bad', columns_host)
        repeated_good_column = _CorrelationColumnWidget(self._t("correlation.good"), 'good', columns_host)
        columns_layout.addWidget(repeated_bad_column, 0, Qt.AlignmentFlag.AlignTop)
        columns_layout.addWidget(repeated_good_column, 0, Qt.AlignmentFlag.AlignTop)
        columns_layout.addStretch(1)
        columns_host_layout.addLayout(columns_layout)
        panel_layout.addWidget(columns_host, stretch=1, alignment=Qt.AlignmentFlag.AlignTop)

        return panel, histogram_cards, repeated_bad_column, repeated_good_column, full_matrix_check, correlation_limit_spin

    def _add_menu_widget(self, menu: QMenu, widget: QWidget) -> None:
        action = QWidgetAction(menu)
        action.setDefaultWidget(widget)
        menu.addAction(action)

    def _connect_signals(self) -> None:
        self.pair_matrix_group.toggled.connect(self.pair_matrix_body.setVisible)
        self.pair_matrix_group.toggled.connect(self._presenter._persist_state)
        self.analysis_settings_group.toggled.connect(self.analysis_settings_body.setVisible)
        self.analysis_settings_group.toggled.connect(self._presenter._persist_state)
        self.app_mode_combo.currentIndexChanged.connect(self._presenter._on_app_mode_changed)
        for grid_view in self.grid_inspection_matrix_views.values():
            grid_view.recordSelected.connect(self._presenter._on_grid_inspection_record_selected)
            grid_view.recordActivated.connect(self._presenter._on_grid_inspection_record_activated)
            grid_view.contextMenuRequested.connect(self._presenter._on_grid_inspection_context_menu)
        self.grid_inspection_layer_tabs.currentChanged.connect(self._activate_grid_inspection_layer_tab)
        self.grid_inspection_error_filter.currentIndexChanged.connect(self._presenter._on_grid_inspection_error_filter_changed)
        self.grid_inspection_error_list.itemClicked.connect(self._presenter._on_grid_inspection_error_item_clicked)
        self.grid_inspection_error_list.itemActivated.connect(self._presenter._on_grid_inspection_error_item_clicked)
        self.grid_reference_frame_select_button.clicked.connect(self._presenter._on_grid_reference_select_requested)
        self.grid_reference_frame_clear_button.clicked.connect(self._presenter._on_grid_reference_clear_requested)
        for _metric_key, card in getattr(self, "grid_inspection_histogram_cards", {}).items():
            if hasattr(card, "binClicked"):
                card.binClicked.connect(
                    lambda clicked_metric_key, bin_index: self._presenter._on_grid_inspection_histogram_bin_selected(
                        str(clicked_metric_key),
                        int(bin_index),
                    )
                )
            if hasattr(card, "binDoubleClicked"):
                card.binDoubleClicked.connect(
                    lambda clicked_metric_key, bin_index: self._presenter._on_grid_inspection_histogram_bin_clicked(
                        str(clicked_metric_key),
                        int(bin_index),
                    )
                )
            if hasattr(card, "binContextMenuRequested"):
                card.binContextMenuRequested.connect(
                    lambda clicked_metric_key, bin_index, global_pos: self._presenter._on_grid_inspection_histogram_bin_context_menu(
                        str(clicked_metric_key),
                        int(bin_index),
                        global_pos,
                    )
                )
        if hasattr(self, "grid_inspection_repeated_bad_column"):
            self.grid_inspection_repeated_bad_column.columnClicked.connect(
                lambda band: self._presenter._on_grid_inspection_correlation_column_selected(str(band))
            )
            if hasattr(self.grid_inspection_repeated_bad_column, "columnDoubleClicked"):
                self.grid_inspection_repeated_bad_column.columnDoubleClicked.connect(
                    lambda band: self._presenter._on_grid_inspection_correlation_column_clicked(str(band))
                )
        if hasattr(self, "grid_inspection_repeated_good_column"):
            self.grid_inspection_repeated_good_column.columnClicked.connect(
                lambda band: self._presenter._on_grid_inspection_correlation_column_selected(str(band))
            )
            if hasattr(self.grid_inspection_repeated_good_column, "columnDoubleClicked"):
                self.grid_inspection_repeated_good_column.columnDoubleClicked.connect(
                    lambda band: self._presenter._on_grid_inspection_correlation_column_clicked(str(band))
                )
        if hasattr(self, "grid_inspection_correlation_limit_spin"):
            self.grid_inspection_correlation_limit_spin.valueChanged.connect(
                self._presenter._on_grid_inspection_correlation_limit_changed
            )
        if hasattr(self, "grid_inspection_percentile_full_matrix_check"):
            self.grid_inspection_percentile_full_matrix_check.toggled.connect(
                self._presenter._on_grid_inspection_percentile_display_mode_changed
            )
        self.btn_add_folder.clicked.connect(self._presenter._add_folder)
        self.btn_clear_folders.clicked.connect(self._presenter._clear_folders)
        self.btn_set_original.clicked.connect(self._presenter._set_original_folder)
        self.btn_clear_original.clicked.connect(self._presenter._clear_original_folder)
        self.btn_set_export.clicked.connect(self._presenter._set_export_folder)
        self.btn_clear_export.clicked.connect(self._presenter._clear_export_folder)
        self.btn_build.clicked.connect(self._presenter._on_build_requested)
        self.btn_compute.clicked.connect(self._presenter._on_compute_requested)
        self.btn_export_layer.clicked.connect(self._presenter._on_export_result_layer_requested)
        self.btn_export_grid_checks.clicked.connect(self._presenter._on_export_grid_check_bmps_requested)
        self.btn_cancel.clicked.connect(self._presenter._request_cancel_build)
        self.btn_frame_search.clicked.connect(self._presenter._on_frame_search_requested)
        self.frame_search_input.returnPressed.connect(self._presenter._on_frame_search_requested)
        self.folder_list.itemClicked.connect(self._presenter._on_folder_item_clicked)
        self.active_pair_list.itemClicked.connect(self._presenter._on_active_pair_item_clicked)
        self.active_pair_list.customContextMenuRequested.connect(self._presenter._on_active_pair_context_menu)

        self.matrix_score_view_combo.currentIndexChanged.connect(self._presenter._on_matrix_score_view_changed)
        self.matrix_gradient_combo.currentIndexChanged.connect(self._presenter._on_matrix_gradient_changed)
        self.analysis_setup_panel.profileChanged.connect(self._presenter._on_analysis_profile_changed)
        self.analysis_setup_panel.runRequested.connect(self._presenter._on_primary_run_requested)
        self.persistent_history_button.clicked.connect(self._open_persistent_analysis_history)
        self.run_history_list.itemClicked.connect(self._presenter._on_run_history_selected)
        self.thumbnail_size_spin.valueChanged.connect(self._presenter._on_matrix_visual_parameter_changed)
        self.analysis_mode_combo.currentIndexChanged.connect(self._presenter._on_analysis_mode_changed)
        self.comparison_target_combo.currentIndexChanged.connect(self._presenter._on_comparison_target_changed)
        self.geometry_mode_combo.currentIndexChanged.connect(self._presenter._on_object_type_changed)
        self.polygon_compare_profile_combo.currentIndexChanged.connect(self._presenter._on_polygon_compare_profile_changed)
        self.confidence_uncertainty_profile_combo.currentIndexChanged.connect(self._presenter._sync_action_buttons)
        self.polygon_confidence_summary_combo.currentIndexChanged.connect(self._presenter._sync_action_buttons)
        self.point_match_radius_spin.valueChanged.connect(self._presenter._sync_action_buttons)
        self.point_confidence_radius_spin.valueChanged.connect(self._presenter._sync_action_buttons)
        self.point_extraction_mode_combo.currentIndexChanged.connect(self._presenter._sync_action_buttons)
        self.frames_per_row_spin.valueChanged.connect(self._presenter._on_matrix_visual_parameter_changed)
        self.metric_scope_combo.currentIndexChanged.connect(self._presenter._on_metric_scope_changed)
        self.metric_combo.currentIndexChanged.connect(self._presenter._on_metric_changed)
        self.frame_type_filter_combo.currentIndexChanged.connect(self._presenter._on_frame_type_filter_changed)
        self.language_toggle_button.clicked.connect(self._toggle_language)
        self.matrix_tabs.currentChanged.connect(self._presenter._on_current_tab_changed)
        self.matrix_tabs.tabCloseRequested.connect(self._presenter._close_matrix_tab)

    def _open_persistent_analysis_history(self) -> StandaloneHistoryDialog:
        dialog = StandaloneHistoryDialog(parent=self)
        dialog.repeatRequested.connect(self.standaloneAnalysisRepeatRequested)
        dialog.exec()
        return dialog

    def set_workflow_summary(self, payload: dict[str, tuple[str, str, str]]) -> None:
        self.analysis_setup_panel.set_workflow_summary(payload)

    def set_analysis_preflight(self, report: AnalysisPreflightReport) -> None:
        self.analysis_setup_panel.set_preflight(report)

    def set_analysis_profile(self, profile: AnalysisProfileKind | str) -> None:
        self.analysis_setup_panel.set_profile(profile)

    def set_analysis_profile_availability(
        self,
        availability: dict[AnalysisProfileKind, tuple[bool, str]],
    ) -> None:
        self.analysis_setup_panel.set_profile_availability(availability)

    def _create_matrix_tab(self, build_result: BuildResult, snapshot: dict[str, object]) -> ExtendMatrixTabState:
        empty_index = self.matrix_tabs.indexOf(self.empty_matrix_page)
        if empty_index >= 0:
            self.matrix_tabs.removeTab(empty_index)
        self.matrix_tabs.setTabsClosable(True)
        self.matrix_tabs.tabBar().show()
        host = QWidget(self.matrix_tabs)
        matrix_layout = QHBoxLayout(host)
        matrix_layout.setContentsMargins(0, 0, 0, 0)

        content_tabs = QTabWidget(host)
        matrix_page = QWidget(content_tabs)
        matrix_page_layout = QVBoxLayout(matrix_page)
        matrix_page_layout.setContentsMargins(0, 0, 0, 0)
        matrix_legend = MatrixLegendWidget(matrix_page)
        matrix_page_layout.addWidget(matrix_legend)
        matrix_view = MatrixListWidget(matrix_page)
        matrix_page_layout.addWidget(matrix_view, stretch=1)
        histogram_metric_keys = tuple(dict.fromkeys(tuple(build_result.available_metric_keys) + (GRID_INSPECTION_DAMAGE_METRIC_KEY,)))
        charts_page, histogram_cards, repeated_bad_column, repeated_good_column, percentile_full_matrix_check, correlation_limit_spin = self._build_histograms_panel(content_tabs, histogram_metric_keys, build_result)
        content_tabs.addTab(matrix_page, self._t("tab.matrix"))
        content_tabs.addTab(charts_page, self._t("tab.percentiles"))
        matrix_layout.addWidget(content_tabs, stretch=1)

        overview_host = QWidget(host)
        overview_layout = QVBoxLayout(overview_host)
        overview_layout.setContentsMargins(0, 0, 0, 0)
        overview_layout.setSpacing(8)
        mini_map = MatrixMiniMapWidget(overview_host)
        overview_layout.addWidget(mini_map)
        preview = self._build_preview_panel(overview_host, tuple(build_result.available_metric_keys), build_result)
        overview_layout.addWidget(preview.group)
        overview_layout.addStretch(1)
        overview_host.setMinimumWidth(max(300, OVERVIEW_PANEL_MAX_WIDTH - 40))
        overview_host.setMaximumWidth(OVERVIEW_PANEL_MAX_WIDTH)
        overview_host.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        matrix_layout.addWidget(overview_host)

        preview.histogram_cards = histogram_cards
        state = ExtendMatrixTabState(
            widget=host,
            matrix_view=matrix_view,
            mini_map=mini_map,
            build_result=build_result,
            legend=matrix_legend,
            content_tabs=content_tabs,
            cell_size=int(snapshot["cell_size"]),
            layout_config=snapshot["layout_config"],
            matrix_score_view_mode=str(snapshot.get("matrix_score_view_mode") or DEFAULT_MATRIX_SCORE_VIEW_MODE),
            gradient_name=str(snapshot.get("gradient_name") or DEFAULT_GRADIENT_NAME),
            metric_key=str(snapshot.get("metric_key") or DEFAULT_MATRIX_METRIC_KEY),
            metric_scope=str(snapshot.get("metric_scope") or ""),
            analysis_mode=str(snapshot.get("analysis_mode") or DEFAULT_ANALYSIS_MODE),
            object_type=str(snapshot.get("object_type") or "polygon"),
            confidence_model_id=str(snapshot.get("confidence_model_id") or snapshot.get("metric_scope") or "") or None,
            frame_type_filter=str(snapshot.get("frame_type_filter") or snapshot.get("object_type") or "all"),
            preview=preview,
            percentile_filter_full_matrix=bool(percentile_full_matrix_check.isChecked()),
            percentile_full_matrix_check=percentile_full_matrix_check,
            correlation_limit=int(correlation_limit_spin.value()),
            correlation_limit_spin=correlation_limit_spin,
            repeated_bad_column=repeated_bad_column,
            repeated_good_column=repeated_good_column,
            excluded_record_keys={str(key) for key in snapshot.get("excluded_record_keys", ()) if str(key)},
        )
        matrix_view.set_excluded_record_keys(set(state.excluded_record_keys))
        matrix_view.colorScaleChanged.connect(matrix_legend.set_scale_info)
        matrix_legend.set_scale_info(matrix_view.color_scale_info())
        matrix_view.recordSelected.connect(lambda record, s=state: self._presenter._on_record_selected(s, record))
        matrix_view.recordActivated.connect(lambda record, s=state: self._presenter._open_record_details(record, s))
        matrix_view.contextMenuRequested.connect(lambda record, global_pos, s=state: self._presenter._on_matrix_context_menu(s, record, global_pos))
        matrix_view.overviewChanged.connect(lambda image, visible_rect, selected_position, selected_blink_on, processing_positions, reference_position, s=state: self._presenter._on_matrix_overview_changed(s, image, visible_rect, selected_position, selected_blink_on, processing_positions, reference_position))
        return state

    def show_empty_matrix_state(self) -> None:
        if self.matrix_tabs.indexOf(self.empty_matrix_page) < 0:
            self.matrix_tabs.addTab(self.empty_matrix_page, "")
        self.matrix_tabs.setTabsClosable(False)
        self.matrix_tabs.tabBar().hide()

    def shutdown(self) -> None:
        self._presenter.shutdown()

    def closeEvent(self, event) -> None:
        self._presenter.shutdown()
        super().closeEvent(event)


class KarakalMainWindow(QMainWindow):
    """Standalone host window for the extended widget."""

    def __init__(self) -> None:
        super().__init__()
        apply_karakal_icon(self)
        self._widget = KarakalWidget(self)
        self.setWindowTitle(self._widget._t("window.title"))
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        self.setCentralWidget(self._widget)

    def plugin_widget(self) -> KarakalWidget:
        return self._widget

    def closeEvent(self, event) -> None:
        self._widget.shutdown()
        super().closeEvent(event)
