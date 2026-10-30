"""Compact guided setup for Karakal analysis runs."""
from __future__ import annotations

from collections.abc import Callable, Mapping

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kraken_core.analysis_protocol import AnalysisProfileKind, AnalysisSourceRole

from ..core.analysis_profiles import AnalysisPreflightReport, PreflightSeverity, analysis_profile_definition


Translate = Callable[..., str]


# Status colours of the source table: ready, partly covered, blocking, not set.
_STATUS_COLORS = {"ready": "#7fd6a0", "partial": "#f0c674", "blocked": "#f08a8a", "idle": "#8a97a5"}
# Which source row a blocking problem belongs to.
_ISSUE_ROLES = {
    "models_required": AnalysisSourceRole.MODEL_OUTPUT,
    "single_model_required": AnalysisSourceRole.MODEL_OUTPUT,
    "model_output_required": AnalysisSourceRole.MODEL_OUTPUT,
    "grid_source_required": AnalysisSourceRole.MODEL_OUTPUT,
    "no_frames": AnalysisSourceRole.MODEL_OUTPUT,
    "duplicate_model_id": AnalysisSourceRole.MODEL_OUTPUT,
    "duplicate_frame_key": AnalysisSourceRole.MODEL_OUTPUT,
    "confidence_required": AnalysisSourceRole.CONFIDENCE,
    "photo_source_required": AnalysisSourceRole.ORIGINAL,
    "single_photo_source_required": AnalysisSourceRole.ORIGINAL,
}


# The two application modes and their switch labels.
APP_MODES: tuple[tuple[str, str], ...] = (
    ("validation", "app_mode.validation"),
    ("grid_inspection", "grid_inspection.mode"),
)


class AnalysisSetupPanel(QGroupBox):
    """Mode switch and the check of its sources; the run itself is the toolbar's main button.

    Validation picks its analysis from the checked sources, so the panel only shows which
    one it is; cell inspection has a single analysis.
    """

    modeChanged = pyqtSignal(str)

    def __init__(self, translate: Translate, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._t = translate
        self._mode_buttons: dict[str, QPushButton] = {}
        self._profile = AnalysisProfileKind.MODEL_COMPARISON
        self._unavailable_reason = ""
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._role_rows = {
            AnalysisSourceRole.ORIGINAL: 0,
            AnalysisSourceRole.MODEL_OUTPUT: 1,
            AnalysisSourceRole.CONFIDENCE: 2,
        }

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.mode_host = QWidget(self)
        mode_layout = QHBoxLayout(self.mode_host)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(6)
        for mode, _label_key in APP_MODES:
            button = QPushButton(self.mode_host)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(40)
            button.setProperty("appMode", mode)
            button.clicked.connect(lambda checked, value=mode: self._mode_clicked(value, checked))
            self._mode_buttons[mode] = button
            self._button_group.addButton(button)
            mode_layout.addWidget(button, 1)
        self._mode_buttons["validation"].setChecked(True)
        layout.addWidget(self.mode_host)
        self.profile_description_label = QLabel(self)
        self.profile_description_label.setWordWrap(True)
        self.profile_description_label.setStyleSheet("color: #aebdce; padding: 2px 4px;")
        layout.addWidget(self.profile_description_label)

        self.role_table = QTableWidget(len(self._role_rows), 3, self)
        self.role_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.role_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.role_table.verticalHeader().setVisible(False)
        self.role_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.role_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.role_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.role_table.setMinimumHeight(118)
        self.role_table.setMaximumHeight(124)
        layout.addWidget(self.role_table)
        self._report: AnalysisPreflightReport | None = None
        self.retranslate(translate)

    def retranslate(self, translate: Translate) -> None:
        self._t = translate
        self.setTitle(self._t("setup.group"))
        self.role_table.setHorizontalHeaderLabels(
            (self._t("setup.role"), self._t("setup.source_coverage"), self._t("setup.status"))
        )
        role_keys = {
            AnalysisSourceRole.ORIGINAL: "setup.role.original",
            AnalysisSourceRole.MODEL_OUTPUT: "setup.role.model_outputs",
            AnalysisSourceRole.CONFIDENCE: "setup.role.confidence",
        }
        for role, row in self._role_rows.items():
            self.role_table.setItem(row, 0, QTableWidgetItem(self._t(role_keys[role])))
            if self.role_table.item(row, 1) is None:
                self.role_table.setItem(row, 1, QTableWidgetItem(self._t("setup.not_configured")))
            if self.role_table.item(row, 2) is None:
                self.role_table.setItem(row, 2, QTableWidgetItem(self._t("workflow.state.pending")))
        for mode, label_key in APP_MODES:
            self._mode_buttons[mode].setText(self._t(label_key))
        self._mode_buttons["validation"].setToolTip(self._t("app_mode.validation.description"))
        self._mode_buttons["grid_inspection"].setToolTip(
            self._t(analysis_profile_definition(AnalysisProfileKind.GRID_DEFECTS).description_key)
        )
        self._refresh_description()
        self.set_preflight(self._report)

    def current_mode(self) -> str:
        return analysis_profile_definition(self._profile).app_mode

    def _mode_clicked(self, mode: str, checked: bool) -> None:
        if checked:
            self.modeChanged.emit(mode)

    def _refresh_description(self) -> None:
        profile = analysis_profile_definition(self._profile)
        description = self._t(profile.description_key)
        if profile.app_mode == "validation":
            text = self._t("setup.auto_profile", title=self._t(profile.title_key), description=description)
            tooltip = self._t("setup.auto_profile.hint")
        else:
            text, tooltip = description, ""
        if self._unavailable_reason:
            text = f"{text}\n{self._unavailable_reason}"
        self.profile_description_label.setText(text)
        self.profile_description_label.setToolTip(tooltip)

    def set_profile(self, value: AnalysisProfileKind | str) -> None:
        """Show the active analysis: its mode button and, for validation, which check was picked."""

        self._profile = analysis_profile_definition(value).key
        button = self._mode_buttons.get(self.current_mode())
        if button is not None and not button.isChecked():
            button.setChecked(True)
        self._refresh_description()

    def set_profile_availability(
        self,
        availability: Mapping[AnalysisProfileKind, tuple[bool, str]],
    ) -> None:
        available, reason = availability.get(self._profile, (True, ""))
        self._unavailable_reason = "" if available else str(reason or "")
        self._refresh_description()

    def issue_text(self, issue) -> str:
        message = self._t(issue.message_key, count=issue.detail)
        detail = str(issue.detail or "")
        return f"{message} ({detail})" if detail and detail not in message else message

    def blocking_reason(self) -> str:
        """Why the analysis cannot start now; empty when it can."""

        report = self._report
        if report is None:
            return self._t("preflight.pending")
        errors = [issue for issue in report.issues if issue.severity == PreflightSeverity.ERROR]
        return "\n".join(self.issue_text(issue) for issue in errors)

    def set_preflight(self, report: AnalysisPreflightReport | None) -> None:
        """Coverage and status per source; blocking problems colour their row and fill its tooltip."""

        self._report = report
        if report is None:
            return
        state_keys = {
            "ready": ("workflow.state.ready", "ready"),
            "partial": ("workflow.state.partial", "partial"),
            "missing": ("setup.not_configured", "idle"),
            "empty": ("preflight.empty", "partial"),
        }
        problems: dict[AnalysisSourceRole, list[str]] = {}
        for issue in report.issues:
            role = _ISSUE_ROLES.get(issue.code)
            if role is not None:
                problems.setdefault(role, []).append(self.issue_text(issue))
        warnings = [
            self.issue_text(issue) for issue in report.issues if issue.severity == PreflightSeverity.WARNING
        ]
        for role_status in report.roles:
            row = self._role_rows.get(role_status.role)
            if row is None:
                continue
            coverage = self._t(
                "preflight.coverage",
                sources=role_status.source_count,
                matched=role_status.matched_count,
                total=role_status.frame_count,
            )
            state_key, tone = state_keys.get(role_status.state, ("workflow.state.pending", "idle"))
            role_problems = problems.get(role_status.role, [])
            tooltip_lines = [role_status.detail] if role_status.detail else []
            if role_problems:
                state_key, tone = "setup.state.blocked", "blocked"
                tooltip_lines.extend(role_problems)
            elif tone == "partial":
                tooltip_lines.extend(warnings)
            source_item = QTableWidgetItem(coverage)
            source_item.setToolTip("\n".join(tooltip_lines))
            self.role_table.setItem(row, 1, source_item)
            status_item = QTableWidgetItem(self._t(state_key))
            status_item.setForeground(QColor(_STATUS_COLORS[tone]))
            status_item.setToolTip("\n".join(tooltip_lines))
            self.role_table.setItem(row, 2, status_item)

    def set_busy(self, busy: bool) -> None:
        for button in self._mode_buttons.values():
            button.setEnabled(not busy)


__all__ = ["APP_MODES", "AnalysisSetupPanel"]
