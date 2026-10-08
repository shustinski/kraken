"""Compact guided setup for Karakal analysis runs."""
from __future__ import annotations

from collections.abc import Callable, Mapping

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kraken_core.analysis_protocol import AnalysisProfileKind, AnalysisSourceRole

from ..core.analysis_profiles import ANALYSIS_PROFILES, AnalysisPreflightReport, PreflightSeverity


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
}


class AnalysisSetupPanel(QGroupBox):
    """Analysis goal and the check of its sources; the run itself is the toolbar's main button."""

    profileChanged = pyqtSignal(str)

    def __init__(self, translate: Translate, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._t = translate
        self._profile_buttons: dict[AnalysisProfileKind, QPushButton] = {}
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

        profile_host = QWidget(self)
        profile_layout = QGridLayout(profile_host)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        profile_layout.setHorizontalSpacing(6)
        profile_layout.setVerticalSpacing(6)
        for index, profile in enumerate(ANALYSIS_PROFILES):
            button = QPushButton(profile_host)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(58)
            button.setProperty("analysisProfile", True)
            button.clicked.connect(lambda checked, key=profile.key: self._profile_clicked(key, checked))
            self._profile_buttons[profile.key] = button
            self._button_group.addButton(button)
            profile_layout.addWidget(button, index // 2, index % 2)
        layout.addWidget(profile_host)
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
        for profile in ANALYSIS_PROFILES:
            button = self._profile_buttons[profile.key]
            title = self._t(profile.title_key)
            description = self._t(profile.description_key)
            button.setText(title)
            button.setToolTip(description)
            if button.isChecked():
                self.profile_description_label.setText(description)
        self.set_preflight(self._report)

    def _profile_clicked(self, key: AnalysisProfileKind, checked: bool) -> None:
        if checked:
            profile = next(candidate for candidate in ANALYSIS_PROFILES if candidate.key == key)
            self.profile_description_label.setText(self._t(profile.description_key))
            self.profileChanged.emit(key.value)

    def set_profile(self, value: AnalysisProfileKind | str) -> None:
        key = AnalysisProfileKind(str(value))
        self._profile_buttons[key].setChecked(True)
        profile = next(candidate for candidate in ANALYSIS_PROFILES if candidate.key == key)
        self.profile_description_label.setText(self._t(profile.description_key))

    def set_profile_availability(
        self,
        availability: Mapping[AnalysisProfileKind, tuple[bool, str]],
    ) -> None:
        for key, button in self._profile_buttons.items():
            available, reason = availability.get(key, (True, ""))
            button.setProperty("profileAvailable", available)
            description_key = next(profile.description_key for profile in ANALYSIS_PROFILES if profile.key == key)
            base_tooltip = self._t(description_key)
            button.setToolTip(base_tooltip if available or not reason else f"{base_tooltip}\n\n{reason}")
            button.style().unpolish(button)
            button.style().polish(button)

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
        for button in self._profile_buttons.values():
            button.setEnabled(not busy)


__all__ = ["AnalysisSetupPanel"]
