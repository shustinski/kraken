"""Compact guided setup for Karakal analysis runs."""
from __future__ import annotations

from collections.abc import Callable, Mapping

from PyQt6.QtCore import Qt, pyqtSignal
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


class AnalysisSetupPanel(QGroupBox):
    """Guide users from an analysis goal to validated source roles."""

    profileChanged = pyqtSignal(str)
    runRequested = pyqtSignal()

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
        self.intro_label = QLabel(self)
        self.intro_label.setWordWrap(True)
        layout.addWidget(self.intro_label)

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
        self.role_table.setMinimumHeight(150)
        self.role_table.setMaximumHeight(180)
        layout.addWidget(self.role_table)

        self.preflight_label = QLabel(self)
        self.preflight_label.setWordWrap(True)
        self.preflight_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.preflight_label)

        self.workflow_label = QLabel(self)
        self.workflow_label.setWordWrap(True)
        self.workflow_label.setStyleSheet("color: #aebdce;")
        layout.addWidget(self.workflow_label)

        self.run_button = QPushButton(self)
        self.run_button.setMinimumHeight(38)
        self.run_button.setProperty("primaryAction", True)
        self.run_button.clicked.connect(self.runRequested.emit)
        layout.addWidget(self.run_button)
        self.retranslate(translate)

    def retranslate(self, translate: Translate) -> None:
        self._t = translate
        self.setTitle(self._t("setup.group"))
        self.intro_label.setText(self._t("setup.intro"))
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
        self.run_button.setText(self._t("setup.run"))
        if not self.preflight_label.text():
            self.set_preflight(None)

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

    def set_preflight(self, report: AnalysisPreflightReport | None) -> None:
        if report is None:
            self.preflight_label.setText(self._t("preflight.pending"))
            self.preflight_label.setStyleSheet("padding: 6px; color: #aebdce; background: #151b23; border-radius: 6px;")
            self.run_button.setEnabled(False)
            return
        state_keys = {
            "ready": "workflow.state.ready",
            "partial": "workflow.state.partial",
            "missing": "workflow.state.pending",
            "empty": "preflight.empty",
        }
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
            source_item = QTableWidgetItem(coverage)
            source_item.setToolTip(role_status.detail)
            self.role_table.setItem(row, 1, source_item)
            state_key = state_keys.get(role_status.state, "workflow.state.pending")
            self.role_table.setItem(row, 2, QTableWidgetItem(self._t(state_key)))
        errors = [issue for issue in report.issues if issue.severity == PreflightSeverity.ERROR]
        warnings = [issue for issue in report.issues if issue.severity == PreflightSeverity.WARNING]
        if errors:
            details = "\n".join(f"• {self._t(issue.message_key, count=issue.detail)} {issue.detail}".strip() for issue in errors)
            self.preflight_label.setText(f"{self._t('preflight.blocked')}\n{details}")
            style = "padding: 6px; color: #ffd9de; background: #4a2028; border: 1px solid #8c3948; border-radius: 6px;"
        elif warnings:
            details = "\n".join(f"• {self._t(issue.message_key, count=issue.detail)} {issue.detail}".strip() for issue in warnings)
            self.preflight_label.setText(
                f"{self._t('preflight.ready_with_warnings', matched=report.matched_frames, total=report.total_frames)}\n{details}"
            )
            style = "padding: 6px; color: #ffe9bd; background: #493719; border: 1px solid #8a6424; border-radius: 6px;"
        else:
            self.preflight_label.setText(
                self._t("preflight.ready", matched=report.matched_frames, total=report.total_frames)
            )
            style = "padding: 6px; color: #d7f8e4; background: #183c2a; border: 1px solid #2d7750; border-radius: 6px;"
        self.preflight_label.setStyleSheet(style)
        self.run_button.setEnabled(report.can_run)

    def set_workflow_summary(self, payload: Mapping[str, tuple[str, str, str]]) -> None:
        parts: list[str] = []
        for key, title_key in (
            ("sources", "workflow.sources"),
            ("models", "workflow.models"),
            ("analysis", "workflow.analysis"),
        ):
            state, detail, _tone = payload.get(key, (self._t("workflow.state.pending"), "", "idle"))
            parts.append(f"{self._t(title_key)}: {state} — {detail}")
        self.workflow_label.setText("\n".join(parts))

    def set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(self.run_button.isEnabled() and not busy)
        for button in self._profile_buttons.values():
            button.setEnabled(not busy)


__all__ = ["AnalysisSetupPanel"]
