"""Operator-facing controls for creating and monitoring Karakal analysis runs."""

from __future__ import annotations

from collections.abc import Iterable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kraken_core.analysis_run_protocol import AnalysisExpression, AnalysisRecipe


SCENARIO_DIRECT = "direct"
SCENARIO_XOR_A = "xor_a"
SCENARIO_XOR_C = "xor_c"
SCENARIO_SUBTRACT_C = "subtract_c"


class AnalysisSetupDialog(QDialog):
    """Compact setup for a project analysis run."""

    def __init__(
        self,
        sources: Iterable[tuple[str, str]] = (),
        *,
        selected_frame_count: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("analysisSetupDialog")
        self.setWindowTitle("Оценить результат")
        self.setMinimumWidth(520)
        root = QVBoxLayout(self)
        intro = QLabel(
            "Выберите кадры, назначьте результаты моделей буквам A/B/C и задайте выражение сравнения."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)
        form = QFormLayout()
        self.frame_count = QSpinBox()
        self.frame_count.setObjectName("analysisFrameCount")
        self.frame_count.setRange(0, 100_000_000)
        self.frame_count.setValue(max(0, int(selected_frame_count)))
        self.frame_count.setReadOnly(True)
        form.addRow("Кадры выборки:", self.frame_count)
        self.source_a = QComboBox()
        self.source_a.setObjectName("analysisSourceA")
        self.source_b = QComboBox()
        self.source_b.setObjectName("analysisSourceB")
        self.source_c = QComboBox()
        self.source_c.setObjectName("analysisSourceC")
        for identifier, label in sources:
            for combo in (self.source_a, self.source_b, self.source_c):
                combo.addItem(str(label), str(identifier))
        if self.source_b.count() > 1:
            self.source_b.setCurrentIndex(1)
        form.addRow("Модель A:", self.source_a)
        form.addRow("Модель B:", self.source_b)
        form.addRow("Модель C:", self.source_c)
        self.scenario = QComboBox()
        self.scenario.setObjectName("analysisScenario")
        self.scenario.addItem("Сравнить A с B", SCENARIO_DIRECT)
        self.scenario.addItem("Сравнить A XOR B с A", SCENARIO_XOR_A)
        self.scenario.addItem("Сравнить A XOR B с C", SCENARIO_XOR_C)
        self.scenario.addItem("Сравнить B − A с C", SCENARIO_SUBTRACT_C)
        form.addRow("Сценарий:", self.scenario)
        self.threshold = QSpinBox()
        self.threshold.setObjectName("analysisThresholdPercent")
        self.threshold.setRange(0, 100)
        self.threshold.setValue(50)
        self.threshold.setSuffix(" %")
        form.addRow("Порог маски:", self.threshold)
        self.metric_summary = QLabel("Все доступные метрики: XOR, IoU и Dice")
        self.metric_summary.setObjectName("analysisMetricSummary")
        form.addRow("Метрики:", self.metric_summary)
        root.addLayout(form)
        self.validation_label = QLabel()
        self.validation_label.setObjectName("analysisValidationLabel")
        self.validation_label.setWordWrap(True)
        root.addWidget(self.validation_label)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        self.start_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.start_button.setText("Запустить анализ")
        self.start_button.setObjectName("startAnalysisButton")
        root.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.scenario.currentIndexChanged.connect(self._validate)
        self.frame_count.valueChanged.connect(self._validate)
        for combo in (self.source_a, self.source_b, self.source_c):
            combo.currentIndexChanged.connect(self._validate)
        self._validate()

    def _validate(self) -> None:
        needs_c = self.scenario.currentData() in {SCENARIO_XOR_C, SCENARIO_SUBTRACT_C}
        self.source_c.setEnabled(needs_c)
        problems: list[str] = []
        if self.frame_count.value() < 1:
            problems.append("Выборка не содержит кадров.")
        if self.source_a.currentData() is None or self.source_b.currentData() is None:
            problems.append("Назначьте модели A и B.")
        elif self.source_a.currentData() == self.source_b.currentData():
            problems.append("Модели A и B должны быть разными версиями.")
        if needs_c and self.source_c.currentData() is None:
            problems.append("Для выбранного сценария назначьте модель C.")
        self.validation_label.setText(" ".join(problems) if problems else "Конфигурация готова к запуску.")
        self.start_button.setEnabled(not problems)

    def recipe(self) -> AnalysisRecipe:
        source_a = AnalysisExpression.source("A")
        source_b = AnalysisExpression.source("B")
        scenario = str(self.scenario.currentData())
        if scenario == SCENARIO_DIRECT:
            expression = AnalysisExpression.binary("compare", source_a, source_b)
        elif scenario == SCENARIO_XOR_A:
            expression = AnalysisExpression.binary(
                "compare", AnalysisExpression.binary("xor", source_a, source_b), source_a
            )
        elif scenario == SCENARIO_XOR_C:
            expression = AnalysisExpression.binary(
                "compare",
                AnalysisExpression.binary("xor", source_a, source_b),
                AnalysisExpression.source("C"),
            )
        elif scenario == SCENARIO_SUBTRACT_C:
            expression = AnalysisExpression.binary(
                "compare",
                AnalysisExpression.binary("subtract", source_b, source_a),
                AnalysisExpression.source("C"),
            )
        else:
            raise ValueError(f"Unsupported analysis scenario: {scenario}")
        return AnalysisRecipe(expression)

    def configuration(self) -> dict[str, object]:
        bindings = {"A": str(self.source_a.currentData()), "B": str(self.source_b.currentData())}
        if self.source_c.isEnabled():
            bindings["C"] = str(self.source_c.currentData())
        return {
            "frame_count": self.frame_count.value(),
            "bindings": bindings,
            "recipe": self.recipe().to_payload(),
            "parameters": {"mask_threshold": self.threshold.value() / 100.0},
        }


class AnalysisRunsPanel(QFrame):
    retryRequested = pyqtSignal(str)
    cancelRequested = pyqtSignal(str)
    repeatRequested = pyqtSignal(str)
    exportRequested = pyqtSignal(str)
    renderMapRequested = pyqtSignal(str, str)
    metricChanged = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("analysisRunsPanel")
        root = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(QLabel("Запуски анализа"))
        self.progress = QProgressBar()
        self.progress.setObjectName("analysisOverallProgress")
        self.progress.setRange(0, 100)
        header.addWidget(self.progress, 1)
        self.metric_combo = QComboBox()
        self.metric_combo.setObjectName("analysisMetricCombo")
        for key in ("xor", "iou", "dice"):
            self.metric_combo.addItem(key.upper(), key)
        header.addWidget(self.metric_combo)
        root.addLayout(header)
        self.run_table = QTableWidget(0, 5)
        self.run_table.setObjectName("analysisRunTable")
        self.run_table.setHorizontalHeaderLabels(("Статус", "Прогресс", "Модели", "Рецепт", "Запущен"))
        self.run_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        root.addWidget(self.run_table)
        self.result_table = QTableWidget(0, 6)
        self.result_table.setObjectName("analysisResultTable")
        self.result_table.setHorizontalHeaderLabels(("Кадр", "X", "Y", "Статус", "Значение", "Процентиль"))
        root.addWidget(self.result_table, 1)
        actions = QHBoxLayout()
        self.retry_button = QPushButton("Повторить неудачные партии")
        self.cancel_button = QPushButton("Отменить")
        self.repeat_button = QPushButton("Повторить запуск")
        self.map_button = QPushButton("Показать карту")
        self.export_button = QPushButton("Экспортировать")
        for button in (
            self.retry_button,
            self.cancel_button,
            self.repeat_button,
            self.map_button,
            self.export_button,
        ):
            actions.addWidget(button)
        actions.addStretch(1)
        root.addLayout(actions)
        self.metric_combo.currentIndexChanged.connect(
            lambda: self.metricChanged.emit(str(self.metric_combo.currentData() or ""))
        )
        self.retry_button.clicked.connect(lambda: self.retryRequested.emit(self.selected_run_id()))
        self.cancel_button.clicked.connect(lambda: self.cancelRequested.emit(self.selected_run_id()))
        self.repeat_button.clicked.connect(lambda: self.repeatRequested.emit(self.selected_run_id()))
        self.export_button.clicked.connect(lambda: self.exportRequested.emit(self.selected_run_id()))
        self.map_button.clicked.connect(
            lambda: self.renderMapRequested.emit(self.selected_run_id(), self.selected_frame_id())
        )

    def selected_run_id(self) -> str:
        row = self.run_table.currentRow()
        item = self.run_table.item(row, 0) if row >= 0 else None
        return "" if item is None else str(item.data(Qt.ItemDataRole.UserRole) or "")

    def selected_frame_id(self) -> str:
        row = self.result_table.currentRow()
        item = self.result_table.item(row, 0) if row >= 0 else None
        return "" if item is None else item.text()

    def set_runs(self, runs: Iterable[dict[str, object]]) -> None:
        values = tuple(runs)
        self.run_table.setRowCount(len(values))
        for row, run in enumerate(values):
            status = QTableWidgetItem(str(run.get("state", "")))
            status.setData(Qt.ItemDataRole.UserRole, str(run.get("run_id", "")))
            self.run_table.setItem(row, 0, status)
            self.run_table.setItem(row, 1, QTableWidgetItem(str(run.get("progress", ""))))
            self.run_table.setItem(row, 2, QTableWidgetItem(str(run.get("models", ""))))
            self.run_table.setItem(row, 3, QTableWidgetItem(str(run.get("recipe", ""))))
            self.run_table.setItem(row, 4, QTableWidgetItem(str(run.get("created_at", ""))))

    def set_results(self, results: Iterable[dict[str, object]]) -> None:
        values = tuple(results)
        self.result_table.setRowCount(len(values))
        for row, result in enumerate(values):
            columns = (
                result.get("frame_id", ""),
                result.get("x", ""),
                result.get("y", ""),
                result.get("status", ""),
                result.get("raw_value", ""),
                result.get("percentile", ""),
            )
            for column, value in enumerate(columns):
                self.result_table.setItem(row, column, QTableWidgetItem(str(value)))


__all__ = [
    "AnalysisRunsPanel",
    "AnalysisSetupDialog",
    "SCENARIO_DIRECT",
    "SCENARIO_SUBTRACT_C",
    "SCENARIO_XOR_A",
    "SCENARIO_XOR_C",
]
