"""Quality desk controls for measured AnalysisRun rule evaluation."""

from functools import partial
from uuid import uuid4

from PyQt6.QtWidgets import QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton


class QualityAnalysisControls:
    def _build_analysis_controls(self, body):
        group = QGroupBox("Контроль по результатам анализа")
        form = QFormLayout(group)
        self.analysis_runs = QComboBox()
        self.analysis_runs.setObjectName("qualityAnalysisRuns")
        row = QHBoxLayout()
        row.addWidget(self.analysis_runs, 1)
        load = QPushButton("Загрузить прогоны")
        load.clicked.connect(self.load_analysis_runs)
        row.addWidget(load)
        readiness = QPushButton("Готовность прогона")
        readiness.clicked.connect(self.show_analysis_readiness)
        row.addWidget(readiness)
        form.addRow("Измеренный прогон", row)
        row = QHBoxLayout()
        self.analysis_metric = QComboBox()
        self.analysis_value = QComboBox()
        self.analysis_value.addItem("Исходное значение", "raw_value")
        self.analysis_value.addItem("Нормированная оценка", "goodness")
        self.analysis_comparison = QComboBox()
        for label, value in (("<", "lt"), ("≤", "le"), (">", "gt"), ("≥", "ge")):
            self.analysis_comparison.addItem(label, value)
        self.analysis_threshold = QLineEdit()
        self.analysis_threshold.setObjectName("qualityThreshold")
        self.analysis_threshold.setPlaceholderText("Введите порог; значения по умолчанию нет")
        for control in (self.analysis_metric, self.analysis_value, self.analysis_comparison, self.analysis_threshold):
            row.addWidget(control)
        form.addRow("Дефект при условии", row)
        self.analysis_binding = QComboBox()
        form.addRow("Исправляемый источник", self.analysis_binding)
        form.addRow(QLabel("Остальные источники и метод расчёта при повторной проверке должны остаться прежними."))
        row = QHBoxLayout()
        create = QPushButton("Создать дефекты из измерений")
        create.clicked.connect(self.create_analysis_findings)
        row.addWidget(create)
        recheck = QPushButton("Проверить исправление выбранным прогоном")
        recheck.clicked.connect(self.recheck_analysis_finding)
        row.addWidget(recheck)
        form.addRow(row)
        self.analysis_runs.currentIndexChanged.connect(self._analysis_run_selected)
        body.addWidget(group)

    def load_analysis_runs(self):
        self._run(partial(self.service.quality_analysis_runs, self.project_id), self._show_analysis_runs)

    def _show_analysis_runs(self, runs):
        previous = self.analysis_runs.currentData()
        selected = None if previous is None else previous["run_id"]
        self.analysis_runs.clear()
        for run in runs:
            self.analysis_runs.addItem(
                f"{run['run_id'][:12]} — {run['state']} — {run['completed_frames']}/{run['total_frames']}", run)
        index = next((i for i in range(self.analysis_runs.count())
                      if self.analysis_runs.itemData(i)["run_id"] == selected), 0)
        self.analysis_runs.setCurrentIndex(index if self.analysis_runs.count() else -1)
        self._analysis_run_selected()

    def _analysis_run_selected(self):
        run = self.analysis_runs.currentData()
        self.analysis_metric.clear()
        self.analysis_binding.clear()
        if run is not None:
            for metric in run["metric_keys"]:
                self.analysis_metric.addItem(metric, metric)
            for binding in run["bindings"]:
                self.analysis_binding.addItem(binding, binding)

    def create_analysis_findings(self):
        run = self.analysis_runs.currentData()
        try:
            if run is None or not self.analysis_metric.currentData() or not self.analysis_binding.currentData():
                raise ValueError("Выберите завершённый прогон, рассчитанную метрику и исправляемый источник")
            if not self.analysis_threshold.text().strip():
                raise ValueError("Укажите явный порог качества")
            threshold = float(self.analysis_threshold.text().replace(",", "."))
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        rule = {"metric_key": self.analysis_metric.currentData(), "value_field": self.analysis_value.currentData(),
                "comparison": self.analysis_comparison.currentData(), "threshold": threshold,
                "target_binding": self.analysis_binding.currentData()}
        self._run(partial(self.service.create_quality_from_analysis, project_id=self.project_id, principal=self.principal,
            run_id=run["run_id"], rule=rule, severity=self.severity.currentData(), idempotency_key=str(uuid4())),
            self._analysis_findings_created)

    def _analysis_findings_created(self, issues):
        self.issue = issues[0] if issues else None
        self._show_issues(issues, self.issue.id if self.issue else None)
        if not issues:
            self.history.setPlainText("Измерения не нарушают указанное правило. Результат проверки сохранён в истории проекта.")

    def recheck_analysis_finding(self):
        run = self.analysis_runs.currentData()
        if self.issue is None or self.issue.analysis_rule is None or run is None:
            self.status.setText("Выберите дефект, полученный из анализа, и повторный прогон")
            return
        self._run(partial(self.service.recheck_quality_from_analysis, project_id=self.project_id, principal=self.principal,
            issue_id=self.issue.id, expected_revision=self.issue.revision, run_id=run["run_id"],
            note=self.note.text(), idempotency_key=str(uuid4())), self._saved)

    def show_analysis_readiness(self):
        run = self.analysis_runs.currentData()
        if run is None:
            self.status.setText("Выберите прогон анализа")
            return
        self._run(partial(self.service.quality_analysis_readiness, self.project_id, run["run_id"]),
            self._show_analysis_readiness)

    def _show_analysis_readiness(self, result):
        self.history.setPlainText(
            ("Прогон готов к использованию" if result["ready"] else "Прогон пока не готов") + "\n" + result["reason"] +
            "\nБлокирующие дефекты: " + str(len(result["blocking_issue_ids"])))
