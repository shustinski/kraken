"""model_train_and_recognition: save chart csv."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Any


class SaveChartCsvMixin:
    @staticmethod
    def _save_chart_csv(
        *,
        save_path: Path,
        x_label: str,
        series: list[tuple[str, list[tuple[float, float]], tuple[int, int, int]]],
    ) -> None:
        non_empty_series = [(label, points) for label, points, _color in series if points]
        if not non_empty_series:
            return

        x_values = sorted({float(x_value) for _label, points in non_empty_series for x_value, _y_value in points})
        series_lookup: dict[str, dict[float, float]] = {
            str(label): {float(x_value): float(y_value) for x_value, y_value in points}
            for label, points in non_empty_series
        }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open('w', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            labels = [str(label) for label, _points in non_empty_series]
            writer.writerow([str(x_label), *labels])
            for x_value in x_values:
                x_cell = int(x_value) if abs(x_value - round(x_value)) < 1e-9 else x_value
                row: list[Any] = [x_cell]
                for label in labels:
                    row.append(series_lookup[label].get(float(x_value), ''))
                writer.writerow(row)
