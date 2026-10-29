"""CSV/XLSX/PDF adapters; calculation and report values live in the application layer."""
from __future__ import annotations
import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime, tzinfo
from pathlib import Path
from typing import Any, Iterable, Mapping

from kraken_manager.application.reporting import (
    ActivityRecord as ActivityRecord,
    METRIC_DEFINITIONS as METRIC_DEFINITIONS,
    MetricChartKind as MetricChartKind,
    MetricDefinition as MetricDefinition,
    MetricValueKind as MetricValueKind,
    PresentedMetric as PresentedMetric,
    ReportBucket as ReportBucket,
    ReportFilters as ReportFilters,
    ReportGranularity as ReportGranularity,
    ReportMetrics as ReportMetrics,
    ReportSeries as ReportSeries,
    _MetricsAccumulator as _MetricsAccumulator,
    format_metric_value as format_metric_value,
    metric_definition as metric_definition,
    present_metrics as present_metrics,
    ReportService as ApplicationReportService,
)

def _safe_tabular_text(value: object) -> object:
    """Prevent spreadsheet formula interpretation in CSV and XLSX exports."""

    if not isinstance(value, str) or not value:
        return value
    stripped = value.lstrip(" \t\r\n")
    if value[0] in "\t\r\n" or (stripped and stripped[0] in "=+-@"):
        return "'" + value
    return value



class ReportService(ApplicationReportService):
    def write_work_plan_csv(self, destination, report):
        from .work_plans import write_csv
        return write_csv(destination, report)


    def write_work_plan_xlsx(self, destination, report):
        from .work_plans import write_xlsx
        return write_xlsx(destination, report)


    def write_csv(
        self,
        destination: Path | str,
        records: Iterable[ActivityRecord],
        filters: ReportFilters,
        *,
        assume_sorted: bool = False,
    ) -> Path:
        """Write an UTF-8 journal.

        ``assume_sorted=True`` enables constant-memory streaming for a cursor
        already ordered by ``(recorded_at, event_id)``.  The default preserves
        the historical deterministic sorting contract for arbitrary iterables.
        """

        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        selected: Iterable[ActivityRecord]
        selected = self.iter_selected(records, filters) if assume_sorted else self.select(records, filters)
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(
                (
                    "event_id",
                    "recorded_at",
                    "event_type",
                    "project_id",
                    "layer_id",
                    "layer_type",
                    "representation_id",
                    "frame_id",
                    "actor_id",
                    "performer_id",
                    "tool",
                    "status",
                    "bytes_count",
                    "payload_json",
                )
            )
            for item in selected:
                row = (
                    item.event_id,
                    item.recorded_at.isoformat(),
                    item.event_type,
                    item.project_id,
                    item.layer_id or "",
                    item.layer_type or "",
                    item.representation_id or "",
                    item.frame_id or "",
                    item.actor_id or "",
                    item.performer_id or "",
                    item.tool or "",
                    item.status or "",
                    item.bytes_count,
                    json.dumps(dict(item.payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                )
                writer.writerow(tuple(_safe_tabular_text(value) for value in row))
        return output


    def write_xlsx(
        self,
        destination: Path | str,
        records: Iterable[ActivityRecord],
        filters: ReportFilters,
        *,
        assume_sorted: bool = False,
        include_series: bool = False,
        timezone: tzinfo | None = None,
    ) -> Path:
        try:
            from openpyxl import Workbook
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install Kraken with the 'reports' extra to export XLSX") from exc

        # write_only avoids retaining worksheet cells.  For an ordered database
        # cursor, assume_sorted also avoids retaining report rows themselves.
        series_records: tuple[ActivityRecord, ...] = ()
        source: Iterable[ActivityRecord]
        if include_series:
            series_records = tuple(records)
            source = (
                series_records
                if assume_sorted
                else sorted(series_records, key=lambda item: (item.recorded_at, item.event_id))
            )
        elif assume_sorted:
            source = records
        else:
            source = sorted(records, key=lambda item: (item.recorded_at, item.event_id))
        workbook = Workbook(write_only=True)
        summary = workbook.create_sheet("Сводка")
        series_sheets = {
            ReportGranularity.DAY: workbook.create_sheet("По дням"),
            ReportGranularity.WEEK: workbook.create_sheet("По неделям"),
            ReportGranularity.MONTH: workbook.create_sheet("По месяцам"),
            ReportGranularity.YEAR: workbook.create_sheet("По годам"),
        } if include_series else {}
        projects = workbook.create_sheet("Projects")
        layers_sheet = workbook.create_sheet("Layers")
        vector_layers_sheet = workbook.create_sheet("Vector layers")
        performers = workbook.create_sheet("Performers")
        tools_sheet = workbook.create_sheet("Tools")
        events = workbook.create_sheet("Events")
        events.append(
            (
                "Event ID",
                "Recorded at",
                "Type",
                "Project",
                "Layer",
                "Vector layer",
                "Frame",
                "Actor",
                "Performer",
                "Tool",
                "Status",
            )
        )
        accumulator = _MetricsAccumulator(filters)
        layer_counters: dict[str, Counter[str]] = defaultdict(Counter)
        tool_counters: dict[str, Counter[str]] = defaultdict(Counter)

        for item in source:
            selected = accumulator.add(item)
            if not selected:
                continue
            if item.layer_id:
                layer_counters[item.layer_id][item.event_type] += 1
            if item.tool:
                tool_counters[item.tool][item.event_type] += 1
            events.append(
                tuple(
                    _safe_tabular_text(value)
                    for value in (
                        item.event_id,
                        item.recorded_at.isoformat(),
                        item.event_type,
                        item.project_id,
                        item.layer_id,
                        item.representation_id,
                        item.frame_id,
                        item.actor_id,
                        item.performer_id,
                        item.tool,
                        item.status,
                    )
                )
            )

        metrics = accumulator.finish()
        summary.append(("Показатель", "Значение"))
        for metric in present_metrics(metrics.values):
            summary.append(
                (
                    _safe_tabular_text(metric.definition.label),
                    _safe_tabular_text(metric.formatted_value),
                )
            )
        local_timezone = timezone or datetime.now().astimezone().tzinfo or UTC
        for granularity, sheet in series_sheets.items():
            series = self.aggregate_series(
                series_records,
                filters,
                granularity,
                timezone=local_timezone,
            )
            sheet.append(("Период", *(definition.label for definition in METRIC_DEFINITIONS)))
            for bucket in series.buckets:
                presented = present_metrics(bucket.metrics.values)
                sheet.append(
                    (
                        bucket.label,
                        *(metric.formatted_value for metric in presented),
                    )
                )
        self._write_counter_sheet(projects, "Project", metrics.by_project)
        self._write_counter_sheet(layers_sheet, "Layer", layer_counters)
        self._write_counter_sheet(
            vector_layers_sheet, "Vector layer", metrics.by_vector_layer
        )
        self._write_counter_sheet(performers, "Performer", metrics.by_performer)
        self._write_counter_sheet(tools_sheet, "Tool", tool_counters)

        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        workbook.save(output)
        return output


    @staticmethod
    def _write_counter_sheet(sheet: Any, item_heading: str, counters: Mapping[str, Mapping[str, int]]) -> None:
        event_types = sorted({event for counter in counters.values() for event in counter})
        sheet.append(tuple(_safe_tabular_text(value) for value in (item_heading, *event_types)))
        for key, counter in sorted(counters.items()):
            sheet.append(
                (_safe_tabular_text(key), *(counter.get(event, 0) for event in event_types))
            )


    def write_pdf(
        self,
        destination: Path | str,
        records: Iterable[ActivityRecord],
        filters: ReportFilters,
        *,
        cyrillic_font: Path | str,
    ) -> Path:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.pdfgen.canvas import Canvas
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install Kraken with the 'reports' extra to export PDF") from exc
        font = Path(cyrillic_font).resolve(strict=True)
        metrics = self.aggregate(records, filters)
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        font_name = "KrakenCyrillic"
        pdfmetrics.registerFont(TTFont(font_name, str(font)))
        canvas = Canvas(str(output), pagesize=A4)
        _, height = A4
        canvas.setFont(font_name, 15)
        canvas.drawString(40, height - 45, "Kraken — отчёт по активности")
        canvas.setFont(font_name, 9)
        canvas.drawString(40, height - 65, f"Период: {filters.start.isoformat()} — {filters.end.isoformat()}")
        y = height - 90
        for key, value in sorted(metrics.values.items()):
            canvas.drawString(45, y, f"{key}: {value}")
            y -= 14
            if y < 45:
                canvas.showPage()
                canvas.setFont(font_name, 9)
                y = height - 45
        canvas.save()
        return output
