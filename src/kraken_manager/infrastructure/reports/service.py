"""Deterministic statistics with drill-down to the underlying activities.

The report adapter accepts both the public, stable activity names and the
CamelCase domain-event names emitted by the application layer.  Classification
of the generic ``ArtifactVersionCreated`` event is deliberately based on its
provenance/format payload: creating a version is not necessarily an import or a
vectorization.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


@dataclass(frozen=True, slots=True)
class ActivityRecord:
    event_id: str
    recorded_at: datetime
    event_type: str
    project_id: str
    layer_id: str | None = None
    layer_type: str | None = None
    frame_id: str | None = None
    actor_id: str | None = None
    performer_id: str | None = None
    tool: str | None = None
    status: str | None = None
    bytes_count: int = 0
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.recorded_at.tzinfo is None:
            raise ValueError("Activity timestamps must be timezone-aware")
        if isinstance(self.bytes_count, bool) or not isinstance(self.bytes_count, int):
            raise ValueError("Activity bytes_count must be an integer")

    @classmethod
    def from_event(cls, event: Any) -> ActivityRecord:
        """Build a report row from a domain ``EventEnvelope``.

        ``Any`` is intentional here: the reports extra is an infrastructure
        adapter and does not require callers to import the domain type merely
        to convert a stored/upcast event.
        """

        payload = dict(event.payload)
        program = getattr(event, "program", None)
        raw_size = payload.get("size_bytes", payload.get("bytes_count", 0))
        size = raw_size if isinstance(raw_size, int) and not isinstance(raw_size, bool) else 0
        return cls(
            event_id=str(event.event_id),
            recorded_at=event.recorded_at,
            event_type=str(event.event_type),
            project_id=str(event.project_id),
            layer_id=_optional_text(payload.get("layer_id")),
            layer_type=_optional_text(payload.get("layer_type")),
            frame_id=_optional_text(payload.get("frame_id")),
            actor_id=str(event.actor.principal_id),
            performer_id=_optional_text(getattr(event, "performer_id", None)),
            tool=(
                _optional_text(getattr(program, "name", None))
                or _optional_text(payload.get("tool_name"))
                or _optional_text(payload.get("tool"))
            ),
            status=_optional_text(payload.get("status")),
            bytes_count=size,
            payload=payload,
        )


@dataclass(frozen=True, slots=True)
class ReportFilters:
    start: datetime
    end: datetime
    project_ids: frozenset[str] = frozenset()
    layer_ids: frozenset[str] = frozenset()
    layer_types: frozenset[str] = frozenset()
    performer_ids: frozenset[str] = frozenset()
    actor_ids: frozenset[str] = frozenset()
    tools: frozenset[str] = frozenset()
    event_types: frozenset[str] = frozenset()
    statuses: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("Report timestamps must be timezone-aware")
        if self.end < self.start:
            raise ValueError("Report end must not precede its start")

    def matches(self, item: ActivityRecord) -> bool:
        if not self.start <= item.recorded_at <= self.end:
            return False
        comparisons = (
            (self.project_ids, item.project_id),
            (self.layer_ids, item.layer_id),
            (self.layer_types, item.layer_type),
            (self.performer_ids, item.performer_id),
            (self.actor_ids, item.actor_id),
            (self.tools, item.tool),
            (self.event_types, item.event_type),
            (self.statuses, item.status),
        )
        return all(not accepted or value in accepted for accepted, value in comparisons)


@dataclass(frozen=True, slots=True)
class ReportMetrics:
    values: Mapping[str, int | float]
    by_project: Mapping[str, Mapping[str, int]]
    by_performer: Mapping[str, Mapping[str, int]]
    event_ids_by_metric: Mapping[str, tuple[str, ...]]


# Dotted values are the stable activity taxonomy. CamelCase values are the
# corresponding domain events already emitted (or reserved) by application use
# cases. Generic events receive additional payload-aware classification below.
_METRIC_EVENT_TYPES: dict[str, frozenset[str]] = {
    "created_artifact_versions": frozenset(
        {
            "ArtifactVersionCreated",
            "ArtifactVersionImported",
            "ArtifactImported",
            "artifact.version.created",
            "artifact.vector.created",
            "artifact.imported",
        }
    ),
    "vectorization_operations": frozenset(
        {"VectorizationCompleted", "FrameVectorized", "artifact.vector.created", "frame.vectorized"}
    ),
    "binary_representations": frozenset(
        {"BinaryRepresentationCreated", "representation.binary.created"}
    ),
    "imported_files": frozenset({"ArtifactImported", "ArtifactVersionImported", "artifact.imported"}),
    "work_issued": frozenset(
        {"WorkBatchIssued", "ReviewBatchIssued", "work_batch.issued", "review_batch.issued"}
    ),
    "returned_changed": frozenset({"ReviewReturnedChanged", "review.returned_changed"}),
    "returned_unchanged": frozenset({"ReviewReturnedUnchanged", "review.returned_unchanged"}),
    "returned_missing": frozenset({"ReviewReturnedMissing", "review.returned_missing"}),
    "accepted": frozenset({"ReviewAccepted", "ReviewCandidateAccepted", "review.accepted"}),
    "changes_requested": frozenset({"ReviewChangesRequested", "review.changes_requested"}),
    "plugin_failures": frozenset({"PluginJobFailed", "plugin_job.failed"}),
    "incomplete_jobs": frozenset(
        {"PluginJobPartial", "PluginJobRecoveryRequired", "plugin_job.partial", "plugin_job.recovery_required"}
    ),
}


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _semantic_values(item: ActivityRecord) -> frozenset[str]:
    fields = (
        "activity_type",
        "operation",
        "origin",
        "source",
        "kind",
        "artifact_kind",
        "representation_kind",
        "pixel_format",
    )
    return frozenset(str(item.payload[key]).strip().casefold() for key in fields if item.payload.get(key) is not None)


def _is_vector_artifact(item: ActivityRecord) -> bool:
    semantic = _semantic_values(item)
    if semantic.intersection({"vector", "vectorized", "vectorization", "contour"}):
        return True
    media_type = str(item.payload.get("media_type", "")).casefold()
    filename = str(item.payload.get("filename", "")).casefold()
    return "cif" in media_type or filename.endswith(".cif")


def _is_binary_representation(item: ActivityRecord) -> bool:
    semantic = _semantic_values(item)
    if bool(item.payload.get("binary")):
        return True
    return bool(
        semantic.intersection(
            {"binary", "binary_image", "binary-image", "neuralimage", "1-bit", "0/255", "0,255"}
        )
    )


def _is_import(item: ActivityRecord) -> bool:
    semantic = _semantic_values(item)
    return any(value in {"import", "imported", "managed_copy", "external_link"} or value.startswith("import:") for value in semantic)


def _metric_categories(item: ActivityRecord) -> frozenset[str]:
    categories = {metric for metric, names in _METRIC_EVENT_TYPES.items() if item.event_type in names}
    if item.event_type in {"ArtifactVersionCreated", "artifact.version.created"}:
        if _is_vector_artifact(item):
            categories.add("vectorization_operations")
        if _is_import(item):
            categories.add("imported_files")
    if item.event_type in {"RepresentationCreated", "representation.created"} and _is_binary_representation(item):
        categories.add("binary_representations")
    return frozenset(categories)


class _MetricsAccumulator:
    """One-pass aggregate builder; only required drill-down IDs are retained."""

    def __init__(self, filters: ReportFilters) -> None:
        self.filters = filters
        self.counts: Counter[str] = Counter()
        self.drill_down: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
        self.first_vectorization: dict[tuple[str, str, str], ActivityRecord] = {}
        self.project_counters: dict[str, Counter[str]] = defaultdict(Counter)
        self.performer_counters: dict[str, Counter[str]] = defaultdict(Counter)
        self.imported_bytes = 0
        self.issued = 0
        self.accepted = 0
        self.rework = 0
        self.overdue = 0
        self.turnaround_total = 0.0
        self.turnaround_count = 0

    def add(self, item: ActivityRecord) -> bool:
        categories = _metric_categories(item)
        if (
            "vectorization_operations" in categories
            and item.frame_id is not None
            and item.layer_id is not None
        ):
            key = (item.project_id, item.layer_id, item.frame_id)
            existing = self.first_vectorization.get(key)
            if existing is None or (item.recorded_at, item.event_id) < (existing.recorded_at, existing.event_id):
                self.first_vectorization[key] = item

        if not self.filters.matches(item):
            return False

        for metric in categories:
            self.counts[metric] += 1
            self.drill_down[metric].append((item.recorded_at, item.event_id))
        if "imported_files" in categories:
            self.imported_bytes += max(0, item.bytes_count)
        self.issued += "work_issued" in categories
        self.accepted += "accepted" in categories
        self.rework += "changes_requested" in categories
        self.overdue += item.status == "overdue"
        if categories.intersection({"returned_changed", "returned_unchanged", "returned_missing"}):
            duration = item.payload.get("turnaround_seconds")
            if isinstance(duration, (int, float)) and not isinstance(duration, bool):
                self.turnaround_total += float(duration)
                self.turnaround_count += 1

        self.project_counters[item.project_id][item.event_type] += 1
        if item.performer_id:
            self.performer_counters[item.performer_id][item.event_type] += 1
        return True

    def finish(self) -> ReportMetrics:
        values: dict[str, int | float] = {
            metric: self.counts[metric] for metric in _METRIC_EVENT_TYPES
        }
        drill_down: dict[str, tuple[str, ...]] = {
            metric: tuple(event_id for _, event_id in sorted(self.drill_down[metric]))
            for metric in _METRIC_EVENT_TYPES
        }

        # "First" is determined over the complete supplied history, not merely
        # over the report period.  The earliest event is then tested against all
        # filters, including the selected period/performer/tool.
        first_in_scope = sorted(
            (item for item in self.first_vectorization.values() if self.filters.matches(item)),
            key=lambda item: (item.recorded_at, item.event_id),
        )
        values["unique_first_vectorized_frames"] = len(first_in_scope)
        drill_down["unique_first_vectorized_frames"] = tuple(item.event_id for item in first_in_scope)
        values["imported_bytes"] = self.imported_bytes
        values["backlog"] = max(0, self.issued - self.accepted)
        values["rework_rate"] = (
            0.0 if not self.accepted and not self.rework else self.rework / (self.accepted + self.rework)
        )
        values["overdue"] = self.overdue
        values["average_turnaround_seconds"] = (
            0.0 if not self.turnaround_count else self.turnaround_total / self.turnaround_count
        )
        return ReportMetrics(
            values=values,
            by_project={key: dict(value) for key, value in self.project_counters.items()},
            by_performer={key: dict(value) for key, value in self.performer_counters.items()},
            event_ids_by_metric=drill_down,
        )


def _safe_tabular_text(value: object) -> object:
    """Prevent spreadsheet formula interpretation in CSV and XLSX exports."""

    if not isinstance(value, str) or not value:
        return value
    stripped = value.lstrip(" \t\r\n")
    if value[0] in "\t\r\n" or (stripped and stripped[0] in "=+-@"):
        return "'" + value
    return value


class ReportService:
    def iter_selected(self, records: Iterable[ActivityRecord], filters: ReportFilters) -> Iterator[ActivityRecord]:
        """Filter without buffering; callers supply canonical event order."""

        return (item for item in records if filters.matches(item))

    def select(self, records: Iterable[ActivityRecord], filters: ReportFilters) -> tuple[ActivityRecord, ...]:
        return tuple(sorted(self.iter_selected(records, filters), key=lambda item: (item.recorded_at, item.event_id)))

    def aggregate(self, records: Iterable[ActivityRecord], filters: ReportFilters) -> ReportMetrics:
        accumulator = _MetricsAccumulator(filters)
        for item in records:
            accumulator.add(item)
        return accumulator.finish()

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
    ) -> Path:
        try:
            from openpyxl import Workbook
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install Kraken with the 'reports' extra to export XLSX") from exc

        # write_only avoids retaining worksheet cells.  For an ordered database
        # cursor, assume_sorted also avoids retaining report rows themselves.
        source: Iterable[ActivityRecord]
        if assume_sorted:
            source = records
        else:
            source = sorted(records, key=lambda item: (item.recorded_at, item.event_id))
        workbook = Workbook(write_only=True)
        summary = workbook.create_sheet("Summary")
        projects = workbook.create_sheet("Projects")
        layers_sheet = workbook.create_sheet("Layers")
        performers = workbook.create_sheet("Performers")
        tools_sheet = workbook.create_sheet("Tools")
        events = workbook.create_sheet("Events")
        events.append(
            ("Event ID", "Recorded at", "Type", "Project", "Layer", "Frame", "Actor", "Performer", "Tool", "Status")
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
                        item.frame_id,
                        item.actor_id,
                        item.performer_id,
                        item.tool,
                        item.status,
                    )
                )
            )

        metrics = accumulator.finish()
        summary.append(("Metric", "Value"))
        for key, value in sorted(metrics.values.items()):
            summary.append((_safe_tabular_text(key), value))
        self._write_counter_sheet(projects, "Project", metrics.by_project)
        self._write_counter_sheet(layers_sheet, "Layer", layer_counters)
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


__all__ = ["ActivityRecord", "ReportFilters", "ReportMetrics", "ReportService"]
