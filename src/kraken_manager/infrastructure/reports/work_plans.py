"""CSV/XLSX adapters for versioned engineering-plan reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

# key, Russian label, English label, unit, formula/interpretation
WORK_PLAN_METRICS = (
    (
        "total_items",
        "Все работы и этапы",
        "All work items and stages",
        "count",
        "Distinct filtered WorkItems including explicitly managed stages.",
    ),
    (
        "completed_plans",
        "Формально завершённые планы",
        "Formally completed plans",
        "count",
        "Plans in the completed state; leaf readiness alone does not complete a plan.",
    ),
    ("leaf_items", "Конечные работы", "Leaf work items", "count", "Distinct filtered WorkItems without children."),
    (
        "accepted_items",
        "Принятые работы",
        "Accepted work items",
        "count",
        "Leaf work items currently formally accepted.",
    ),
    (
        "readiness_fraction",
        "Готовность работ",
        "Work readiness",
        "fraction",
        "Accepted leaves / non-cancelled leaves; unknown for an empty denominator.",
    ),
    (
        "reported_effort_hours",
        "Зарегистрированные трудозатраты",
        "Reported effort",
        "hours",
        "Sum of explicitly reported cumulative effort on all selected items and stages; never elapsed-time inference.",
    ),
    (
        "mean_lead_hours",
        "Среднее время от создания до приёмки",
        "Mean lead time",
        "hours",
        "Mean final acceptance event time minus first creation event time for currently accepted leaves.",
    ),
    (
        "mean_cycle_hours",
        "Среднее время от запуска до приёмки",
        "Mean cycle time",
        "hours",
        "Mean final acceptance event time minus first start event time, including blocking and rework.",
    ),
    (
        "rework_fraction",
        "Доля работ с повторным выполнением",
        "Reworked work fraction",
        "fraction",
        "Started leaves with attempt > 1 / started non-cancelled leaves.",
    ),
    (
        "rework_attempts",
        "Повторные попытки",
        "Rework attempts",
        "count",
        "Sum max(attempt - 1, 0) over leaf work items.",
    ),
    (
        "acceptance_events_in_period",
        "События приёмки за период",
        "Acceptance events in period",
        "count",
        "Transitions into accepted, including acceptance after rework; snapshot updates are not new acceptance.",
    ),
    (
        "acceptances_per_calendar_day",
        "Приёмок за календарный день",
        "Acceptances per calendar day",
        "count/day",
        "Acceptance events / explicit report period duration in calendar days.",
    ),
    (
        "overdue_items",
        "Просроченные работы",
        "Overdue work items",
        "count",
        "Open leaves whose explicit planned finish precedes as_of.",
    ),
    (
        "accepted_frame_scope_count",
        "Кадры в области принятых работ",
        "Frames covered by accepted work",
        "layer-frame pairs",
        "Distinct layer/frame scope pairs; does not substitute for artifact quality approval.",
    ),
    (
        "measured_compute_utilization",
        "Измеренная загрузка вычислителей",
        "Measured compute utilization",
        "fraction",
        "Unknown without resource telemetry; running job counts are not utilization.",
    ),
)


def _cell(value):
    from .service import _safe_tabular_text

    if isinstance(value, (list, tuple, dict)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return _safe_tabular_text(value)


def rows(report):
    base = {"project_id": report["project_id"], "as_of": report["as_of"]}
    definitions = {r[0]: r for r in WORK_PLAN_METRICS}
    for name, value in report["summary"].items():
        yield base | {
            "section": "summary",
            "metric": name,
            "value": value,
            "unit": definitions.get(name, (None,) * 4)[3],
        }
    for name, value in report["forecast"].items():
        yield base | {"section": "forecast", "metric": name, "value": value}
    for plan_id, revision in report["source_plan_revisions"].items():
        yield base | {"section": "metadata", "plan_id": plan_id, "metric": "source_plan_revision", "value": revision}
    for key in ("start", "end", "complete_event_history"):
        yield base | {"section": "metadata", "metric": key, "value": report[key]}
    for key, values in report["filters"].items():
        for value in values:
            yield base | {"section": "metadata", "metric": f"filter:{key}", "value": value}
    for event_id in report["source_event_ids"]:
        yield base | {"section": "provenance", "event_id": event_id}
    for item in report.get("compute_workload", ()):
        for name in (
            "active_operations",
            "reserved",
            "declared_capacity",
            "reservation_fraction",
            "measured_utilization",
        ):
            yield base | {
                "section": "compute",
                "metric": f"{item['layer_id']}:{name}",
                "value": item[name],
                "event_id": item["source_event_id"],
            }
    for item in report["items"]:
        identity = {key: item[key] for key in ("plan_id", "plan_revision", "item_id", "title", "state", "performer_id")}
        for name in (
            "layer_ids",
            "frame_ids",
            "hierarchy",
            "is_leaf",
            "planned_effort_hours",
            "reported_effort_hours",
            "lead_hours",
            "cycle_hours",
            "remaining_estimated_hours",
            "planned_start",
            "planned_finish",
            "accepted_at",
            "blocking_reason",
        ):
            values = item[name] if isinstance(item[name], (tuple, list)) else [item[name]]
            for value in values:
                yield (
                    base
                    | identity
                    | {
                        "section": "tasks",
                        "metric": name,
                        "value": value,
                        "unit": "hours" if name.endswith("hours") else None,
                    }
                )
        for criterion in item["criteria"]:
            for name in ("description", "required", "satisfied", "reviewed_by", "evidence_version_ids"):
                values = criterion[name] if isinstance(criterion[name], list) else [criterion[name]]
                for value in values:
                    yield (
                        base
                        | identity
                        | {"section": "acceptance", "metric": f"{criterion['id']}:{name}", "value": value}
                    )
        for change in item["changes"]:
            yield (
                base
                | identity
                | {
                    "section": "history",
                    "event_id": change["event_id"],
                    "metric": "state",
                    "value": change["state"],
                    "recorded_at": change["at"],
                    "plan_revision": change["plan_revision"],
                    "actor_id": change.get("actor_id"),
                    "actor_name": change.get("actor_name"),
                    "operation": change.get("operation"),
                }
            )
    for item in report["workload"]:
        for name, value in item.items():
            if name == "performer_id":
                continue
            values = value if isinstance(value, list) else [value]
            for entry in values:
                yield base | {
                    "section": "workload",
                    "performer_id": item["performer_id"],
                    "metric": name,
                    "value": entry,
                }
    for item in report["baseline_variance"]:
        for name in (
            "scope_change",
            "planned_effort_delta_hours",
            "planned_finish_delta_hours",
            "acceptance_finish_variance_hours",
            "baseline_plan_effort_hours",
            "current_plan_effort_hours",
        ):
            yield base | {
                "section": "baselines",
                "plan_id": item["plan_id"],
                "baseline_id": item["baseline_id"],
                "item_id": item["item_id"],
                "title": item["title"],
                "metric": name,
                "value": item[name],
                "unit": "hours" if name.endswith("hours") else None,
            }


COLUMNS = (
    "section",
    "project_id",
    "as_of",
    "plan_id",
    "plan_revision",
    "baseline_id",
    "item_id",
    "title",
    "state",
    "performer_id",
    "metric",
    "value",
    "unit",
    "event_id",
    "recorded_at",
    "actor_id",
    "actor_name",
    "operation",
)


def write_csv(destination, report):
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in rows(report):
            writer.writerow(_cell(row.get(column)) for column in COLUMNS)
    return output


def write_xlsx(destination, report):
    from openpyxl import Workbook

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook(write_only=True)
    sheets = {}
    for row in rows(report):
        section = row["section"]
        if section not in sheets:
            sheet = workbook.create_sheet(section.title())
            sheet.freeze_panes = "A2"
            sheet.append(COLUMNS)
            sheets[section] = sheet
        sheets[section].append([_cell(row.get(column)) for column in COLUMNS])
    workbook.save(output)
    return output
