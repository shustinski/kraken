"""persistent_services: statistics."""
from __future__ import annotations
from datetime import UTC, datetime
from typing import Any
from kraken_manager.infrastructure.reports import (
    ActivityRecord,
    ReportFilters,
    ReportGranularity,
    ReportService,
)
from kraken_manager.infrastructure.filesystem._codec import encode_model
from kraken_server.services import NotFoundError

from kraken_server.persistent_services_parts.layer_source_scan import _project_id


class StatisticsMixin:
    def statistics(
        self,
        project_id: str,
        *,
        start: datetime,
        end: datetime,
        timezone: Any = UTC,
    ) -> dict[str, Any]:
        project = self.projections.get_project(_project_id(project_id))
        if project is None:
            raise NotFoundError("Project was not found")
        records: list[ActivityRecord] = []
        after = 0
        while True:
            page = self.events.list_project_events(
                project.id, after_position=after, limit=500
            )
            if not page:
                break
            records.extend(ActivityRecord.from_event(event) for _position, event in page)
            after = page[-1][0]
        filters = ReportFilters(start, end, project_ids=frozenset((str(project.id),)))
        reports = ReportService()
        metrics = reports.aggregate(records, filters)
        return {
            "metrics": encode_model(metrics),
            "series": {
                granularity.value: encode_model(
                    reports.aggregate_series(
                        records, filters, granularity, timezone=timezone
                    )
                )
                for granularity in ReportGranularity
            },
        }
