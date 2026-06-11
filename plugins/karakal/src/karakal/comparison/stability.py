"""Stability metric placeholders for temporal extensions."""
from __future__ import annotations

from .models import MetricValue


def stability_metrics() -> tuple[MetricValue, ...]:
    return (
        MetricValue(
            name="temporal_stability_available",
            value=0,
            group="stability",
            description="Temporal stability requires neighboring frame context and is reserved for the existing sequence layer.",
            valid=True,
        ),
    )
