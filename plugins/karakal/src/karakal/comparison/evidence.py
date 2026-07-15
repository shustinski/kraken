"""Evidence-provider placeholders for future external explanations."""
from __future__ import annotations

from .models import MetricValue


def evidence_metrics(provider_version: str | None = None) -> tuple[MetricValue, ...]:
    version = str(provider_version or "none")
    return (
        MetricValue(
            name="evidence_provider_available",
            value=0 if version == "none" else 1,
            group="evidence",
            description="Whether external evidence overlays were attached.",
            valid=True,
        ),
    )
