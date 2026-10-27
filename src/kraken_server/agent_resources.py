"""Explicit admission policy for the existing PostgreSQL Agent lease queue.

These are logical server-queue reservations, not per-host hardware limits and
not OS CPU/RAM/GPU enforcement. Missing legacy requests reserve the entire
configured budget rather than silently becoming zero-cost jobs.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
import math
import re
from datetime import datetime

from kraken_manager.domain.pipeline import PipelineResources

RESOURCE_PARAMETER = "execution_resources"


def validate_telemetry(value):
    """Validate authenticated but untrusted host sensor reports."""
    if not isinstance(value, Mapping) or value.get("schema") != "kraken.host-telemetry.v1":
        raise ValueError("Unsupported host telemetry schema")
    if value.get("scope") != "whole_host" or not re.fullmatch(r"[0-9a-f]{32}", str(value.get("host_id", ""))):
        raise ValueError("Host telemetry requires a stable anonymous host ID and whole_host scope")
    observed = datetime.fromisoformat(str(value.get("observed_at", "")))
    if observed.tzinfo is None:
        raise ValueError("Telemetry observation time requires a timezone")

    def number(item, maximum=None):
        if item is None:
            return None
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0:
            raise ValueError("Telemetry metrics must be finite non-negative numbers or null")
        if maximum is not None and item > maximum:
            raise ValueError("Telemetry percentage exceeds 100")
        return item

    result = dict(schema=value["schema"], scope="whole_host", host_id=value["host_id"],
                  observed_at=observed.isoformat(), cpu_percent=number(value.get("cpu_percent"), 100),
                  memory_total_bytes=number(value.get("memory_total_bytes")),
                  memory_available_bytes=number(value.get("memory_available_bytes")))
    gpus = value.get("gpus")
    if gpus is not None:
        if not isinstance(gpus, list) or len(gpus) > 64:
            raise ValueError("Host telemetry supports at most 64 GPU observations")
        if any(not isinstance(gpu, Mapping) for gpu in gpus):
            raise ValueError("GPU telemetry requires metric objects")
        gpus = [{"uuid": str(gpu.get("uuid", ""))[:200], "name": str(gpu.get("name", ""))[:200],
                 **{key: number(gpu.get(key), 100 if key == "utilization_percent" else None)
                    for key in ("utilization_percent", "memory_used_mib", "memory_total_mib", "power_watts")}}
                for gpu in gpus if isinstance(gpu, Mapping)]
    result["gpus"] = gpus
    errors = value.get("errors", [])
    if not isinstance(errors, list) or len(errors) > 8:
        raise ValueError("Invalid telemetry errors")
    result["errors"] = [str(item)[:200] for item in errors]
    result["sources"] = {"cpu_memory": "psutil", "gpu": "nvidia-smi"}
    return result


def requested_resources(parameters):
    if not isinstance(parameters, Mapping):
        raise ValueError("Job parameters must be a mapping")
    value = parameters.get(RESOURCE_PARAMETER)
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"cpu", "gpu", "memory_mb"}:
        raise ValueError("execution_resources requires exactly cpu, gpu and memory_mb")
    return PipelineResources(**value)


@dataclass(frozen=True, slots=True)
class QueueResourceBudget:
    max_active_jobs: int
    capacity: PipelineResources

    def __post_init__(self):
        if isinstance(self.max_active_jobs, bool) or not isinstance(self.max_active_jobs, int) or self.max_active_jobs < 1:
            raise ValueError("Queue max_active_jobs must be a positive integer")
        if not isinstance(self.capacity, PipelineResources):
            raise ValueError("Queue capacity must be PipelineResources")

    @classmethod
    def from_environment(cls, environment=None):
        env = os.environ if environment is None else environment
        keys = ("KRAKEN_QUEUE_MAX_JOBS", "KRAKEN_QUEUE_CPU", "KRAKEN_QUEUE_GPU", "KRAKEN_QUEUE_MEMORY_MB")
        present = [key for key in keys if str(env.get(key, "")).strip()]
        if not present:
            return None
        if len(present) != len(keys):
            raise ValueError("Declare all KRAKEN_QUEUE_MAX_JOBS/CPU/GPU/MEMORY_MB dimensions or leave the budget unconfigured")
        values = []
        for key in keys:
            text = str(env[key]).strip()
            if not text.isdecimal():
                raise ValueError(f"{key} must be an integer")
            values.append(int(text))
        return cls(values[0], PipelineResources(*values[1:]))

    def to_dict(self):
        return {"max_active_jobs": self.max_active_jobs, **self.capacity.to_dict()}


class QueueAdmission:
    def __init__(self, jobs, budget):
        self.jobs, self.budget = jobs, budget

    def lock(self, connection):
        if self.budget is not None:
            import sqlalchemy as sa
            # Transaction-scoped and schema-scoped: all server processes leasing
            # this queue serialize admission, while isolated test schemas remain
            # independent. No second job queue or reservation table is created.
            connection.execute(sa.text("SELECT pg_advisory_xact_lock(hashtextextended(current_schema() || :queue_suffix, 0))"),
                               {"queue_suffix": ":kraken-agent-admission-v1"})

    def cost(self, payload):
        resources = requested_resources(payload.get("parameters", {}))
        return resources if resources is not None else (self.budget.capacity if self.budget else None)

    def usage(self, connection, now):
        import sqlalchemy as sa
        payloads = connection.execute(sa.select(self.jobs.c.payload).where(
            self.jobs.c.state == "leased", self.jobs.c.lease_until > now,
        )).scalars().all()
        used = dict(cpu=0, gpu=0, memory_mb=0)
        legacy = 0
        for payload in payloads:
            try:
                declared = requested_resources(payload.get("parameters", {}))
            except ValueError:
                # Historical malformed requests already executing conservatively
                # consume the complete budget; they are never zero-cost.
                declared = None
            legacy += declared is None
            cost = declared or (self.budget.capacity if self.budget else None)
            if cost is not None:
                for key in used:
                    used[key] += getattr(cost, key)
        return len(payloads), used, legacy

    def decision(self, payload, active, used):
        # Validate even when admission is explicitly disabled.
        cost = self.cost(payload)
        if self.budget is None:
            return None
        if active >= self.budget.max_active_jobs:
            return "admission:max_active_jobs"
        if not cost.fits(self.budget.capacity):
            return "admission:request_exceeds_budget"
        if any(used[key] + getattr(cost, key) > getattr(self.budget.capacity, key) for key in used):
            return "admission:capacity_reserved"
        return None

    def status(self, connection, now):
        active, used, legacy = self.usage(connection, now)
        return {"scope": "global_server_queue", "admission_enforced": self.budget is not None,
                "admission_ready": self.budget is not None,
                "state": "configured" if self.budget else "unenforced",
                "limits": self.budget.to_dict() if self.budget else None,
                "active_leases": active, "declared_reserved": used if self.budget else None,
                "legacy_unknown_requests": legacy,
                "legacy_policy": "reserve_entire_configured_budget",
                "os_enforced": False, "per_host_capacity_enforced": False}
