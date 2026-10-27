"""Bounded host observations without loading scientific or GPU runtimes.

Measurements describe the whole host, including unrelated processes. They are
not job accounting, queue reservations, or physical resource enforcement.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import socket
import subprocess
from datetime import UTC, datetime


def _number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def sample_host(*, gpu_timeout: float = 1.5, psutil_module=None, run=None):
    """Return real observations or explicit unknowns, never synthetic zeroes."""
    result = {
        "schema": "kraken.host-telemetry.v1",
        "host_id": hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()[:32],
        "observed_at": datetime.now(UTC).isoformat(),
        "scope": "whole_host", "cpu_percent": None,
        "memory_total_bytes": None, "memory_available_bytes": None,
        "gpus": None, "errors": [],
        "sources": {"cpu_memory": "psutil", "gpu": "nvidia-smi"},
    }
    try:
        if psutil_module is None:
            import psutil as psutil_module
        # A blocking interval avoids psutil's meaningless first nonblocking 0.
        result["cpu_percent"] = _number(psutil_module.cpu_percent(interval=0.1))
        memory = psutil_module.virtual_memory()
        result["memory_total_bytes"] = _number(memory.total)
        result["memory_available_bytes"] = _number(memory.available)
    except Exception as exc:  # noqa: BLE001 - measurement failure is an unknown observation
        result["errors"].append(f"cpu_memory:{type(exc).__name__}")
    try:
        completed = (run or subprocess.run)(
            ["nvidia-smi", "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=max(0.1, min(float(gpu_timeout), 5.0)),
            check=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        gpus = []
        for row in csv.reader(io.StringIO(completed.stdout)):
            if len(row) != 6:
                raise ValueError("Unexpected nvidia-smi columns")
            identifier, name, utilization, used, total, power = (value.strip() for value in row)
            gpus.append({"uuid": identifier, "name": name,
                         "utilization_percent": _number(utilization),
                         "memory_used_mib": _number(used), "memory_total_mib": _number(total),
                         "power_watts": _number(power)})
        if not gpus:
            raise ValueError("No GPU observations")
        result["gpus"] = gpus
    except Exception as exc:  # noqa: BLE001 - absent driver/sensor/timeout is unknown, not zero GPUs
        result["errors"].append(f"gpu:{type(exc).__name__}")
    result["observed_at"] = datetime.now(UTC).isoformat()
    return result
