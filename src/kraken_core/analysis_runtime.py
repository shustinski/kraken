"""Validate worker receipts independently of any local/server projection."""
from .analysis_run_protocol import AnalysisRuntimeIdentity


def measured_runtime(results, *, require_all=True):
    values = [result.get("worker_runtime") for result in results]
    measured = [AnalysisRuntimeIdentity.from_payload(value) for value in values if value is not None]
    if measured and any(value != measured[0] for value in measured[1:]):
        raise ValueError("Analysis partitions report incompatible worker runtimes")
    if not measured or (require_all and any(value is None for value in values)):
        return None
    return measured[0]


def validate_worker_runtime(manifest, result, *, pinned=None):
    actual = result.worker_runtime
    if actual is None:
        if manifest.require_worker_runtime or manifest.expected_runtime is not None or pinned is not None:
            raise ValueError("Worker runtime receipt is missing; update the analysis Worker")
        return
    if manifest.require_worker_runtime:
        build = actual.engine_build.removeprefix("sha256:")
        if not actual.engine_build.startswith("sha256:") or len(build) != 64 or any(c not in "0123456789abcdef" for c in build):
            raise ValueError("Worker runtime requires a measured computation-code SHA-256")
    # A bounded explicit runtime refusal may report its actual environment.
    if result.bundle is None and result.outcome.value in {"failed", "cancelled"}:
        return
    if manifest.expected_runtime is not None and actual != manifest.expected_runtime:
        raise ValueError("Worker runtime differs from the pinned requested runtime")
    if pinned is not None and actual != pinned:
        raise ValueError("Analysis partitions report incompatible worker runtimes")
