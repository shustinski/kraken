"""composition: synchronize plugin jobs."""
from __future__ import annotations
from kraken_core.job_recovery import progress_fraction
from kraken_manager.application.dto import (
    CommandContext,
    RetryPluginJobCommand,
    CancelPluginJobCommand,
    SynchronizePluginJobCommand,
)
from kraken_manager.application.plugin_jobs import (
    CancelPluginJobHandler,
    RetryPluginJobHandler,
)
from kraken_manager.domain.identity import (
    Principal,
)


class SynchronizePluginJobsMixin:
    def cancel_plugin_job(
        self,
        *,
        principal: Principal,
        gateway: object,
        job: object,
        idempotency_key: str,
    ):
        return CancelPluginJobHandler(
            self._uow(str(job.project_id)),
            self.profiles,
            self.clock,
            gateway,
        )(
            CancelPluginJobCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=job.project_id,
                job_id=job.id,
                expected_revision=job.revision,
            )
        )

    def retry_plugin_job(
        self,
        *,
        principal: Principal,
        gateway: object,
        job: object,
        idempotency_key: str,
    ):
        return RetryPluginJobHandler(
            self._uow(str(job.project_id)),
            self.profiles,
            self.clock,
            gateway,
        )(
            RetryPluginJobCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=job.project_id,
                job_id=job.id,
                expected_revision=job.revision,
            )
        )

    def synchronize_plugin_jobs(
        self,
        *,
        principal: Principal,
        gateway: object,
    ) -> tuple[object, ...]:
        synchronized: list[object] = []
        for job in self.plugin_jobs():
            if job.state.value in {"succeeded", "failed", "cancelled"}:
                synchronized.append(job)
                continue
            try:
                agent = gateway.get_job(job.id)
            except Exception:
                target = "recovery_required"
                error = "Kraken Agent недоступен или потерял состояние задания"
                progress = job.progress
            else:
                target = str(agent.get("state", job.state.value))
                error = (
                    None
                    if agent.get("error") in {None, ""}
                    else str(agent.get("error"))
                )
                progress = max(job.progress, progress_fraction(agent.get("progress"), job.progress))
            if target == job.state.value and progress == job.progress:
                synchronized.append(job)
                continue
            updated = self._commands(str(job.project_id)).execute(
                SynchronizePluginJobCommand(
                    context=CommandContext(
                        actor=principal,
                        idempotency_key=f"agent-sync:{job.id}:{job.revision}:{target}:{progress}",
                    ),
                    project_id=job.project_id,
                    job_id=job.id,
                    expected_revision=job.revision,
                    state=target,
                    progress=progress,
                    error=error,
                )
            )
            synchronized.append(updated)
        return tuple(synchronized)
