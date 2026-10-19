"""persistent_services: start desktop job."""
from __future__ import annotations
from kraken_manager.domain.common import (
    PluginJobId,
)
from kraken_server.services import CommandContext, NotFoundError

from kraken_server.persistent_services_parts.layer_source_scan import (
    SystemClock,
    _project_id,
)


class StartDesktopJobMixin:
    def start_desktop_job(self, project_id: str, job_id: str, context: CommandContext) -> None:
        from kraken_manager.application.plugin_jobs import SynchronizePluginJobHandler
        from kraken_manager.application.dto import SynchronizePluginJobCommand

        job = self.projections.get_plugin_job(PluginJobId(job_id))
        if job is None:
            raise NotFoundError("Job was not found")
        state = "running" if job.capability in {"frames.binary-segment.v2", "frames.confidence.v1"} else "waiting_for_user"
        try:
            SynchronizePluginJobHandler(self.uow_factory, self.profiles, SystemClock())(
                SynchronizePluginJobCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    job_id=job.id,
                    expected_revision=job.revision,
                    state=state,
                    progress=job.progress,
                )
            )
        except Exception as exc:
            self._translate_lifecycle_error(exc)
