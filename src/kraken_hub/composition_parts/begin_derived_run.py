"""composition: begin derived run."""
from __future__ import annotations
from kraken_manager.workspace import (
    DerivedRun,
    DerivedRunKind,
)
from kraken_manager.domain.common import (
    LayerId,
    ProjectId,
)
from kraken_manager.domain.identity import (
    Principal,
)


class BeginDerivedRunMixin:
    def begin_derived_run(
        self,
        *,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        layer_name: str,
        kind: DerivedRunKind,
        plugin_id: str,
        operation: str,
        principal: Principal | None = None,
    ) -> DerivedRun:
        run = self.workspace_files.begin_run(
            project_id=str(project_id),
            layer_id=str(layer_id),
            layer_name=layer_name,
            kind=kind,
            plugin_id=plugin_id,
            operation=operation,
        )
        if principal is not None:
            self._record_workspace_event(
                principal=principal,
                project_id=project_id,
                stream_id=f"derived-run:{run.run_id}",
                event_type="DerivedRunStartedV1",
                payload={
                    "run_id": run.run_id,
                    "layer_id": run.layer_id,
                    "kind": run.kind.value,
                    "state": run.state.value,
                    "path": run.path,
                    "plugin_id": run.plugin_id,
                    "operation": run.operation,
                    "created_at": run.created_at,
                },
                idempotency_key=f"derived-run:{run.run_id}:start",
            )
        return run

    def list_derived_runs(
        self,
        project_id: ProjectId | str,
        layer_id: LayerId | str = "",
    ) -> tuple[DerivedRun, ...]:
        return self.workspace_registry.list_runs(
            str(project_id),
            str(layer_id) if layer_id else "",
        )
