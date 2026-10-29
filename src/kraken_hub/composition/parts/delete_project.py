"""composition: delete project."""
from __future__ import annotations
import shutil
from collections.abc import Mapping
from pathlib import Path
from kraken_core.frame_matrix import StoreNamespace
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.workspace import (
    WorkspaceValidationError,
)
from kraken_manager.domain.identity import (
    Permission,
    Principal,
)
from kraken_manager.domain.project import (
    Project,
)

from kraken_hub.composition.parts.frame_cell_snapshot import ProjectDeletionResult


class DeleteProjectMixin:
    def delete_project(
        self,
        *,
        principal: Principal,
        project: Project,
        confirmation_name: str,
    ) -> ProjectDeletionResult:
        """Forget a local project while preserving both workspace data trees."""

        current = self.get_project(project.id)
        if current is None:
            raise WorkspaceValidationError("project is no longer available")
        if confirmation_name != current.name:
            raise WorkspaceValidationError("project name confirmation does not match")
        AuthorizationPolicy().require(
            principal=principal,
            storage=self.profile,
            permission=Permission.ARCHIVE_PROJECT,
            roles=self.project_roles(current.id, principal.id),
        )

        active_runs = [
            run
            for run in self.list_derived_runs(current.id)
            if run.state.value == "running"
        ]
        latest_jobs: dict[str, str] = {}
        for event in self.history(current.id):
            payload = event.payload
            job = payload.get("job", {})
            if not isinstance(job, Mapping):
                continue
            job_id = str(payload.get("plugin_job_id", job.get("id", "")))
            if job_id:
                latest_jobs[job_id] = str(job.get("state", ""))
        active_job_states = {
            state
            for state in latest_jobs.values()
            if state and state not in {"succeeded", "failed", "cancelled"}
        }
        if active_runs or active_job_states:
            raise WorkspaceValidationError(
                "project has an active plugin run; finish or cancel it before deletion"
            )

        identifier = str(current.id)
        projects_root = (self.catalog_root / "projects").resolve()
        project_cache = (projects_root / identifier).resolve()
        binding = self.project_workspace(current.id)
        if binding is not None:
            for workspace_path in (
                binding.source_project_dir,
                binding.derived_project_dir,
            ):
                resolved_workspace = Path(workspace_path).resolve()
                if resolved_workspace == project_cache or resolved_workspace.is_relative_to(project_cache):
                    raise WorkspaceValidationError(
                        "Папка данных проекта находится внутри внутреннего кэша Kraken; "
                        "безопасное удаление невозможно."
                    )

        thumbnail_root = (self.data_dir / "cache" / "frame-thumbnails").resolve()
        namespace = StoreNamespace(
            plugin="matrix",
            project=identifier,
            generation="v1",
        )
        thumbnail_cache = (thumbnail_root / namespace.digest()).resolve()
        thumbnail_removed = self._remove_cache_tree(thumbnail_root, thumbnail_cache)

        staging_removed = 0
        staging_root = (self.data_dir / "agent-staging").resolve()
        if staging_root.is_dir():
            for entry in tuple(staging_root.iterdir()):
                if (
                    not entry.is_dir()
                    or entry.is_symlink()
                    or entry.resolve().parent != staging_root
                    or not self._staging_entry_belongs_to_project(entry, identifier)
                ):
                    continue
                shutil.rmtree(entry)
                staging_removed += 1

        catalog_removed = self._remove_cache_tree(projects_root, project_cache)
        self.identities.remove_project(current.id)
        self._external_source_indexes.clear()
        return ProjectDeletionResult(
            project_id=identifier,
            catalog_cache_removed=catalog_removed,
            thumbnail_cache_removed=thumbnail_removed,
            staging_directories_removed=staging_removed,
        )
