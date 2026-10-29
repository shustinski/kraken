"""composition: staging entry belongs project."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from kraken_manager.application.dto import (
    ArchiveProjectCommand,
    CommandContext,
    RestoreProjectCommand,
)
from kraken_manager.workspace import (
    WorkspaceValidationError,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.project import (
    Project,
)


class StagingEntryBelongsProjectMixin:
    def archive_project(
        self, *, principal: Principal, project: Project, idempotency_key: str
    ) -> Project:
        return self._commands(str(project.id)).execute(
            ArchiveProjectCommand(
                context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                project_id=project.id,
                expected_revision=project.revision,
            )
        )

    def restore_project(
        self, *, principal: Principal, project: Project, idempotency_key: str
    ) -> Project:
        return self._commands(str(project.id)).execute(
            RestoreProjectCommand(
                context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                project_id=project.id,
                expected_revision=project.revision,
            )
        )

    @staticmethod
    def _remove_cache_tree(root: Path, target: Path) -> bool:
        safe_root = root.resolve()
        safe_target = target.resolve()
        if safe_target.parent != safe_root:
            raise WorkspaceValidationError(
                f"Путь кэша проекта выходит за допустимые границы: {safe_target}"
            )
        if not safe_target.exists():
            return False
        if safe_target.is_symlink():
            raise WorkspaceValidationError(
                f"Кэш проекта не должен быть символической ссылкой: {safe_target}"
            )
        shutil.rmtree(safe_target)
        return True

    @staticmethod
    def _staging_entry_belongs_to_project(entry: Path, project_id: str) -> bool:
        if entry.name.startswith(f"{project_id}-"):
            return True
        inspected = 0
        try:
            manifests = entry.rglob("*.json")
            for manifest in manifests:
                inspected += 1
                if inspected > 128:
                    break
                try:
                    if not manifest.is_file() or manifest.stat().st_size > 2 * 1024 * 1024:
                        continue
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict) and str(payload.get("project_id", "")) == project_id:
                    return True
        except OSError:
            return False
        return False
