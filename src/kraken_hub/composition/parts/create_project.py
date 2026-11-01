"""composition: create project."""
from __future__ import annotations
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from kraken_core.safe_files import ensure_regular_directory
from kraken_manager.application.dto import (
    CommandContext,
    CreateProjectCommand,
)
from kraken_manager.domain.common import ProjectId
from kraken_manager.workspace import (
    project_workspace_to_dict,
    ProjectWorkspaceBinding,
    validate_workspace_name,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.project import (
    GridOrientation,
    LayerType,
    Project,
)
from kraken_manager.infrastructure.filesystem.event_store import FilesystemEventStore


def project_id_for_idempotency_key(principal_id: object, idempotency_key: str) -> str:
    """Stable project id shared by the local catalog and the PostgreSQL server."""

    return str(uuid5(NAMESPACE_URL, f"kraken:project:{principal_id}:{idempotency_key}"))


class CreateProjectMixin:
    def create_project(
        self,
        *,
        principal: Principal,
        name: str,
        width: int,
        height: int,
        orientation: GridOrientation,
        idempotency_key: str,
        layer_template: bool = False,
        source_root: Path | str | None = None,
        derived_root: Path | str | None = None,
        storage_profile_id: str | None = None,
    ) -> Project:
        if storage_profile_id not in {None, self.profile.id}:
            raise ValueError(
                f"Local catalog only supports storage profile {self.profile.id!r}"
            )
        safe_name = validate_workspace_name(name, field_name="Название проекта")
        project_id = project_id_for_idempotency_key(principal.id, idempotency_key)
        command = CreateProjectCommand(
            context=CommandContext(actor=principal, idempotency_key=idempotency_key),
            name=safe_name,
            width=width,
            height=height,
            orientation=orientation,
            storage_profile_id=self.profile.id,
            project_id=ProjectId(project_id),
        )
        with self.workspace_registry.mutation_lock(project_id):
            project = self._recorded_project(principal, idempotency_key, project_id)
            if project is None:
                binding = self.workspace_registry.get_project(project_id)
                recovering = binding is not None
                if recovering:
                    self._validate_uncommitted_workspace(
                        binding, safe_name, source_root or self.default_source_root,
                        derived_root or self.default_derived_root,
                    )
                else:
                    binding = self.workspace_files.create_project(
                        project_id=project_id,
                        project_name=safe_name,
                        source_root=source_root or self.default_source_root,
                        derived_root=derived_root or self.default_derived_root,
                    )
                try:
                    project = self._commands(project_id).execute(command)
                except Exception:
                    if not recovering:
                        self.workspace_files.remove_project_layout(binding)
                        self.workspace_registry.remove_project(project_id)
                    raise
            else:
                binding = self.workspace_registry.get_project(str(project.id))
            self._record_workspace_binding_once(
                principal=principal,
                project=project,
                binding=binding,
                idempotency_key=f"{idempotency_key}:workspace",
            )
        if layer_template:
            for order, layer_type in enumerate(LayerType, start=1):
                self.create_layer(
                    principal=principal,
                    project=project,
                    name=layer_type.value.capitalize(),
                    layer_type=layer_type,
                    order=order,
                    idempotency_key=f"{idempotency_key}:layer:{layer_type.value}",
                )
                project = self.get_project(project.id) or project
        return self.get_project(project.id) or project

    def _record_workspace_binding_once(
        self,
        *,
        principal: Principal,
        project: Project,
        binding: object,
        idempotency_key: str,
    ) -> None:
        """Append the workspace event unless this key is already in the journal.

        Caller holds the project mutation lock, so a second create cannot pass
        the check before the first append finishes.
        """

        if binding is None:
            return
        store = FilesystemEventStore(self.catalog_root, str(project.id))
        if store.find_by_idempotency_key(project.id, idempotency_key):
            return
        self._record_workspace_event(
            principal=principal,
            project_id=project.id,
            stream_id=f"project-workspace:{project.id}",
            event_type="ProjectWorkspaceBoundV1",
            payload=project_workspace_to_dict(binding),
            idempotency_key=idempotency_key,
        )

    def _recorded_project(self, principal: Principal, idempotency_key: str, stable_id: str) -> Project | None:
        """Return the project already stored for this account and key.

        New creates use ``stable_id``. Catalogs written before that id still
        keep the key on their original ``ProjectCreated`` event.
        """

        current = self.get_project(stable_id)
        if current is not None:
            return current
        root = self.catalog_root / "projects"
        if not root.is_dir():
            return None
        for directory in root.iterdir():
            if not directory.is_dir() or directory.name == stable_id:
                continue
            try:
                store = FilesystemEventStore(self.catalog_root, directory.name)
                events = store.find_by_idempotency_key(directory.name, idempotency_key)
            except (OSError, ValueError):
                continue
            if not any(
                event.event_type == "ProjectCreated"
                and str(getattr(getattr(event, "actor", None), "principal_id", "")) == str(principal.id)
                for event in events
            ):
                continue
            recorded = self.get_project(directory.name)
            if recorded is not None:
                return recorded
        return None

    @staticmethod
    def _validate_uncommitted_workspace(
        binding: ProjectWorkspaceBinding, name: str, source_root: Path | str, derived_root: Path | str,
    ) -> None:
        """Reuse an interrupted create without deleting files added before retry."""
        source = Path(source_root).expanduser().resolve(strict=True)
        derived = Path(derived_root).expanduser().resolve(strict=True)
        if (
            binding.project_name != name
            or Path(binding.source_root) != source
            or Path(binding.derived_root) != derived
            or Path(binding.source_project_dir) != source / name
            or Path(binding.derived_project_dir) != derived / name
        ):
            raise ValueError("Uncommitted workspace differs from the requested project name or storage roots")
        ensure_regular_directory(binding.source_project_dir)
        ensure_regular_directory(binding.derived_project_dir)
