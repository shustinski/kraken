"""Materialize returned CIF files without changing domain import approval."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from uuid import uuid4

from kraken_core.safe_files import contained_path, make_contained_directories, open_regular_read
from kraken_manager.application.ports import PluginResultContentReader
from kraken_manager.domain.workflows import PluginFrameOutcome, PluginFrameResultV1, PluginJob, PluginResultManifestV1
from kraken_manager.infrastructure.workspace_files import WorkspaceFileService
from kraken_manager.workspace import WorkspaceValidationError, validate_workspace_name


def _verify_existing_outputs(destination: Path, outputs: dict[str, PluginFrameResultV1]) -> None:
    if {path.name for path in destination.iterdir()} != set(outputs):
        raise WorkspaceValidationError("The existing vector result folder has different files")
    for name, output in outputs.items():
        with open_regular_read(contained_path(destination, (name,)), root=destination) as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != output.sha256:
                raise WorkspaceValidationError(f"The existing CIF was changed: {name}")


def materialize_vector_result(
    workspace_files: WorkspaceFileService | None,
    *,
    job: PluginJob,
    layer_name: str,
    manifest: PluginResultManifestV1,
    reader: PluginResultContentReader,
) -> Path | None:
    """Keep each job's verified outputs in its own immutable workspace folder."""
    if workspace_files is None or job.capability != "frames.vectorize.v1":
        return None
    binding = workspace_files.registry.get_project(str(job.project_id))
    if binding is None:
        return None
    outputs = {
        validate_workspace_name(Path(item.relative_path).name, field_name="Имя CIF"): item
        for item in manifest.results
        if item.outcome is PluginFrameOutcome.SUCCEEDED
        and item.relative_path is not None
        and item.role == "vector"
        and item.media_type == "application/x-cif"
    }
    if not outputs:
        return None
    if len(outputs) != sum(item.outcome is PluginFrameOutcome.SUCCEEDED for item in manifest.results):
        raise WorkspaceValidationError("Returned CIF filenames must be unique and have the vector format")
    with workspace_files.registry.mutation_lock(str(job.project_id)):
        if workspace_files.registry.plugin_result_removed(str(job.project_id), str(job.id)):
            return None
        parent = make_contained_directories(
            binding.derived_project_dir,
            ("vector", validate_workspace_name(layer_name, field_name="Название слоя")),
        )
        directories = workspace_files.registry.plugin_result_directories(str(job.project_id))
        saved = directories.get(str(job.id))
        if saved is not None:
            destination = contained_path(binding.derived_project_dir, Path(saved).parts)
        else:
            label = "Contour — " + job.created_at.astimezone().strftime("%d.%m.%Y %H-%M-%S")
            destination = contained_path(parent, (label,))
            index = 2
            while (
                destination.exists()
                or str(destination.relative_to(binding.derived_project_dir)) in directories.values()
            ):
                destination = contained_path(parent, (f"{label} ({index})",))
                index += 1
            workspace_files.registry.save_plugin_result_directory(
                str(job.project_id),
                str(job.id),
                str(destination.relative_to(binding.derived_project_dir)),
            )
        legacy = contained_path(parent, (str(job.id),))
        if legacy.exists() and not destination.exists():
            _verify_existing_outputs(legacy, outputs)
            os.rename(legacy, destination)
        if destination.exists():
            _verify_existing_outputs(destination, outputs)
            return destination
        # A normal project directory inherits its ACL. Windows mkdtemp uses
        # a private ACL that would otherwise follow the published directory.
        stage = contained_path(parent, (f".vector-{uuid4().hex}",))
        stage.mkdir()
        try:
            for name, output in outputs.items():
                digest = hashlib.sha256()
                with (stage / name).open("xb") as stream:
                    for chunk in reader.iter_output(manifest, output.relative_path):
                        digest.update(chunk)
                        stream.write(chunk)
                if digest.hexdigest() != output.sha256:
                    raise WorkspaceValidationError(f"Returned CIF has a bad SHA-256: {name}")
            os.rename(stage, destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return destination


def remove_vector_result(
    workspace_files: WorkspaceFileService | None,
    *,
    project_id: str,
    job_id: str,
) -> None:
    """Delete only the recorded per-job folder; block delayed publication."""
    if workspace_files is None:
        return
    with workspace_files.registry.mutation_lock(project_id):
        binding = workspace_files.registry.get_project(project_id)
        if binding is None:
            return
        directories = workspace_files.registry.plugin_result_directories(project_id)
        relative = directories.get(job_id)
        destination = None
        if relative is not None:
            parts = Path(relative).parts
            if len(parts) != 3 or parts[0] != "vector":
                raise WorkspaceValidationError("Only an individual vector result folder can be deleted")
            destination = contained_path(binding.derived_project_dir, parts)
            if sum(value == relative for value in directories.values()) != 1:
                raise WorkspaceValidationError("The vector result folder belongs to multiple jobs")
            if destination.exists() and not destination.is_dir():
                raise WorkspaceValidationError("The vector result path is not a directory")
        workspace_files.registry.mark_plugin_result_removed(project_id, job_id)
        if destination is not None and destination.exists():
            shutil.rmtree(destination)
        workspace_files.registry.forget_plugin_result_directory(project_id, job_id)
