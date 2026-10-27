"""Resolve managed model ports from the existing immutable artifact catalog."""
from __future__ import annotations

from kraken_manager.application.errors import ConflictError
from kraken_manager.domain.artifacts import ArtifactScope, is_published_model_artifact
from kraken_manager.domain.common import validate_uuid


def resolve_model_input(projections, project_id, layer_id, parameters):
    identifier = validate_uuid(str(parameters.get("model_version_id", "")), field="model_version_id")
    version = projections.get_artifact_version(identifier)
    series = None if version is None else projections.get_artifact_series(version.series_id)
    if (version is None or series is None or series.archived
            or str(series.project_id) != str(project_id) or str(series.layer_id) != str(layer_id)
            or series.scope is not ArtifactScope.LAYER_ATTACHMENT or not is_published_model_artifact(version)):
        raise ConflictError("Model input must be a published managed model attachment in this layer")
    filename = version.filename
    if (any(ord(char) < 32 or char in '/\\:<>"|?*' for char in filename)
            or filename.endswith((".", " "))
            or filename.split(".", 1)[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}):
        raise ConflictError("Model filename must be a safe plain filename")
    if "model_source_path" in parameters or "model_relative_path" in parameters:
        raise ConflictError("Managed model paths are resolved by Kraken staging")
    for key, expected in (("model_sha256", version.sha256), ("model_filename", filename)):
        if key in parameters and parameters[key] != expected:
            raise ConflictError(f"{key} does not match the published model")
    metadata = {"model_version_id": identifier, "model_sha256": version.sha256, "model_filename": filename,
                "model_version": parameters.get("model_version") or identifier}
    return version, series, metadata
