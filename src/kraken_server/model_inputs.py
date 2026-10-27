"""Authorize auxiliary model downloads using the existing artifact service."""

from kraken_manager.application.model_inputs import resolve_model_input
from kraken_manager.domain.artifacts import ArtifactSeries, ArtifactVersion
from kraken_manager.infrastructure.filesystem._codec import decode_model, encode_model


class _InputProjection:
    def __init__(self, backend, manifest):
        self.backend, self.manifest = backend, manifest

    def get_artifact_version(self, identifier):
        return decode_model(
            ArtifactVersion, self.backend.get_artifact_version(str(self.manifest.project_id), str(identifier))
        )

    def get_artifact_series(self, identifier):
        values = self.backend.list_artifact_series(str(self.manifest.project_id), layer_id=str(self.manifest.layer_id))
        value = next((value for value in values if str(value["id"]) == str(identifier)), None)
        return None if value is None else decode_model(ArtifactSeries, value)


def model_input_metadata(backend, manifest):
    version, _series, _metadata = resolve_model_input(
        _InputProjection(backend, manifest), manifest.project_id, manifest.layer_id, manifest.parameters
    )
    return encode_model(version)
