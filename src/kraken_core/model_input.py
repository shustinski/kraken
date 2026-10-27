"""Dependency-free staging contract for a pinned auxiliary model artifact."""

from dataclasses import dataclass
from collections.abc import Mapping
from uuid import UUID

from .plugin_protocol import safe_relative_path


@dataclass(frozen=True, slots=True)
class PinnedModelInput:
    version_id: str
    sha256: str
    filename: str

    @property
    def relative_path(self):
        return "inputs/model/" + self.filename

    @classmethod
    def from_parameters(cls, parameters: Mapping):
        if "model_version_id" not in parameters:
            return None
        identifier = str(UUID(str(parameters["model_version_id"])))
        digest = parameters.get("model_sha256")
        filename = parameters.get("model_filename")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Pinned model requires an authoritative SHA-256")
        if (
            not isinstance(filename, str)
            or safe_relative_path(filename) != filename
            or "/" in filename
            or any(c in filename for c in '<>"|?*')
        ):
            raise ValueError("Pinned model filename must be a safe basename")
        reference = cls(identifier, digest, filename)
        if "model_source_path" in parameters:
            raise ValueError("Pinned models cannot declare filesystem source paths")
        if parameters.get("model_relative_path", reference.relative_path) != reference.relative_path:
            raise ValueError("Pinned model path does not match its managed staging path")
        return reference
