"""File and authenticated server streams for immutable plugin inputs."""

from __future__ import annotations

import hashlib
import io
import os
import urllib.request
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Protocol
from urllib.parse import quote, urlparse

from .plugin_protocol import PluginFrameInput
from .safe_files import contained_path, open_regular_read


class AssetSource(Protocol):
    def open(self, relative_path: str) -> AbstractContextManager[BinaryIO]:
        """Return a context manager yielding a readable, seekable stream."""
        ...


@dataclass(frozen=True)
class FileAssetSource:
    root: Path
    inputs: Mapping[str, PluginFrameInput]

    @contextmanager
    def open(self, relative_path: str) -> Iterator[BinaryIO]:
        item = self.inputs[relative_path]
        path = contained_path(self.root, tuple(Path(item.relative_path).parts))
        with open_regular_read(path, root=self.root) as stream:
            digest = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != item.sha256:
                raise ValueError(f"Input checksum mismatch: {relative_path}")
            stream.seek(0)
            yield stream


@dataclass(frozen=True)
class ServerAssetSource:
    base_url: str
    job_id: str
    inputs: Mapping[str, PluginFrameInput]
    token: str = field(repr=False)
    timeout: float = 60
    maximum_input_bytes: int = 512 * 1024 * 1024
    lease_claim: str | None = field(default_factory=lambda: os.environ.get("KRAKEN_ASSET_LEASE_CLAIM"), repr=False)

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Server asset streams require an HTTP or HTTPS server URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Server asset URL must not contain credentials, query or fragment")
        if not self.token or not self.job_id or self.maximum_input_bytes <= 0:
            raise ValueError("Server asset access requires a token, job and positive size limit")

    @contextmanager
    def open(self, relative_path: str) -> Iterator[BinaryIO]:
        item = self.inputs[relative_path]
        version_id = item.artifact_version_id
        expected_sha256 = item.sha256
        if not version_id or not expected_sha256:
            raise ValueError("Server input requires an artifact version and checksum")
        endpoint = (
            f"{self.base_url.rstrip('/')}/api/v1/agent/jobs/{quote(self.job_id, safe='')}"
            f"/inputs/{quote(version_id, safe='')}"
        )
        headers = {"Authorization": f"Bearer {self.token}"}
        if self.lease_claim:
            headers["X-Kraken-Lease-Claim"] = self.lease_claim
        request = urllib.request.Request(endpoint, headers=headers)
        digest = hashlib.sha256()
        # Image decoders and archive readers need seek; this is a temporary
        # memory cache of one verified input, never a local project directory.
        with io.BytesIO() as cached:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                while chunk := response.read(1024 * 1024):
                    if cached.tell() + len(chunk) > self.maximum_input_bytes:
                        raise ValueError(f"Input exceeds stream size limit: {relative_path}")
                    digest.update(chunk)
                    cached.write(chunk)
            if digest.hexdigest() != expected_sha256:
                raise ValueError(f"Input checksum mismatch: {relative_path}")
            cached.seek(0)
            yield cached
