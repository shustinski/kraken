"""Managed remote cache must recover from corruption of unchanged byte length."""
import hashlib
from types import SimpleNamespace

import pytest

from kraken_hub.infrastructure.remote.parts.create_review_batch import CreateReviewBatchMixin
from kraken_manager.domain.artifacts import BlobRef


class Cache(CreateReviewBatchMixin):
    def __init__(self, root):
        self.data_dir = root
        self.expected = b"verified immutable source"
        self.downloaded = self.expected
        self.downloads = 0

    def artifact_version(self, project_id, version_id):
        return SimpleNamespace(id=version_id, blob=BlobRef(hashlib.sha256(self.expected).hexdigest(), len(self.expected)))

    def _download_managed_artifact(self, project_id, version_id, target):
        self.downloads += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.downloaded)


def test_same_size_corrupt_cache_is_replaced_only_by_verified_download(tmp_path):
    cache = Cache(tmp_path)
    target = cache.managed_artifact_path("project", "version")
    assert target.read_bytes() == cache.expected and cache.downloads == 1
    assert cache.managed_artifact_path("project", "version") == target and cache.downloads == 1
    target.write_bytes(b"x" * len(cache.expected))
    assert cache.managed_artifact_path("project", "version") == target
    assert target.read_bytes() == cache.expected and cache.downloads == 2


def test_corrupt_redownload_cannot_be_reused_as_a_valid_cache(tmp_path):
    cache = Cache(tmp_path)
    target = cache.managed_artifact_path("project", "version")
    target.write_bytes(b"x" * len(cache.expected))
    cache.downloaded = b"y" * len(cache.expected)
    with pytest.raises(ValueError):
        cache.managed_artifact_path("project", "version")
    assert not target.exists()
    assert not list(target.parent.glob("*.part-*"))
    cache.downloaded = cache.expected
    assert cache.managed_artifact_path("project", "version").read_bytes() == cache.expected
