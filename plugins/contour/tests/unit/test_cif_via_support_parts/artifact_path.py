"""test_cif_via_support: artifact path."""
from __future__ import annotations
from pathlib import Path


class ArtifactPathMixin:
    def _artifact_path(self, name: str) -> Path:
        root = Path(".tmp-tests")
        root.mkdir(exist_ok=True)
        path = root / name
        if path.exists():
            path.unlink()
        return path
