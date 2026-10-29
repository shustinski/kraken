"""Resolve historic filesystem locators within explicitly restored storage roots.

Canonical events and artifact provenance keep their original paths. Only the
physical I/O location changes, and every destination stays inside configured
roots with the normal no-reparse checks.
"""
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

from kraken_core.safe_files import contained_path, ensure_regular_directory, open_regular_read

SCHEMA = "kraken-restored-roots/v1"
RELATIVE_FILE = (".server", "recovery-roots.json")


def original_path(value):
    cls = PureWindowsPath if re.match(r"^(?:[A-Za-z]:|\\\\)", str(value)) else PurePosixPath
    path = cls(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Relocation requires a canonical absolute source path")
    return path


class RootRelocations:
    def __init__(self, mappings=(), *, allowed_roots=()):
        self.allowed = tuple(ensure_regular_directory(root) for root in allowed_roots if root is not None)
        self.mappings = []
        if len(mappings) > 10000:
            raise ValueError("Too many storage root relocations")
        seen = set()
        for item in mappings:
            source = original_path(item["original"])
            if source in seen:
                raise ValueError("Duplicate source root relocation")
            seen.add(source)
            target = Path(item["target"]).absolute()
            target = self._contained(target)
            self.mappings.append((source, target))
        self.mappings.sort(key=lambda pair: len(pair[0].parts), reverse=True)

    def _contained(self, path):
        for root in self.allowed:
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            return contained_path(root, relative.parts)
        raise ValueError("Relocated path is outside configured storage roots")

    @classmethod
    def load(cls, blob_root, allowed_roots, *, required=False):
        root = ensure_regular_directory(blob_root)
        path = contained_path(root, RELATIVE_FILE)
        if not path.exists():
            if required:
                raise ValueError("Required storage relocation metadata is missing")
            return cls()
        with open_regular_read(path, root=root) as stream:
            encoded = stream.read(1024 * 1024 + 1)
        if len(encoded) > 1024 * 1024:
            raise ValueError("Storage relocation metadata exceeds size limit")
        value = json.loads(encoded)
        if not isinstance(value, dict) or value.get("schema") != SCHEMA or not isinstance(value.get("mappings"), list):
            raise ValueError("Unsupported storage relocation metadata")
        return cls(value["mappings"], allowed_roots=allowed_roots)

    def resolve(self, value):
        text = str(value)
        current = Path(value)
        if current.is_absolute():
            for root in self.allowed:
                if current.is_relative_to(root):
                    return self._contained(current)
        for source, target in self.mappings:
            try:
                candidate = type(source)(text)
                relative = candidate.relative_to(source)
            except ValueError:
                continue
            if ".." in relative.parts:
                raise ValueError("Traversal in relocated path")
            return self._contained(target.joinpath(*relative.parts))
        return Path(value)
