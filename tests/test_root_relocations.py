import json
from pathlib import Path

import pytest

from kraken_manager.infrastructure.root_relocations import RootRelocations, SCHEMA


def test_nested_recovery_namespace_is_idempotent_and_confined(tmp_path):
    old = tmp_path / "old"
    current = old / "restored" / "source"
    current.mkdir(parents=True)
    aliases = RootRelocations([dict(original=str(old), target=str(current))], allowed_roots=[current])
    assert aliases.resolve(old / "project" / "image.jpg") == current / "project" / "image.jpg"
    assert aliases.resolve(current / "project" / "image.jpg") == current / "project" / "image.jpg"
    with pytest.raises(ValueError):
        aliases.resolve(old / ".." / "outside")
    with pytest.raises(ValueError):
        aliases.resolve(current / ".." / "outside")


def test_windows_historical_paths_can_resolve_on_either_platform(tmp_path):
    aliases = RootRelocations([dict(original=r"C:\OLD\root", target=str(tmp_path))], allowed_roots=[tmp_path])
    assert aliases.resolve(r"c:\old\ROOT\images\0000.jpg") == tmp_path / "images" / "0000.jpg"
    assert aliases.resolve("unrelated-relative") == Path("unrelated-relative")
    with pytest.raises(ValueError):
        aliases.resolve(r"C:\old\root\images\..\other")


def test_unmapped_historic_location_is_not_redirected(tmp_path):
    aliases = RootRelocations([dict(original=r"C:\old", target=str(tmp_path))], allowed_roots=[tmp_path])
    assert aliases.resolve(r"C:\old-other\file") == Path(r"C:\old-other\file")
    with pytest.raises(ValueError, match="outside"):
        RootRelocations([dict(original=r"C:\old", target=str(tmp_path.parent))], allowed_roots=[tmp_path])


def test_required_metadata_cannot_silently_fall_back_to_old_paths(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        RootRelocations.load(tmp_path, [tmp_path], required=True)
    directory = tmp_path / ".server"
    directory.mkdir()
    (directory / "recovery-roots.json").write_text(json.dumps(dict(schema=SCHEMA, mappings=[
        dict(original=r"C:\old", target=str(tmp_path)), dict(original=r"c:\OLD", target=str(tmp_path)),
    ])), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        RootRelocations.load(tmp_path, [tmp_path], required=True)
