from __future__ import annotations

import io
import stat
import zipfile

import pytest

from kraken_manager.infrastructure.review.package import ReviewPackageLimits
from kraken_manager.infrastructure.work_package import extract_dataset_archive


def archive(*entries):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as stream:
        for name, content in entries:
            stream.writestr(name, content)
    return data.getvalue()


@pytest.mark.parametrize("name", ["../escape", "C:escape", "images/a.png:secret", "images/CON.png", "images/a. ", "images/../a"])
def test_dataset_validates_every_path_before_creating_files(tmp_path, name):
    target = tmp_path / "dataset"
    with pytest.raises(ValueError, match="небезопасный"):
        extract_dataset_archive(archive(("images/good.png", b"good"), (name, b"bad")), target)
    assert not target.exists()
    assert not list(tmp_path.iterdir())


def test_dataset_rejects_symbolic_links_and_case_collisions(tmp_path):
    link = zipfile.ZipInfo("images/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    for content in (archive((link, b"../../outside")), archive(("images/A.png", b"a"), ("images/a.png", b"b"))):
        with pytest.raises(ValueError):
            extract_dataset_archive(content, tmp_path / "dataset")
        assert not list(tmp_path.iterdir())


def test_dataset_limits_and_exclusive_atomic_publication(tmp_path):
    content = archive(("images/a.png", b"image"), ("cif/a.cif", b"vector"))
    target = tmp_path / "dataset"
    with pytest.raises(ValueError):
        extract_dataset_archive(content, target, limits=ReviewPackageLimits(max_total_bytes=5))
    assert not target.exists()
    extract_dataset_archive(content, target)
    assert (target / "images/a.png").read_bytes() == b"image"
    assert (target / "cif/a.cif").read_bytes() == b"vector"
    with pytest.raises(FileExistsError):
        extract_dataset_archive(archive(("images/a.png", b"replacement")), target)
    assert (target / "images/a.png").read_bytes() == b"image"
    assert not list(tmp_path.glob(".dataset-*"))
