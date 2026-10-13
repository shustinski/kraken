"""Seekable in-memory inputs from a verified Kraken dataset archive."""

from __future__ import annotations

import io
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from zipfile import ZipFile
from PIL import Image

from kraken_core.asset_source import AssetSource
from kraken_core.plugin_protocol import safe_relative_path
from neuralimage.lib.backend import cif_to_jpg, find_size, make_lines_splitted

_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
_MAX_IMAGE_PIXELS = 64 * 1024 * 1024


@dataclass(frozen=True)
class DatasetStreamSource:
    image_root: Path
    label_root: Path
    sample_pairs: tuple[tuple[Path, Path], ...]
    contents: dict[str, bytes] = field(repr=False)

    @classmethod
    def load(cls, source: AssetSource, archive_key: str, logical_root: Path) -> DatasetStreamSource:
        images = logical_root / "images"
        labels = logical_root / "labels"
        contents: dict[str, bytes] = {}
        pairs = []
        with source.open(archive_key) as data, ZipFile(data) as archive:
            seen: set[str] = set()
            members = archive.infolist()
            if sum(item.file_size for item in members) > _MAX_ARCHIVE_BYTES:
                raise ValueError("Dataset exceeds the temporary memory cache limit")
            for item in members:
                name = safe_relative_path(item.filename.rstrip("/"))
                if name.casefold() in seen or stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("Dataset contains duplicate names or symbolic links")
                seen.add(name.casefold())
            image_members = [
                item
                for item in members
                if not item.is_dir() and PurePosixPath(item.filename).parent == PurePosixPath("images")
            ]
            archive_names = set(archive.namelist())
            for item in image_members:
                filename = PurePosixPath(item.filename).name
                label_name = f"cif/{PurePosixPath(filename).stem}.cif"
                if label_name not in archive_names:
                    raise ValueError(f"Dataset is missing a CIF label for {filename}")
                image_path = images / filename
                label_path = labels / f"{Path(filename).stem}.png"
                if str(label_path) in contents:
                    raise ValueError("Dataset contains ambiguous image stems")
                image_bytes = archive.read(item)
                label_text = archive.read(label_name).decode("utf-8")
                found, size = find_size(make_lines_splitted(label_text.splitlines(), " "))
                with Image.open(io.BytesIO(image_bytes)) as image:
                    if not found or tuple(size) != image.size:
                        raise ValueError(f"CIF dimensions do not match image {filename}")
                    if image.width * image.height > _MAX_IMAGE_PIXELS:
                        raise ValueError("Dataset image exceeds the decoded pixel limit")
                mask = cif_to_jpg(io.StringIO(label_text))
                if isinstance(mask, tuple):
                    raise ValueError(f"Invalid CIF label {label_name}: {mask[1]}")
                with io.BytesIO() as encoded:
                    mask.convert("L").save(encoded, "PNG")
                    contents[str(label_path)] = encoded.getvalue()
                contents[str(image_path)] = image_bytes
                pairs.append((image_path, label_path))
        if not pairs:
            raise ValueError("Dataset has no paired images and CIF labels")
        return cls(images, labels, tuple(pairs), contents)

    @contextmanager
    def open(self, relative_path: str) -> Iterator[BinaryIO]:
        with io.BytesIO(self.contents[str(Path(relative_path))]) as stream:
            yield stream
